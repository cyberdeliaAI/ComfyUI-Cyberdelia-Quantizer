"""Planning, conversion, atomic writing, and validation for safetensors files."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .profiles import get_profile
from .quant_formats import (
    FORMAT_SPECS,
    get_format_spec,
    layer_name_from_weight,
    marker_config,
    marker_key_for_weight,
    prepare_tensor_for_save,
    quantize_tensor,
    require_runtime,
    shape_compatibility,
    sidecar_key_for_weight,
)


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    format_name: str


@dataclass
class ConversionPlan:
    selected: list[PlannedTensor] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    incompatible: list[tuple[str, str]] = field(default_factory=list)
    copied: list[str] = field(default_factory=list)


@dataclass
class ConversionReport:
    output_path: str
    format_name: str
    profile_name: str
    quantized_count: int
    copied_count: int
    incompatible_count: int
    failed_count: int
    output_bytes: int = 0
    dry_run: bool = False
    warnings: list[str] = field(default_factory=list)

    def status_text(self) -> str:
        action = "Plan ready" if self.dry_run else "Conversion complete"
        size = ""
        if self.output_bytes:
            size = f" | {self.output_bytes / (1024**3):.2f} GiB"
        warning = ""
        if self.warnings:
            warning = f" | warnings: {len(self.warnings)}"
        return (
            f"{action}: {self.format_name} | quantized {self.quantized_count} | "
            f"copied {self.copied_count} | incompatible {self.incompatible_count} | "
            f"failed {self.failed_count}{size}{warning}"
        )


ProgressCallback = Callable[[int, int, str], None]
_DENSE_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
_QUANTIZED_FLOAT_DTYPES = {"F8_E4M3", "F8_E5M2", "F8_E4M3FN", "F8_E5M2FN"}
_QUANTIZED_STORAGE_DTYPES = _QUANTIZED_FLOAT_DTYPES | {"I8", "U8"}
_EXPECTED_STORAGE_DTYPES = {
    "float8_e4m3fn": {"F8_E4M3", "F8_E4M3FN"},
    "float8_e5m2": {"F8_E5M2", "F8_E5M2FN"},
    "nvfp4": {"U8"},
    "mxfp8": {"F8_E4M3", "F8_E4M3FN"},
    "int8_tensorwise": {"I8"},
    "convrot_w4a4": {"I8"},
    "asym_w4a8_int8": {"I8"},
}


def normalize_output_filename(source_name: str, requested_name: str, format_name: str) -> str:
    source_stem = Path(source_name).stem
    requested = (requested_name or "").strip()
    base = Path(requested).name if requested else source_stem
    if base.lower().endswith(".safetensors"):
        base = base[: -len(".safetensors")]
    if not base or base in {".", ".."}:
        raise ConversionError("The output filename is empty or invalid.")
    suffix = get_format_spec(format_name).suffix
    if not base.lower().endswith(f"_{suffix}".lower()):
        base = f"{base}_{suffix}"
    return f"{base}.safetensors"


def resolve_output_path(input_path: Path, requested_name: str, format_name: str) -> Path:
    filename = normalize_output_filename(input_path.name, requested_name, format_name)
    return input_path.parent / filename


def _slice_info(handle, key: str) -> tuple[tuple[int, ...], str]:
    tensor_slice = handle.get_slice(key)
    return tuple(int(value) for value in tensor_slice.get_shape()), str(tensor_slice.get_dtype())


def _parse_old_quant_metadata(metadata: dict[str, str]) -> dict:
    raw = metadata.get("_quantization_metadata")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ConversionError("The source contains invalid _quantization_metadata JSON.") from exc
    return parsed if isinstance(parsed, dict) else {}


def build_plan(input_path: Path, profile_name: str, format_name: str) -> tuple[ConversionPlan, dict[str, str]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ConversionError("safetensors is missing from the ComfyUI environment.") from exc

    profile = get_profile(profile_name)
    plan = ConversionPlan()
    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = sorted(handle.keys())
        key_set = set(keys)

        physical_markers = [key for key in keys if key.endswith(".comfy_quant")]
        old_quant = _parse_old_quant_metadata(metadata)
        if physical_markers or old_quant.get("layers"):
            raise ConversionError(
                "The source is already quantized. Use an original FP16/BF16 checkpoint; "
                "re-quantizing an FP8/INT8/FP4 file compounds quality loss."
            )

        for key in keys:
            shape, dtype = _slice_info(handle, key)
            if not key.endswith(".weight") or len(shape) != 2:
                plan.copied.append(key)
                continue
            if dtype in _QUANTIZED_STORAGE_DTYPES:
                raise ConversionError(
                    f"{key} is already stored as {dtype}; use the original FP16/BF16 checkpoint."
                )
            if profile.excludes(key):
                plan.excluded.append(key)
                continue
            if dtype not in _DENSE_FLOAT_DTYPES:
                plan.copied.append(key)
                continue
            tensor_format = profile.format_for(key, format_name)
            compatible, reason = shape_compatibility(tensor_format, shape)
            if not compatible:
                plan.incompatible.append((key, reason or "unsupported shape"))
                continue
            layer = layer_name_from_weight(key)
            reserved = {
                f"{layer}.{sidecar}" for sidecar in get_format_spec(tensor_format).required_sidecars
            }
            collisions = sorted(reserved & key_set)
            if collisions:
                raise ConversionError(
                    f"The source already contains quantization side tensors for {layer}: "
                    + ", ".join(collisions)
                )
            plan.selected.append(PlannedTensor(key, shape, dtype, tensor_format))

    if not plan.selected:
        raise ConversionError(
            "No eligible Linear weights were found. Check the model profile and source checkpoint."
        )
    return plan, metadata


def _json_marker_tensor(torch, config: dict):
    payload = json.dumps(config, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device="cpu")


def _save_key_for_suffix(weight_name: str, suffix: str) -> str:
    layer = layer_name_from_weight(weight_name)
    return f"{layer}.weight{suffix}"


def _clean_metadata(source_metadata: dict[str, str], report: ConversionReport) -> dict[str, str]:
    metadata = {str(key): str(value) for key, value in source_metadata.items()}
    metadata.pop("_quantization_metadata", None)
    metadata.update(
        {
            "converted_by": "ComfyUI-Cyberdelia-Quantizer",
            "converter_version": "1.0.0",
            "quant_format": report.format_name,
            "quant_profile": report.profile_name,
            "quantized_tensor_count": str(report.quantized_count),
            "quantization_metadata_layout": "per-layer-comfy_quant",
        }
    )
    return metadata


def validate_checkpoint(
    path: Path,
    expected_quantized: int | None = None,
    expected_markers: dict[str, dict] | None = None,
) -> dict[str, int]:
    """Validate marker-to-weight coupling without loading model weights into RAM."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ConversionError("safetensors is missing from the ComfyUI environment.") from exc

    marker_count = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = dict(handle.metadata() or {})
        if "_quantization_metadata" in metadata:
            raise ConversionError("Output unexpectedly contains legacy _quantization_metadata.")

        for marker_key in sorted(key for key in keys if key.endswith(".comfy_quant")):
            marker_count += 1
            layer = marker_key[: -len(".comfy_quant")]
            weight_key = f"{layer}.weight"
            if weight_key not in keys:
                raise ConversionError(f"Orphan quantization marker: {marker_key}")
            marker_tensor = handle.get_tensor(marker_key)
            try:
                config = json.loads(bytes(marker_tensor.tolist()).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                raise ConversionError(f"Invalid quantization marker: {marker_key}") from exc
            if expected_markers is not None:
                expected_config = expected_markers.get(layer)
                if expected_config is None:
                    raise ConversionError(f"Unexpected quantized layer in output: {layer}")
                if config != expected_config:
                    raise ConversionError(
                        f"Marker mismatch for {layer}: got {config}, expected {expected_config}"
                    )
            marker_format = config.get("format")
            matching_specs = [
                spec for spec in FORMAT_SPECS.values() if spec.marker_format == marker_format
            ]
            if not matching_specs:
                raise ConversionError(f"Unknown marker format {marker_format!r} in {marker_key}")
            stored_dtype = str(handle.get_slice(weight_key).get_dtype())
            expected_dtypes = _EXPECTED_STORAGE_DTYPES[marker_format]
            if stored_dtype not in expected_dtypes:
                raise ConversionError(
                    f"Wrong storage dtype for {weight_key}: {stored_dtype}; "
                    f"expected one of {sorted(expected_dtypes)}"
                )
            required_sets = [set(spec.required_sidecars) for spec in matching_specs]
            available_sidecars = {
                key[len(layer) + 1 :]
                for key in keys
                if key.startswith(f"{layer}.") and key != weight_key and key != marker_key
            }
            if not any(required <= available_sidecars for required in required_sets):
                expected = " or ".join(", ".join(sorted(required)) for required in required_sets)
                raise ConversionError(f"Missing side tensor(s) for {layer}; expected {expected}")

    if expected_quantized is not None and marker_count != expected_quantized:
        raise ConversionError(
            f"Validation found {marker_count} quantized layers; expected {expected_quantized}."
        )
    if expected_markers is not None and marker_count != len(expected_markers):
        raise ConversionError(
            f"Validation found {marker_count} marker(s); expected {len(expected_markers)}."
        )
    return {"marker_count": marker_count}


def convert_checkpoint(
    input_path: str | Path,
    requested_output_name: str,
    profile_name: str,
    format_name: str,
    device_name: str = "auto",
    strict: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
) -> ConversionReport:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ConversionError("safetensors is missing from the ComfyUI environment.") from exc

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file() or input_path.suffix.lower() != ".safetensors":
        raise ConversionError(f"Source checkpoint not found or not safetensors: {input_path}")

    output_path = resolve_output_path(input_path, requested_output_name, format_name).resolve()
    if output_path == input_path:
        raise ConversionError("The output must not overwrite the source checkpoint.")
    if output_path.exists() and not overwrite:
        raise ConversionError(f"Output already exists: {output_path.name}. Enable overwrite to replace it.")

    torch, _quant_ops, _layout = require_runtime(format_name)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ConversionError("CUDA was requested, but PyTorch cannot see a CUDA device.")
    device = torch.device(device_name)

    plan, source_metadata = build_plan(input_path, profile_name, format_name)
    report = ConversionReport(
        output_path=str(output_path),
        format_name=format_name,
        profile_name=profile_name,
        quantized_count=len(plan.selected) if dry_run else 0,
        copied_count=len(plan.copied) + len(plan.excluded) + len(plan.incompatible),
        incompatible_count=len(plan.incompatible),
        failed_count=0,
        dry_run=dry_run,
    )
    if plan.incompatible:
        preview = "; ".join(f"{name}: {reason}" for name, reason in plan.incompatible[:5])
        report.warnings.append(f"Shape-incompatible layers kept unchanged: {preview}")
    if dry_run:
        return report

    selected = {item.name: item for item in plan.selected}
    output_tensors = {}
    failed: list[str] = []
    failed_names: set[str] = set()
    total = len(selected)
    completed = 0

    try:
        with safe_open(str(input_path), framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                source_tensor = handle.get_tensor(key)
                planned = selected.get(key)
                if planned is None:
                    output_tensors[key] = source_tensor.detach().contiguous()
                    continue

                actual_shape = tuple(int(value) for value in source_tensor.shape)
                if actual_shape != planned.shape:
                    raise ConversionError(
                        f"Source changed during conversion: {key} was {planned.shape}, now {actual_shape}."
                    )
                try:
                    work_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                    work = source_tensor.to(device=device, dtype=work_dtype, non_blocking=device.type == "cuda")
                    quant_tensors, config = quantize_tensor(planned.format_name, work)
                    layer_output = {}
                    for suffix, tensor in quant_tensors.items():
                        save_key = _save_key_for_suffix(key, suffix)
                        layer_output[save_key] = prepare_tensor_for_save(torch, tensor)
                    if planned.format_name in {"FP8_E4M3", "FP8_E5M2"}:
                        layer_output[sidecar_key_for_weight(key, "input_scale")] = torch.ones(
                            (), dtype=torch.float32, device="cpu"
                        )
                    layer_output[marker_key_for_weight(key)] = _json_marker_tensor(torch, config)
                    output_tensors.update(layer_output)
                    report.quantized_count += 1
                    del work, quant_tensors, layer_output
                except Exception as exc:
                    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
                    if isinstance(exc, ConversionError) or (
                        cuda_oom is not None and isinstance(exc, cuda_oom)
                    ):
                        raise
                    if strict:
                        raise ConversionError(f"Quantization failed for {key}: {exc}") from exc
                    output_tensors[key] = source_tensor.detach().contiguous()
                    failed.append(f"{key}: {exc}")
                    failed_names.add(key)
                    report.failed_count += 1
                    report.copied_count += 1

                completed += 1
                if progress is not None:
                    progress(completed, total, key)

        if report.quantized_count == 0:
            raise ConversionError("No weights were quantized; no output was written.")
        if failed:
            report.warnings.append("Layers kept unchanged after errors: " + "; ".join(failed[:5]))

        metadata = _clean_metadata(source_metadata, report)
        expected_markers = {
            layer_name_from_weight(item.name): marker_config(item.format_name, item.shape)
            for item in plan.selected
            if item.name not in failed_names
        }
        temporary_path = output_path.with_name(
            f".{output_path.stem}.{uuid.uuid4().hex}.partial.safetensors"
        )
        try:
            save_file(output_tensors, str(temporary_path), metadata=metadata)
            validate_checkpoint(
                temporary_path,
                expected_quantized=report.quantized_count,
                expected_markers=expected_markers,
            )
            if output_path.exists() and not overwrite:
                raise ConversionError(f"Output appeared during conversion: {output_path.name}")
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    finally:
        output_tensors.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report.output_bytes = output_path.stat().st_size
    return report
