"""Fast compatibility inspection for dense and quantized safetensors files."""

from __future__ import annotations

import json
from pathlib import Path

from .checkpoint_io import ConversionError, validate_checkpoint


def inspect_checkpoint(path: str | Path) -> str:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is missing from the ComfyUI environment.") from exc

    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = dict(handle.metadata() or {})
        weight_layers = {key[: -len(".weight")] for key in keys if key.endswith(".weight")}
        marker_layers = {
            key[: -len(".comfy_quant")] for key in keys if key.endswith(".comfy_quant")
        }
        orphan_markers = sorted(marker_layers - weight_layers)
        missing_markers: list[str] = []
        prefix_matches = 0
        ambiguous = 0

        raw_old = metadata.get("_quantization_metadata")
        old_layers: dict = {}
        if raw_old:
            try:
                parsed = json.loads(raw_old)
                old_layers = parsed.get("layers", {}) if isinstance(parsed, dict) else {}
            except (TypeError, json.JSONDecodeError):
                return f"INVALID: {path.name} has malformed _quantization_metadata JSON"

        for layer in old_layers:
            if layer in weight_layers:
                continue
            candidates = [physical for physical in weight_layers if physical.endswith(f".{layer}")]
            if len(candidates) == 1:
                prefix_matches += 1
            elif len(candidates) > 1:
                ambiguous += 1
            else:
                missing_markers.append(layer)

    if orphan_markers:
        return f"INVALID: {len(orphan_markers)} orphan .comfy_quant marker(s) in {path.name}"
    if marker_layers:
        try:
            validate_checkpoint(path, expected_quantized=len(marker_layers))
        except ConversionError as exc:
            return f"INVALID: {path.name}: {exc}"
        return (
            f"OK: {path.name} uses {len(marker_layers)} exact per-layer .comfy_quant marker(s); "
            "ComfyUI/Forge Neo packaging is compatible."
        )
    if old_layers and (prefix_matches or missing_markers or ambiguous):
        return (
            f"RISK: legacy central metadata has {prefix_matches} prefix mismatch(es), "
            f"{len(missing_markers)} missing and {ambiguous} ambiguous mapping(s)."
        )
    if old_layers:
        return f"LEGACY: {len(old_layers)} central metadata layer(s), with exact names."
    return f"DENSE/UNMARKED: no quantization markers found in {path.name}."
