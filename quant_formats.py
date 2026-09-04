"""ComfyUI-native quantization format contracts and adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuantFormatSpec:
    name: str
    suffix: str
    marker_format: str
    required_sidecars: tuple[str, ...]
    minimum_input_multiple: int | None = None


FORMAT_SPECS: dict[str, QuantFormatSpec] = {
    "FP8_E4M3": QuantFormatSpec(
        "FP8_E4M3", "fp8_e4m3", "float8_e4m3fn", ("weight_scale", "input_scale")
    ),
    "FP8_E5M2": QuantFormatSpec(
        "FP8_E5M2", "fp8_e5m2", "float8_e5m2", ("weight_scale", "input_scale")
    ),
    "NVFP4": QuantFormatSpec(
        "NVFP4", "nvfp4", "nvfp4", ("weight_scale", "weight_scale_2")
    ),
    "MXFP8": QuantFormatSpec("MXFP8", "mxfp8", "mxfp8", ("weight_scale",)),
    "INT8": QuantFormatSpec("INT8", "int8", "int8_tensorwise", ("weight_scale",)),
    "INT8_CONVROT": QuantFormatSpec(
        "INT8_CONVROT", "int8_convrot", "int8_tensorwise", ("weight_scale",), 16
    ),
    "INT4_CONVROT": QuantFormatSpec(
        "INT4_CONVROT", "int4_convrot", "convrot_w4a4", ("weight_scale",), 64
    ),
    "W4A8_INT8": QuantFormatSpec(
        "W4A8_INT8",
        "w4a8_int8",
        "asym_w4a8_int8",
        ("weight_s_rel", "weight_s_channel", "weight_codebook"),
        16,
    ),
}

FORMAT_NAMES = tuple(FORMAT_SPECS)
CONVROT_GROUP_CANDIDATES = (256, 128, 64, 32, 16)


def get_format_spec(name: str) -> QuantFormatSpec:
    try:
        return FORMAT_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown quantization format: {name}") from exc


def layer_name_from_weight(tensor_name: str) -> str:
    """Return the exact physical layer name without changing any prefix."""
    suffix = ".weight"
    if not tensor_name.endswith(suffix):
        raise ValueError(f"Expected a *.weight tensor, got: {tensor_name}")
    return tensor_name[: -len(suffix)]


def marker_key_for_weight(tensor_name: str) -> str:
    return f"{layer_name_from_weight(tensor_name)}.comfy_quant"


def sidecar_key_for_weight(tensor_name: str, sidecar: str) -> str:
    if not sidecar or "." in sidecar:
        raise ValueError(f"Invalid sidecar name: {sidecar!r}")
    return f"{layer_name_from_weight(tensor_name)}.{sidecar}"


def choose_convrot_groupsize(input_features: int) -> int | None:
    for group_size in CONVROT_GROUP_CANDIDATES:
        if group_size <= input_features and input_features % group_size == 0:
            return group_size
    return None


def shape_compatibility(format_name: str, shape: tuple[int, ...]) -> tuple[bool, str | None]:
    if len(shape) != 2:
        return False, "only rank-2 Linear weights can be quantized"

    input_features = int(shape[1])
    if format_name in {"INT8_CONVROT", "INT4_CONVROT", "W4A8_INT8"}:
        convrot_group = choose_convrot_groupsize(input_features)
        if convrot_group is None:
            return False, f"input dimension {input_features} has no supported ConvRot group"

    required = get_format_spec(format_name).minimum_input_multiple
    if required is not None and input_features % required != 0:
        return False, f"input dimension {input_features} is not divisible by {required}"

    return True, None


def marker_config(format_name: str, shape: tuple[int, ...]) -> dict[str, Any]:
    spec = get_format_spec(format_name)
    marker: dict[str, Any] = {"format": spec.marker_format}

    if format_name in {"FP8_E4M3", "FP8_E5M2"}:
        # This is the official portable FP8 checkpoint behavior: FP8 storage,
        # but a full-precision matmul. It avoids hardware-specific FP8 failures.
        marker["full_precision_matrix_mult"] = True
    elif format_name == "INT8_CONVROT":
        marker["convrot"] = True
        marker["convrot_groupsize"] = choose_convrot_groupsize(int(shape[1]))
    elif format_name == "INT4_CONVROT":
        marker.update(
            {
                "convrot_groupsize": choose_convrot_groupsize(int(shape[1])),
                "quant_group_size": 64,
                "linear_dtype": "int4",
            }
        )
    elif format_name == "W4A8_INT8":
        marker.update(
            {
                "group_size": 16,
                "convrot_groupsize": choose_convrot_groupsize(int(shape[1])),
            }
        )
    return marker


def marker_json(format_name: str, shape: tuple[int, ...]) -> str:
    return json.dumps(marker_config(format_name, shape), separators=(",", ":"), sort_keys=True)


def require_runtime(format_name: str):
    """Return ``(torch, comfy.quant_ops, layout class)`` or raise a useful error."""
    try:
        import torch
        from comfy import quant_ops
    except ImportError as exc:
        raise RuntimeError(
            "This node needs a current ComfyUI installation with comfy-kitchen. "
            "Update ComfyUI and its bundled dependencies."
        ) from exc

    spec = get_format_spec(format_name)
    if spec.marker_format not in quant_ops.QUANT_ALGOS:
        raise RuntimeError(
            f"{format_name} is unavailable in this ComfyUI/comfy-kitchen build. "
            "Update ComfyUI and its requirements."
        )

    layout_name = {
        "FP8_E4M3": "TensorCoreFP8E4M3Layout",
        "FP8_E5M2": "TensorCoreFP8E5M2Layout",
        "NVFP4": "TensorCoreNVFP4Layout",
        "MXFP8": "TensorCoreMXFP8Layout",
        "INT8": "TensorWiseINT8Layout",
        "INT8_CONVROT": "TensorWiseINT8Layout",
        "INT4_CONVROT": "TensorCoreConvRotW4A4Layout",
        "W4A8_INT8": "AsymW4A8Int8Layout",
    }[format_name]
    layout = getattr(quant_ops, layout_name, None)
    if layout is None:
        raise RuntimeError(f"ComfyUI does not expose the required layout {layout_name}.")
    return torch, quant_ops, layout


def _safe_fp8_scale(torch: Any, tensor: Any, dtype: Any):
    amax = tensor.detach().abs().amax().to(dtype=torch.float32)
    scale = amax / float(torch.finfo(dtype).max)
    one = torch.ones((), device=tensor.device, dtype=torch.float32)
    return torch.where(amax > 0, scale.clamp(min=1.0e-12), one)


def _safe_nvfp4_scale(torch: Any, tensor: Any):
    amax = tensor.detach().abs().amax().to(dtype=torch.float32)
    scale = amax / (448.0 * 6.0)
    one = torch.ones((), device=tensor.device, dtype=torch.float32)
    return torch.where(amax > 0, scale.clamp(min=2.0**-126), one)


def quantize_tensor(format_name: str, tensor: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Quantize one Linear weight and return side tensors plus its marker config."""
    torch, _quant_ops, layout = require_runtime(format_name)
    shape = tuple(int(value) for value in tensor.shape)
    compatible, reason = shape_compatibility(format_name, shape)
    if not compatible:
        raise ValueError(reason)

    if format_name == "FP8_E4M3":
        scale = _safe_fp8_scale(torch, tensor, torch.float8_e4m3fn)
        qdata, params = layout.quantize(tensor, scale=scale)
    elif format_name == "FP8_E5M2":
        scale = _safe_fp8_scale(torch, tensor, torch.float8_e5m2)
        qdata, params = layout.quantize(tensor, scale=scale)
    elif format_name == "NVFP4":
        qdata, params = layout.quantize(tensor, scale=_safe_nvfp4_scale(torch, tensor))
    elif format_name == "MXFP8":
        qdata, params = layout.quantize(tensor)
    elif format_name == "INT8":
        qdata, params = layout.quantize(
            tensor,
            is_weight=True,
            per_channel=True,
            convrot=False,
            stochastic_rounding=0,
        )
    elif format_name == "INT8_CONVROT":
        qdata, params = layout.quantize(
            tensor,
            is_weight=True,
            per_channel=True,
            convrot=True,
            convrot_groupsize=choose_convrot_groupsize(shape[1]),
            stochastic_rounding=0,
        )
    elif format_name == "INT4_CONVROT":
        qdata, params = layout.quantize(
            tensor,
            convrot_groupsize=choose_convrot_groupsize(shape[1]),
            quant_group_size=64,
            stochastic_rounding=0,
            linear_dtype="int4",
        )
    elif format_name == "W4A8_INT8":
        qdata, params = layout.quantize(
            tensor,
            group_size=16,
            convrot_groupsize=choose_convrot_groupsize(shape[1]),
            symmetric=True,
            codebook=True,
            stochastic_rounding=0,
        )
    else:  # pragma: no cover - guarded by get_format_spec
        raise ValueError(f"Unsupported quantization format: {format_name}")

    tensors = dict(layout.state_dict_tensors(qdata, params))
    return tensors, marker_config(format_name, shape)


def prepare_tensor_for_save(torch: Any, tensor: Any):
    """Move a tensor to contiguous CPU storage without losing FP8 bit patterns."""
    tensor = tensor.detach().contiguous()
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    fp8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if dtype is not None
    )
    if e8m0 is not None and tensor.dtype == e8m0:
        # Safetensors does not consistently support E8M0 yet. ComfyUI's loader
        # explicitly accepts the uint8 bit view and restores the dtype.
        return tensor.view(torch.uint8).cpu().contiguous()
    if tensor.dtype in fp8_dtypes:
        dtype = tensor.dtype
        return tensor.view(torch.uint8).cpu().contiguous().view(dtype)
    return tensor.cpu().contiguous()
