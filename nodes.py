"""ComfyUI node definitions."""

from __future__ import annotations

import folder_paths
import comfy.model_management
import comfy.utils

from .checkpoint_io import ConversionError, convert_checkpoint
from .inspection import inspect_checkpoint
from .profiles import PROFILE_NAMES
from .quant_formats import FORMAT_NAMES


class CyberdeliaQuantizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "output_filename": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "empty = source name + format suffix",
                        "tooltip": "A filename only; output is written beside the source model.",
                    },
                ),
                "model_profile": (list(PROFILE_NAMES), {"default": "Krea2"}),
                "quant_format": (list(FORMAT_NAMES), {"default": "INT8_CONVROT"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "strict": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Abort on a quantization error instead of silently keeping that layer dense.",
                    },
                ),
                "overwrite": ("BOOLEAN", {"default": False}),
                "dry_run": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Plan and validate without writing a checkpoint."},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "output_path")
    FUNCTION = "convert"
    CATEGORY = "Cyberdelia/Quantization"
    OUTPUT_NODE = True

    def convert(
        self,
        model_name,
        output_filename,
        model_profile,
        quant_format,
        device,
        strict,
        overwrite,
        dry_run,
    ):
        input_path = folder_paths.get_full_path("diffusion_models", model_name)
        if not input_path:
            raise ConversionError(f"Could not resolve diffusion model: {model_name}")

        progress_bar = None

        def update_progress(current: int, total: int, tensor_name: str):
            nonlocal progress_bar
            comfy.model_management.throw_exception_if_processing_interrupted()
            if progress_bar is None:
                progress_bar = comfy.utils.ProgressBar(total)
            progress_bar.update_absolute(current)
            print(f"[Cyberdelia Quantizer] {current}/{total}: {tensor_name}")

        report = convert_checkpoint(
            input_path=input_path,
            requested_output_name=output_filename,
            profile_name=model_profile,
            format_name=quant_format,
            device_name=device,
            strict=bool(strict),
            overwrite=bool(overwrite),
            dry_run=bool(dry_run),
            progress=update_progress,
        )
        print(f"[Cyberdelia Quantizer] {report.status_text()}")
        for warning in report.warnings:
            print(f"[Cyberdelia Quantizer] Warning: {warning}")
        return report.status_text(), report.output_path


class CyberdeliaQuantInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "inspect"
    CATEGORY = "Cyberdelia/Quantization"
    OUTPUT_NODE = True

    def inspect(self, model_name):
        input_path = folder_paths.get_full_path("diffusion_models", model_name)
        if not input_path:
            raise ConversionError(f"Could not resolve diffusion model: {model_name}")
        report = inspect_checkpoint(input_path)
        print(f"[Cyberdelia Quantizer] {report}")
        return (report,)


NODE_CLASS_MAPPINGS = {
    "CyberdeliaQuantizer": CyberdeliaQuantizer,
    "CyberdeliaQuantInspector": CyberdeliaQuantInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CyberdeliaQuantizer": "🍳 Cyberdelia Quantizer",
    "CyberdeliaQuantInspector": "🔎 Cyberdelia Quant Check",
}
