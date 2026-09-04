"""ComfyUI entry point for the Cyberdelia Quantizer nodes."""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ModuleNotFoundError as exc:
    # Allow dependency-free contract tests outside a ComfyUI installation.
    if exc.name not in {"folder_paths", "comfy", "comfy.model_management", "comfy.utils"}:
        raise
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
