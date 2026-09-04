"""Model-family profiles for mixed-precision checkpoint conversion.

Profiles only decide which two-dimensional ``*.weight`` tensors are eligible.
Format-specific shape checks are handled separately by :mod:`quant_formats`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    excluded_fragments: tuple[str, ...]
    description: str
    forced_fp8_fragments: tuple[str, ...] = ()

    def excludes(self, tensor_name: str) -> bool:
        lowered = tensor_name.lower()
        return any(fragment.lower() in lowered for fragment in self.excluded_fragments)

    def format_for(self, tensor_name: str, requested_format: str) -> str:
        lowered = tensor_name.lower()
        if requested_format not in {"FP8_E4M3", "FP8_E5M2"} and any(
            fragment.lower() in lowered for fragment in self.forced_fp8_fragments
        ):
            return "FP8_E4M3"
        return requested_format


_COMMON_SENSITIVE = (
    "bias",
    "norm",
    "scale",
    "final_layer",
    "proj_out",
    "head",
    "embedder",
    "embedding",
)


PROFILES: dict[str, ModelProfile] = {
    "Generic (safe)": ModelProfile(
        "Generic (safe)",
        _COMMON_SENSITIVE
        + (
            "patch_embed",
            "input_proj",
            "output_proj",
            "time_in",
            "vector_in",
            "guidance_in",
        ),
        "Conservative fallback for an architecture without a dedicated profile.",
    ),
    "Krea2": ModelProfile(
        "Krea2",
        ("first", "last", "tmlp", "tproj", "txtfusion", "txtmlp"),
        "Krea 2 / SingleStreamDiT profile.",
    ),
    "Z-Image-Turbo": ModelProfile(
        "Z-Image-Turbo",
        (
            "cap_embedder",
            "x_embedder",
            "noise_refiner",
            "context_refiner",
            "t_embedder",
            "final_layer",
        ),
        "Z-Image Turbo profile.",
    ),
    "Z-Image-Base": ModelProfile(
        "Z-Image-Base",
        (
            "attention",
            "adaln_modulation",
            "norm",
            "final_layer",
            "cap_embedder",
            "x_embedder",
            "noise_refiner",
            "context_refiner",
            "t_embedder",
        ),
        "Conservative Z-Image Base profile.",
    ),
    "Anima": ModelProfile(
        "Anima",
        _COMMON_SENSITIVE
        + ("llm_adapter", "x_embedder", "t_embedder", "context_embedder"),
        "Anima profile.",
    ),
    "Flux.1-dev": ModelProfile(
        "Flux.1-dev",
        (
            "bias",
            "txt_attn",
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "final_layer",
            "class_embedding",
            "single_stream_modulation",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
        ),
        "Flux.1 dev profile.",
    ),
    "Flux.1-Fill": ModelProfile(
        "Flux.1-Fill",
        (
            "bias",
            "txt_attn",
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "final_layer",
            "class_embedding",
            "single_stream_modulation",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
        ),
        "Flux.1 Fill profile.",
    ),
    "Flux.2-dev": ModelProfile(
        "Flux.2-dev",
        (
            "bias",
            "txt_attn",
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "final_layer",
            "class_embedding",
            "single_stream_modulation",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
        ),
        "Flux.2 dev profile.",
    ),
    "Flux.2-Klein-9b": ModelProfile(
        "Flux.2-Klein-9b",
        (
            "bias",
            "txt_attn",
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "final_layer",
            "class_embedding",
            "single_stream_modulation",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
        ),
        "Flux.2 Klein 9B profile.",
    ),
    "Qwen-Image-Edit-2511": ModelProfile(
        "Qwen-Image-Edit-2511",
        ("img_in", "txt_in", "time_text_embed", "norm_out", "proj_out"),
        "Qwen Image Edit 2511 profile.",
    ),
    "Qwen-Image-2512": ModelProfile(
        "Qwen-Image-2512",
        ("img_in", "txt_in", "time_text_embed", "norm_out", "proj_out", "img_mod.1"),
        "Qwen Image 2512 profile.",
        ("txt_mlp", "txt_mod"),
    ),
    "Wan2.2-i2v-high-low": ModelProfile(
        "Wan2.2-i2v-high-low",
        ("text_embedding", "time_embedding", "time_projection", "head"),
        "Wan 2.2 image-to-video high/low profile.",
    ),
    "LTX-2-19b-dev-or-distilled": ModelProfile(
        "LTX-2-19b-dev-or-distilled",
        (
            "vae.",
            "vocoder.",
            "connector",
            "proj_out",
            "norm",
            "bias",
            "scale",
            "embedder",
            "patchify",
            "table",
            "transformer_blocks.0.",
            "transformer_blocks.43.",
            "transformer_blocks.44.",
            "transformer_blocks.45.",
            "transformer_blocks.46.",
            "transformer_blocks.47.",
            "projection",
            "adaln_single",
        ),
        "LTX-2 19B dev/distilled profile.",
    ),
    "ACE-Step": ModelProfile(
        "ACE-Step",
        _COMMON_SENSITIVE
        + (
            "time_projection",
            "adaln",
            "t_embedder",
            "x_embedder",
            "y_embedder",
            "project_in",
            "quantizer",
            "embed_tokens",
            "timbre_encoder",
        ),
        "ACE-Step profile.",
    ),
    "Boogu-Image": ModelProfile(
        "Boogu-Image",
        (
            "image_index_embedding",
            "ref_image_patch_embedder",
            "norm1.linear",
            "norm_out.linear_1",
            "norm_out.linear_2",
        ),
        "Boogu Image profile.",
    ),
    "Chroma": ModelProfile(
        "Chroma",
        (
            "bias",
            "txt_attn",
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "final_layer",
            "class_embedding",
            "single_stream_modulation",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
        ),
        "Chroma profile.",
    ),
    "ERNIE-Image": ModelProfile(
        "ERNIE-Image",
        _COMMON_SENSITIVE
        + (
            "time_projection",
            "adaln",
            "t_embedder",
            "x_embedder",
            "y_embedder",
            "context_embedder",
            "single_stream_modulation",
            "enhancer",
        ),
        "ERNIE Image profile.",
    ),
    "Ideogram-4": ModelProfile(
        "Ideogram-4",
        _COMMON_SENSITIVE
        + (
            "x_embedder",
            "t_embedder",
            "context_embedder",
            "time_embedding",
            "modulation",
            "adaln",
            "q_norm",
            "k_norm",
            "img_in",
            "txt_in",
            "time_in",
            "vector_in",
            "guidance_in",
            "single_stream_modulation",
            "embed_image_indicator",
            "embed_text_indicator",
            "adaln_proj",
            "input_proj",
            "llm_cond_proj",
            "t_embedding",
        ),
        "Ideogram 4 profile.",
    ),
    "MiniMax-H3": ModelProfile(
        "MiniMax-H3",
        (
            "bias",
            "norm",
            "adaln_proj",
            "adaln_t_table",
            "condition_proj",
            "final_layer",
            "token_refiner",
            "patch_proj",
            "rope",
        ),
        "MiniMax H3 profile.",
    ),
    "SeedVR": ModelProfile(
        "SeedVR",
        _COMMON_SENSITIVE
        + (
            "pos_emb",
            "neg_emb",
            "patch_embed",
            "pos_embed",
            "x_embedder",
            "t_embedder",
            "context_embedder",
            "y_embedder",
            "time_in",
            "vector_in",
            "guidance_in",
            "txt_in",
            "img_in",
            "single_stream_modulation",
            "double_stream_modulation",
            "vae",
            "attn.proj_qkv_vid",
        ),
        "SeedVR profile.",
    ),
}


PROFILE_NAMES = tuple(PROFILES)


def get_profile(name: str) -> ModelProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model profile: {name}") from exc
