from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from test_contract import FakeSafeOpen, FakeSlice, checkpoint_io, fake_safetensors_module, package, profiles


class MossHeader(FakeSafeOpen):
    def __init__(self, shapes, dtypes=None):
        super().__init__(list(shapes), dtypes=dtypes)
        self.shapes = shapes

    def get_slice(self, key):
        return FakeSlice(self.shapes[key], self._dtypes.get(key, "BF16"))

    def get_tensor(self, key):
        raise AssertionError("Planning a MOSS checkpoint must not load its weights")


def moss_shapes(prefix="", fused=True):
    shapes = {}
    for i in range(42):
        block = f"{prefix}layers.{i}.proj."
        if fused:
            shapes[block + "qkv.weight"] = (9216, 3072)
        else:
            for name in ("q", "k", "v"):
                shapes[block + name + ".weight"] = (3072, 3072)
        shapes[block + "o.weight"] = (3072, 3072)
        shapes[block + "ffn_up_gate.weight"] = (16384, 3072)
        shapes[block + "ffn_down.weight"] = (3072, 8192)
    shapes.update(
        {
            prefix + "video_embedder.weight": (3072, 128),
            prefix + "text_embedder.proj.weight": (3072, 4096),
            prefix + "time_embedding.0.weight": (3072, 256),
            prefix + "time_embedding.2.weight": (3072, 3072),
            prefix + "shared_block_modulation.weight": (18432, 3072),
            prefix + "video_head_modulation.weight": (6144, 3072),
            prefix + "final_linear_video.weight": (128, 3072),
            prefix + "final_norm_video.weight": (3072,),
            prefix + "layers.0.q_norm.weight": (128,),
            prefix + "video_embedder.bias": (3072,),
        }
    )
    for i in (0, 1):
        for name, shape in {
            "qkv": (9216, 3072),
            "proj": (3072, 3072),
            "ffn_gate": (8192, 3072),
            "ffn_up": (8192, 3072),
            "ffn_down": (3072, 8192),
        }.items():
            shapes[f"{prefix}text_embedder.blocks.{i}.{name}.weight"] = shape
    return shapes


class MossProfileTests(unittest.TestCase):
    def plan(self, shapes, quant_format="INT8_CONVROT", dtypes=None):
        header = MossHeader(shapes, dtypes)
        with patch.dict(sys.modules, {"safetensors": fake_safetensors_module(header)}):
            return checkpoint_io.build_plan(Path("moss.safetensors"), "MOSS-Image", quant_format)[0]

    def test_main_block_coverage_for_fused_split_and_physical_prefixes(self):
        for prefix in ("", "diffusion_model.", "model.diffusion_model."):
            for fused in (False, True):
                with self.subTest(prefix=prefix, fused=fused):
                    shapes = moss_shapes(prefix, fused)
                    plan = self.plan(shapes)
                    self.assertEqual(len(plan.selected), 42 * (4 if fused else 6))
                    self.assertFalse(plan.incompatible)
                    self.assertEqual({p.name.split("layers.")[0] for p in plan.selected}, {prefix})
                    protected = {key for key in shapes if ".proj." not in key or "text_embedder" in key}
                    self.assertTrue(protected <= set(plan.excluded + plan.copied))

    def test_sensitive_moss_layers_and_unknown_weights_are_preserved(self):
        profile = profiles.get_profile("MOSS-Image")
        for prefix in ("", "diffusion_model.", "model.diffusion_model."):
            for name in (
                "shared_block_modulation.weight",
                "final_linear_video.weight",
                "time_embedding.0.weight",
                "time_embedding.2.weight",
                "video_head_modulation.weight",
                "video_embedder.weight",
                "text_embedder.blocks.0.qkv.weight",
                "text_embedder.blocks.1.ffn_down.weight",
                "text_encoders.qwen35_9b.transformer.layers.0.proj.qkv.weight",
                "vae.layers.0.proj.qkv.weight",
                "control_projection.weight",
            ):
                with self.subTest(weight=prefix + name):
                    self.assertTrue(profile.excludes(prefix + name))

    def test_all_formats_accept_native_moss_main_projection_shapes(self):
        shapes = moss_shapes()
        for fmt in package.FORMAT_NAMES:
            with self.subTest(format=fmt):
                plan = self.plan(shapes, fmt)
                self.assertEqual(len(plan.selected), 168)
                self.assertFalse(plan.incompatible)
                self.assertEqual({p.format_name for p in plan.selected}, {fmt})
                for item in plan.selected:
                    self.assertEqual(
                        package.marker_key_for_weight(item.name),
                        item.name.removesuffix(".weight") + ".comfy_quant",
                    )

    def test_prefixed_moss_markers_keep_exact_layer_names(self):
        plan = self.plan(moss_shapes("model.diffusion_model."))
        for item in plan.selected:
            marker = package.marker_config(item.format_name, item.shape)
            self.assertTrue(marker["convrot"])
            self.assertEqual(marker["convrot_groupsize"], 256)
            self.assertTrue(
                package.marker_key_for_weight(item.name).startswith("model.diffusion_model.layers.")
            )

    def test_non_moss_model_is_not_accidentally_quantized_by_moss_profile(self):
        with self.assertRaisesRegex(checkpoint_io.ConversionError, "No eligible Linear"):
            self.plan(
                {
                    "blocks.0.attn.to_q.weight": (3072, 3072),
                    "text_encoder.layers.0.proj.qkv.weight": (9216, 3072),
                }
            )

    def test_already_quantized_moss_source_is_rejected(self):
        shapes = moss_shapes()
        first = next(iter(shapes))
        for dtype in ("I8", "F8_E4M3"):
            with (
                self.subTest(dtype=dtype),
                self.assertRaisesRegex(checkpoint_io.ConversionError, "already stored"),
            ):
                self.plan(shapes, dtypes={first: dtype})

    def test_converter_version_matches_package_metadata(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
        version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
        report = checkpoint_io.ConversionReport(
            "moss_int8.safetensors", "INT8_CONVROT", "MOSS-Image", 168, 20, 0, 0
        )
        self.assertEqual(checkpoint_io._clean_metadata({}, report)["converter_version"], version)


if __name__ == "__main__":
    unittest.main()
