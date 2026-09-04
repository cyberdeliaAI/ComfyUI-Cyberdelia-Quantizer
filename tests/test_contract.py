from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PARENT))

package = __import__("ComfyUI-Cyberdelia-Quantizer.quant_formats", fromlist=["dummy"])
checkpoint_io = __import__("ComfyUI-Cyberdelia-Quantizer.checkpoint_io", fromlist=["dummy"])
profiles = __import__("ComfyUI-Cyberdelia-Quantizer.profiles", fromlist=["dummy"])
inspection = __import__("ComfyUI-Cyberdelia-Quantizer.inspection", fromlist=["dummy"])


class FakeSlice:
    def __init__(self, shape=(2, 256), dtype="I8"):
        self.shape = shape
        self.dtype = dtype

    def get_shape(self):
        return self.shape

    def get_dtype(self):
        return self.dtype


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeSafeOpen:
    def __init__(self, keys, metadata=None, dtypes=None, tensors=None):
        self._keys = keys
        self._metadata = metadata or {}
        self._dtypes = dtypes or {}
        self._tensors = tensors or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def keys(self):
        return self._keys

    def metadata(self):
        return self._metadata

    def get_slice(self, key):
        return FakeSlice(dtype=self._dtypes.get(key, "F16"))

    def get_tensor(self, key):
        return self._tensors[key]


def fake_safetensors_module(handle):
    module = types.ModuleType("safetensors")
    module.safe_open = lambda *args, **kwargs: handle
    return module


class QuantizationContractTests(unittest.TestCase):
    def test_registry_metadata_matches_cyberdelia_publisher(self):
        metadata = (PACKAGE_PARENT / "ComfyUI-Cyberdelia-Quantizer" / "pyproject.toml").read_text()
        self.assertIn('name = "cyberdelia-quantizer"', metadata)
        self.assertIn('PublisherId = "cyberdelia"', metadata)
        self.assertIn('/assets/icon.png"', metadata)

    def test_layer_name_keeps_model_prefix(self):
        weight = "model.diffusion_model.blocks.0.attn.to_q.weight"
        self.assertEqual(
            package.layer_name_from_weight(weight),
            "model.diffusion_model.blocks.0.attn.to_q",
        )
        self.assertEqual(
            package.marker_key_for_weight(weight),
            "model.diffusion_model.blocks.0.attn.to_q.comfy_quant",
        )
        self.assertEqual(
            package.sidecar_key_for_weight(weight, "input_scale"),
            "model.diffusion_model.blocks.0.attn.to_q.input_scale",
        )

    def test_int8_convrot_marker_is_native(self):
        marker = package.marker_config("INT8_CONVROT", (3072, 3072))
        self.assertEqual(marker["format"], "int8_tensorwise")
        self.assertTrue(marker["convrot"])
        self.assertEqual(marker["convrot_groupsize"], 256)

    def test_fp8_marker_uses_compatible_matmul(self):
        marker = package.marker_config("FP8_E4M3", (128, 256))
        self.assertEqual(marker["format"], "float8_e4m3fn")
        self.assertTrue(marker["full_precision_matrix_mult"])

    def test_fp8_e5m2_has_its_own_native_contract(self):
        spec = package.get_format_spec("FP8_E5M2")
        marker = package.marker_config("FP8_E5M2", (128, 256))
        self.assertEqual(marker["format"], "float8_e5m2")
        self.assertEqual(spec.required_sidecars, ("weight_scale", "input_scale"))
        self.assertTrue(marker["full_precision_matrix_mult"])

    def test_falls_back_to_largest_valid_convrot_group(self):
        self.assertEqual(package.choose_convrot_groupsize(2688), 128)
        self.assertEqual(package.choose_convrot_groupsize(96), 32)
        self.assertIsNone(package.choose_convrot_groupsize(15))

    def test_int4_requires_64_column_alignment(self):
        self.assertEqual(package.shape_compatibility("INT4_CONVROT", (64, 192)), (True, None))
        compatible, reason = package.shape_compatibility("INT4_CONVROT", (64, 160))
        self.assertFalse(compatible)
        self.assertIn("64", reason)

    def test_output_filename_is_sanitized_and_suffixed(self):
        result = checkpoint_io.normalize_output_filename(
            "source.safetensors", "../My Model.safetensors", "NVFP4"
        )
        self.assertEqual(result, "My Model_nvfp4.safetensors")

    def test_output_suffix_is_not_duplicated(self):
        result = checkpoint_io.normalize_output_filename(
            "source.safetensors", "release_fp8_e4m3", "FP8_E4M3"
        )
        self.assertEqual(result, "release_fp8_e4m3.safetensors")

    def test_krea_profile_keeps_normal_block_and_excludes_sensitive_layer(self):
        profile = profiles.get_profile("Krea2")
        self.assertFalse(profile.excludes("model.diffusion_model.blocks.2.attn.to_q.weight"))
        self.assertTrue(profile.excludes("model.diffusion_model.first_layer.weight"))

    def test_qwen_profile_keeps_sensitive_text_layers_in_fp8(self):
        profile = profiles.get_profile("Qwen-Image-2512")
        self.assertEqual(
            profile.format_for("model.diffusion_model.blocks.2.txt_mlp.weight", "NVFP4"),
            "FP8_E4M3",
        )
        self.assertEqual(
            profile.format_for("model.diffusion_model.blocks.2.img_mlp.weight", "NVFP4"),
            "NVFP4",
        )

    def test_inspector_detects_legacy_prefix_mismatch(self):
        layer = "blocks.0.attn.to_q"
        handle = FakeSafeOpen(
            keys=[
                f"model.diffusion_model.{layer}.weight",
                f"model.diffusion_model.{layer}.weight_scale",
            ],
            metadata={
                "_quantization_metadata": (
                    '{"layers":{"blocks.0.attn.to_q":{"format":"int8_tensorwise"}}}'
                )
            },
        )
        with patch.dict(sys.modules, {"safetensors": fake_safetensors_module(handle)}):
            result = inspection.inspect_checkpoint("legacy.safetensors")
        self.assertIn("RISK", result)
        self.assertIn("prefix mismatch", result)

    def test_inspector_accepts_exact_native_int8_marker(self):
        layer = "model.diffusion_model.blocks.0.attn.to_q"
        marker = list(b'{"convrot":true,"convrot_groupsize":256,"format":"int8_tensorwise"}')
        keys = [f"{layer}.weight", f"{layer}.weight_scale", f"{layer}.comfy_quant"]
        handle = FakeSafeOpen(
            keys=keys,
            dtypes={f"{layer}.weight": "I8"},
            tensors={f"{layer}.comfy_quant": FakeTensor(marker)},
        )
        with patch.dict(sys.modules, {"safetensors": fake_safetensors_module(handle)}):
            result = inspection.inspect_checkpoint("native.safetensors")
        self.assertIn("OK", result)
        self.assertIn("exact per-layer", result)


if __name__ == "__main__":
    unittest.main()
