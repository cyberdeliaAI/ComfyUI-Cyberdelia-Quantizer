# ComfyUI Cyberdelia Quantizer

[![Tests](https://github.com/cyberdeliaAI/ComfyUI-Cyberdelia-Quantizer/actions/workflows/tests.yml/badge.svg)](https://github.com/cyberdeliaAI/ComfyUI-Cyberdelia-Quantizer/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

An independent ComfyUI custom node for converting original FP16/BF16 diffusion
checkpoints to native mixed-precision safetensors files.

The main compatibility fix is deliberately simple: every quantized weight gets
an explicit marker beside the **exact physical layer name**:

```text
model.diffusion_model.blocks.0.attn.to_q.weight
model.diffusion_model.blocks.0.attn.to_q.weight_scale
model.diffusion_model.blocks.0.attn.to_q.comfy_quant
```

The prefix is never stripped. This prevents the central-metadata prefix mismatch
that can make Forge Neo treat an INT8 checkpoint as FP16 and report unexpected
`SingleStreamDiT` keys.

## Formats

| Option | Storage | Runtime notes |
| --- | --- | --- |
| `FP8_E4M3` | FP8 E4M3 | Portable FP8 storage; full-precision matmul for compatibility |
| `FP8_E5M2` | FP8 E5M2 | Wider range, lower mantissa precision; full-precision matmul |
| `NVFP4` | packed 4-bit | Native acceleration on NVIDIA Blackwell (SM 10.0+) |
| `MXFP8` | block-scaled FP8 | Native acceleration on NVIDIA Blackwell (SM 10.0+) |
| `INT8` | per-channel INT8 | Native ComfyUI format; accelerated on NVIDIA Turing or newer |
| `INT8_CONVROT` | rotated per-channel INT8 | Recommended broad option; NVIDIA Turing or newer |
| `INT4_CONVROT` | packed rotated INT4 | Smaller; NVIDIA Turing or newer; profile-sensitive |
| `W4A8_INT8` | grouped 4-bit weights / INT8 runtime | Requires a recent kitchen backend; Ampere-class path or newer |

NVFP4 and MXFP8 files still load on unsupported hardware, but ComfyUI may fall
back to dequantized computation instead of providing a speed-up. FP8 in this node
uses the official compatibility-oriented marker `full_precision_matrix_mult=true`.

## Requirements

- A current ComfyUI installation (version 0.34 or newer is recommended).
- The `comfy-kitchen` version pinned by that ComfyUI release.
- An original dense FP16/BF16 `.safetensors` diffusion model.

Do not manually upgrade only `comfy-kitchen` while leaving ComfyUI behind. Update
ComfyUI and its requirements together so their quantization contracts match.

Do not use an FP8, INT8, GGUF, NVFP4, or other already-quantized checkpoint as the
source. The node rejects recognized quantized inputs instead of silently applying
a second lossy conversion.

## Installation

From the ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/cyberdeliaAI/ComfyUI-Cyberdelia-Quantizer.git
```

Alternatively, download a release archive and extract it as:

```text
ComfyUI/custom_nodes/ComfyUI-Cyberdelia-Quantizer
```

Then update ComfyUI's normal requirements and restart ComfyUI. No separate node
dependencies are installed because Torch, safetensors, and comfy-kitchen must
match the host ComfyUI build.

## Nodes

### 🍳 Cyberdelia Quantizer

1. Select the original model in `model_name`.
2. Select the matching `model_profile`.
3. Select the output `quant_format`.
4. Leave `output_filename` empty to use the source filename plus a format suffix.
5. Run once with `dry_run=true` if you want a plan without writing a file.
6. Keep `strict=true` for release builds.

The output is written beside the source model. Existing files are protected unless
`overwrite=true`. Saving uses a temporary file, validates its marker/weight/sidecar
contract, and only then moves it to the final filename.

### 🔎 Cyberdelia Quant Check

This checks only the safetensors header and tiny JSON markers. It identifies:

- exact per-layer `.comfy_quant` packaging;
- legacy central `_quantization_metadata`;
- the `model.diffusion_model.*` prefix mismatch;
- orphan markers and missing side tensors.

It does not load the model weights into RAM.

## Profiles

Dedicated profiles are included for Krea2, Anima, Z-Image Base/Turbo, Flux 1/2,
Qwen Image, Wan 2.2, LTX-2, ACE-Step, Boogu Image, Chroma, ERNIE Image,
Ideogram 4, MiniMax H3, and SeedVR. A conservative generic profile is also
available.

A profile controls which sensitive layers stay in their original source dtype.
Shape-incompatible layers are also kept unchanged and listed in the result.
The Qwen Image 2512 profile keeps its sensitive `txt_mlp`/`txt_mod` weights in
FP8 E4M3 when a more aggressive format is selected.

## Design differences from the Tritant converter

- Writes exact per-layer `.comfy_quant` tensors; no prefix-stripped central map.
- Adds real FP8 E4M3 and FP8 E5M2 output modes (MXFP8 remains separate).
- Reads source tensors one at a time instead of loading the complete dense source
  checkpoint before conversion.
- Preserves excluded and non-floating tensors exactly instead of casting every
  unquantized tensor to BF16.
- Aborts on layer failures by default instead of returning a misleading success.
- Refuses accidental source overwrite and uses an atomic validated save.
- Chooses the largest valid ConvRot group per layer (256, 128, 64, 32, or 16).

## Tests

The dependency-free contract tests can be run outside ComfyUI:

```bash
python -m unittest discover -s tests -v
```

They specifically guard against reintroducing the prefix-stripping bug.

## License and upstream references

Apache-2.0. See `NOTICE` for attribution. This project is not an official Comfy
Org project. The checkpoint contract follows the public ComfyUI native
quantization loader and comfy-kitchen layouts.

Repository: <https://github.com/cyberdeliaAI/ComfyUI-Cyberdelia-Quantizer>
