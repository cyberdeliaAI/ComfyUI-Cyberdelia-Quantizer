# Changelog

## 1.1.0 - 2026-11-03

- Add a conservative `MOSS-Image` diffusion-model profile for Base, SFT,
  fine-tunes and Turbo/DMD checkpoints, with split and fused QKV layouts.
- Restrict MOSS conversion to recognized main-block attention/FFN projection
  weights. Keep text refiners, image/text embedders, time embedding, shared
  modulation, output projection, norms, biases and unknown layers in source dtype.
- Support native unprefixed, `diffusion_model.*` and `model.diffusion_model.*`
  keys without changing physical layer names or per-layer quantization markers.
- Add regression tests for all 42 main blocks, sensitive layers, all format
  shape contracts, source rejection and version metadata consistency.
- Update conversion metadata to report converter version 1.1.0.

## 1.0.1 - 2026-10-30

- Fix a Windows access violation crash in `get_tensor()` when converting large checkpoints
  inside ComfyUI. Source and output files are now opened with the safetensors `pread` backend
  on Windows (safetensors 0.8+), with a mmap fallback and a one-time warning on older versions.
- Unload models held by ComfyUI before a real conversion to free RAM.

## 1.0.0 - 2026-09-04

- Initial independent release.
- Exact physical per-layer `.comfy_quant` markers for ComfyUI and Forge Neo.
- FP8 E4M3, FP8 E5M2, NVFP4, MXFP8, INT8, INT8 ConvRot, INT4 ConvRot, and W4A8 INT8.
- Atomic save, post-save header validation, dry-run, strict mode, and overwrite protection.
- Header-only compatibility inspector.
- Model profiles for the architectures supported by the original Kitchen converter.
