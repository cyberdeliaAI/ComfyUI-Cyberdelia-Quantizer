# Contributing

Bug reports and narrowly scoped pull requests are welcome.

## Before opening an issue

Please run the `🔎 Cyberdelia Quant Check` node on the affected checkpoint and
include its result. Also include:

- ComfyUI version or commit;
- comfy-kitchen version;
- Torch and CUDA versions;
- GPU model;
- selected model profile and quantization format;
- whether the source was original FP16/BF16;
- the complete Python traceback as text.

Do not upload model checkpoints or private workflow data unless you have the
right to share them.

## Pull requests

Run the dependency-free contract suite before submitting:

```bash
python -m unittest discover -s tests -v
```

Keep the exact physical layer prefix when adding or changing checkpoint metadata.
Every quantized `*.weight` must have a sibling `*.comfy_quant` marker and all
side tensors required by the selected native ComfyUI format.
