## Summary

Describe the change and the checkpoint contract it affects.

## Validation

- [ ] `python -m unittest discover -s tests -v` passes
- [ ] Exact physical layer prefixes are preserved
- [ ] Every quantized weight has a valid sibling `.comfy_quant` marker
- [ ] Output side tensors match the current ComfyUI loader contract
- [ ] Documentation and changelog are updated when user-facing behavior changes
