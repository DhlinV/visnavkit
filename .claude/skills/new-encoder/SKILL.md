---
name: new-encoder
description: Add a new vision encoder variant to visnavkit. Use when swapping or implementing a new image backbone (CNN, ViT, DINO, ...).
---

# New vision encoder

## Fast path: it's a timm backbone (almost always)
No code. Add `visnavkit/configs/model/vision_encoder/<name>.yaml`:
```yaml
defaults: [cnn, _self_]          # cnn: features_only pyramid (ResNet, EfficientNet, ConvNeXt, FastViT, ...)
backbone_name: convnext_tiny     # vit: forward_features ViTs (DINOv2/v3, DeiT, EVA-02, SigLIP, CLIP, ...)
out_indices: [1, 2, 3]           # cnn only: last three stages; act_layer: null keeps native activations
```
Token mode (`global|patch|fused`, `patch_grid`) and pair fusion (`early|late`) are base options and need no code.

## New encoder class (non-timm backbone)
Subclass `BaseVisionEncoder` in `visnavkit/models/vision/<file>.py` and implement
`_encode(x) -> (global (N, E), patch_map (N, E, h, w) | None, pyramid list)` for normalized
frames with `backbone_in_chans` channels. The base handles pair preparation, token
projection, the speed head, `freeze_backbone`, and `get_losses`. Override
`prepare_for_export(image_size)` only if tracing needs a fixed-resolution copy.

## Verify
```bash
uv run visnavkit-sanity-check --onnx model/vision_encoder=<name> model.vision_encoder.token_mode=fused
```
Add `<name>` to `VISION_PRESETS` in `tests/models/test_model_configs.py` (or the large-preset list if it is too big for CI), then pytest + ruff. Commit before training.
