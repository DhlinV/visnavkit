---
name: new-encoder
description: Add a new vision encoder variant to navigators. Use when swapping or implementing a new image backbone (CNN, ViT, DINO, ...).
---

# New vision encoder

One file per encoder in `navigators/models/encoders/`, selected via `_target_` in a
`configs/model/<name>.yaml` recipe. Never edit an existing encoder class to add a variant.

## Fast path: it's just a different timm backbone
`VisionEncoder` already takes `backbone_name`, `out_indices` (last 3 stages — `[2, 3, 4]`
for 5-stage CNNs like mobilenet/efficientnet), and `act_layer` (`null` keeps the backbone's
native activations, required for pretrained CNN weights). Yaml-only, no new code.

## New encoder class
Match the VisionEncoder forward contract exactly:
- input: (B*S, 6, h, w) float in [0,1] — stacked prev+cur RGB pair
- output: `dict(pose=(B*S, 1), feat_out=(B*S, feat_size), prev_img_mask=(B*S,))`
- `get_losses(preds, targets)` -> loss dict with `total` (copy VisionEncoder's)
- reuse `_neck` and `PoseHead`; per-frame normalization from the timm data config
- frozen backbones must stay in eval mode under `.train()` — override `train()`
  (see `DinoEncoder.freeze_backbone`)

## Recipe yaml
```yaml
defaults: [base, _self_]
modules:
  vision_encoder:
    _target_: navigators.models.encoders.<file>.<Class>
    ...
```

## Verify
```bash
uv run python -m navigators.scripts.smoke_forward model=<name> model.modules.vision_encoder.pretrained=false
```
Add `<name>` to the zoo parametrization in `tests/models/test_smoke_forward.py`, then
pytest + ruff. Commit before training (clean-tree requirement).
