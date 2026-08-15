---
name: new-head
description: Add a new plan head variant to navigators. Use when implementing a new trajectory decoder (deterministic regression, MHP, diffusion, ...).
---

# New plan head

One file per head in `navigators/models/heads/`, swapped via
`modules.action_decoder.plan_head._target_` in a `configs/model/<name>.yaml` recipe.

## Contract (all four required)
- `forward(x)` with x (B*S, feat_size) -> `dict(plans=(B*S, flat_size), ...)`. Extra keys
  (e.g. conditioning features) are allowed — the same dict comes back to you in get_losses.
- `plans` MUST use the flat MHP layout: per mode `[mu(num_pts*pose_size),
  log_b(num_pts*pose_size), conf]`, `flat_size = num_modes * (2*num_pts*pose_size + 1)`.
  Heads without scales/confidences emit log_b=0, conf=0 (uniform after softmax) — this
  keeps metrics, export, and `parse_plan_output` unchanged.
- `get_losses(preds, gt)` where `preds` is YOUR forward dict and gt is future_poses
  (B*S, num_pts, pose_size) -> `(dict(total, reg, cls), debug_dict)`. No cls concept?
  Use `torch.zeros_like(reg).detach()`.
- `parse_output(output)` -> delegate to `parse_plan_output` from `plan_head.py`.

Also required: a `self.pretrained` attribute (ActionDecoder checks it before re-initializing
weights) and `**_ignored` kwargs (recipes inherit the base PlanHead keys via hydra merge).

## Copy from
`plan_head.py` (MHP + Laplace NLL), `waypoint_head.py` (deterministic single-mode),
`diffusion_plan_head.py` (DDPM/DDIM; uses forward-dict conditioning in the loss).

## Verify
```bash
uv run python -m navigators.scripts.smoke_forward model=<name> model.modules.vision_encoder.pretrained=false
```
Add the recipe to the zoo test parametrization + a train-mode loss test (finite loss),
pytest + ruff. If the head keeps the flat layout, export needs no changes.
