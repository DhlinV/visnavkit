---
name: new-head
description: Add a new action decoder variant to visnavkit. Use when implementing a new trajectory decoder (deterministic regression, MHP, anchors, diffusion, flow matching, ...).
---

# New action decoder

## Fast path: it's a denoiser x scheduler combination
No code. Add `visnavkit/configs/model/action_decoder/<name>.yaml`:
```yaml
defaults: [generative, {override denoiser: dit}, {override scheduler: flow}, _self_]
sample_steps: 4
anchors: {_target_: visnavkit.models.action.anchors.AnchorSet, num_anchors: 16}  # optional
```
New denoisers go in `models/action/denoisers/` with `forward(x_t (N,T,A), t (N,) in [0,1], cond (N,D), tokens (N,L,D)) -> (N,T,A)`
and a `_partial_: true` yaml in `configs/model/action_decoder/denoiser/`; new schedulers subclass
`BaseScheduler` (`sample_t`, `add_noise`, `target`, `step`).

## New decoder class
Subclass `BaseActionDecoder` in `visnavkit/models/action/<file>.py`:
- `decode(cond (N,D), tokens (N,L,D), noise) -> PlanOutput(plans=self.pack(mu, log_b, logits), mu=..., ...)`
  where `mu` is `(N, M, T, A)` in **normalized action space**; `pack` unnormalizes, maps
  the action space to poses, and writes the flat `[mu, log_scale, conf]` layout.
- `loss(preds, gt (N,T,A) normalized, targets) -> (dict(total, reg, cls), debug)`; use
  `torch.zeros_like(reg)` when there is no classification term.
- Set `num_modes` in `super().__init__`; set `uses_noise = True` and implement `example_noise`
  if inference consumes explicit noise (export adds a `noise` input automatically).
- Generative decoders return zero `plans` in training and keep `cond`/`tokens` for the loss.

Add `configs/model/action_decoder/<name>.yaml` with `defaults: [base, _self_]` and `_target_`.

## Verify
```bash
uv run visnavkit-sanity-check --onnx model/action_decoder=<name>
```
Add `<name>` to `DECODERS` in `tests/models/test_model_configs.py` and to `decoders()` in
`tests/models/test_action_decoders.py`, then pytest + ruff.
