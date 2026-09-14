---
name: downstream-impl
description: Implement downstream features in visnavkit (new components, datasets, experiments, export consumers). Use when adding or modifying components in this repo so contracts and verification steps are honored.
---

# Downstream implementation in visnavkit

## Ground rules
- Smallest viable diff; reuse the stage bases (`BaseVisionEncoder`, `BaseTemporalEncoder`, `BaseGoalEncoder`, `BaseActionDecoder`) over new abstractions.
- `uv run` for everything. Training records Git provenance; `strict_git=true` requires a clean tree (untracked files count).
- Configs live under `visnavkit/configs/` only (shipped as `visnavkit.configs` package-data).

## Contracts (do not break)

### Batch (both dataloaders emit exactly this)
| key | shape | dtype |
|---|---|---|
| `frames` | (B, S, 6, h, w) — prev+cur RGB pair | uint8 |
| `future_poses` | (B, S, plan_len_points, 3) — x, y, v | float32 |
| `frame_speeds` | (B, S, 1) | float32 |
| `frame_times_s` | (B, S) | float64 |
| `target_times_s` | (B, T) relative seconds | float32 |
| `goal` (optional) | point (B, S, 3) · image (B, 3, h, w) uint8 · route_image (B, C, h, w) · instruction (B, E) | — |

`dataset=torch` (torchcodec, CPU) supports every goal type; `dataset=dali` (GPU) supports `none`/`point`. The dataset reads `goal_type` from `${model.goal_encoder.goal_type}`. Pose and goal targets come from `visnavkit/data/pose_targets.py` — reuse, never reimplement.

### Layout
`models/vision` · `models/temporal` · `models/goal` · `models/action` (+ `denoisers/`, `schedulers/`) · `models/policy.py` (NavigationPolicy) · `models/lit_model.py` · `data/` · `evaluation/` · `benchmark/` · `scripts/` (thin entry points) · `configs/model/<group>/` mirrors the packages.

### Policy
- Training: `policy(frames, goal=None, noise=None)` with frames (B, S, 6, h, w) float in [0,1] → `PolicyOutput(vision=VisionOutput(tokens (B*S, K, D), pose), plan=PlanOutput(plans (decisions, M*(2*T*P+1))), goal_tokens)`.
- Deployment: `policy.predict(frame, feature_buffer, goal=None, noise=None)` → `(plan, pose, feat_out, *heads)`; `export_input_names()` / `export_output_names()` define the ONNX contract and are presence-driven (goal, noise).
- Losses: `policy.get_losses(out, targets)` combines `vision_encoder.get_losses` and `action_decoder.get_losses` with `loss_cfg` weights.

### Configs
- Experiments: self-contained `# @package _global_` files in `configs/experiment/`; never edit recipe files for an experiment.
- Component groups are nested: `model/vision_encoder=...`, `model/temporal_encoder=...`, `model/goal_encoder=...`, `model/action_decoder=...` (+ `model/action_decoder/denoiser=...`, `.../scheduler=...`). Interpolate widths from `${model.feat_size}`.

## Verify before claiming done (in order)
```bash
uv run visnavkit-sanity-check --onnx <overrides>          # shapes, loss/backward, buffer parity, ONNX parity
uv run pytest tests/ -q
uv run ruff check visnavkit tests
uv run python -m visnavkit.scripts.benchmark_dataloader --batches 20 common.data_root=<root>  # if loaders touched
uv run visnavkit-export checkpoint=<ckpt> output=/tmp/test.onnx                             # if model/export touched
```
Report real shapes/losses from these runs. Then commit, push, and give the exact train command:
`uv run visnavkit-train experiment=<name> dataset=<dali|torch>`.
