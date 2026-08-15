---
name: downstream-impl
description: Implement downstream features in navigators (new heads, datasets, experiments, export consumers). Use when adding or modifying components in this repo so contracts and verification steps are honored.
---

# Downstream implementation in navigators

## Ground rules
- Port proven components from `~/projects/driving-model-track` (autopilot) instead of writing new ones. Smallest viable diff.
- `uv run` for everything. `train.py` requires a clean git tree (untracked files count) — commit every new file before training.
- Configs live under `navigators/configs/` only (shipped as `navigators.configs` package-data so the installed lib can compose them).

## Contracts (do not break)

### Batch (both dataloaders emit exactly this)
| key | shape | dtype |
|---|---|---|
| `frames` | (B, S, 6, h, w) — prev+cur RGB pair | uint8 |
| `future_poses` | (B, S, plan_len_points, 3) — x, y, v | float32 |
| `frame_speeds` | (B, S, 1) | float32 |
| `frame_times_s` | (B, S) | float32 |

`dataset=dali` (GPU decode) and `dataset=torch` (torchcodec, CPU) are interchangeable; any new loader must match this dict and the `path label start end` file_list format (`navigators/datasets/file_list.py`). Pose targets come from `navigators/datasets/pose_targets.py` — reuse, never reimplement.

### Model (E2EModel = VisionEncoder + ActionDecoder)
- Train forward: `model(x)` with x (B, S, 6, h, w) float in [0,1] → `{"vision": {"pose": ...}, "action": {"plan": {"plans": (B*S, num_modes*(2*num_pts*pose_size+1))}}}`.
- Export forward: `model(x, fb=...)` single frame + feature buffer → tuple `(plan, pose, feat_out, *head_outputs)`; keep `get_export_output_names()` in sync.
- New heads: add to `VisionEncoder.heads` ModuleDict or as a `PlanHead`-style module; wire losses through `E2EModel.get_losses` (vision_weight/action_weight).

### Configs
- Experiments are self-contained `# @package _global_` files in `configs/experiment/` — copy `baseline.yaml`, keep ALL overrides there, never edit recipe files (model/optimizer/dataset yamls).
- New dataset config = new file in `configs/dataset/`, keys interpolate from `common.*` in `train.yaml`.

## Verify before claiming done (in order)
```bash
uv run python -m navigators.smoke_forward model.modules.vision_encoder.pretrained=false  # forward shapes
uv run pytest tests/ -q
uv run ruff check .
uv run python -m navigators.benchmark_dataloader --batches 20 common.data_root=<root>  # if loaders touched
uv run python -m navigators.export checkpoint=<ckpt> output=/tmp/test.onnx            # if model/export touched
```
Report real shapes/losses from these runs. Then commit, push, and give the exact train command:
`uv run navigators/train.py experiment=<name> dataset=<dali|torch>`.
