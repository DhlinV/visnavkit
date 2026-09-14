---
name: new-dataset
description: Wire a new clip dataset into visnavkit dataloaders. Use when new data lands or a new dataset config/file_list is needed.
---

# New dataset

Both loaders (`dataset=dali`, `dataset=torch`) consume the same on-disk contract.
Do not invent a new format — produce this one.

## Clip directory contract
Each clip is a directory containing:
| file | content |
|---|---|
| `<video>.mp4` | h264, any frame rate (targets are interpolated at relative times), already at target downscale |
| `frame_times.npy` | int64 ns timestamps, one per frame |
| `frame_positions.npy` | (N, 3) odom xyz |
| `frame_orientations.npy` | (N, 4) quaternion (w, x, y, z) |
| `frame_speeds.npy` | (N,) m/s |

Optional sidecars: `route_images.npy` (N, h, w, 3) uint8 for `goal_type=route_image`,
`instruction_embedding.npy` (E,) float for `goal_type=instruction`. Point and image goals
need nothing extra (sampled from the clip's future via `common.goal_horizon_s`).
Windows without `plan_len_seconds` of future poses are dropped automatically at index time.

## file_list format
One line per clip: `path label start end` — labels contiguous from 0, `end` exclusive,
relative paths resolved against `data_root` (`visnavkit/data/file_list.py`).

## Steps
1. Write `train.txt` / `val.txt` file_lists next to the clips (or pass absolute paths).
2. Add `visnavkit/configs/dataset/<name>.yaml` — copy `dali.yaml` (or `torch.yaml`) and
   change only file_list names / loader knobs; keep `${common.*}` interpolations.
3. Verify both loaders and compare throughput:
   ```bash
   uv run python -m visnavkit.scripts.benchmark_dataloader --batches 50 common.data_root=<root> dataset=<name>
   ```
   Check the printed frames shape is `(B, S, 6, h, w)` uint8 and first-batch latency is sane.
4. Spot-check targets: poses of a straight-driving clip should have y ~ 0 and v matching
   `frame_speeds`. Pose targets must come from `visnavkit/data/pose_targets.py` — never reimplement.
5. Commit everything (file_lists too if they live in-repo), then train.
