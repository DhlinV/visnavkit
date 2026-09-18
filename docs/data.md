# Data

```bash
uv run visnavkit-dataset command=preprocess                      # validate clips, write manifests
uv run visnavkit-dataset command=visualize dataset=torch         # what the policy is actually fed
uv run visnavkit-dataset command=cache     dataset=torch         # window targets, no decode
uv run visnavkit-dataset command=stats     dataset=torch name=city   # normalizer NPZ + plots
uv run visnavkit-dataset command=anchors   dataset=torch num_anchors=64
```

`stats` and `anchors` read the cache and write the NPZs that
`model.action_decoder.normalizer.stats_path` and `model.action_decoder.anchors.anchors_path`
load, named after `name=` so each corpus keeps its own statistics. The tool shares `train.yaml`,
so its windows, action space and goal type are the ones training uses.

## Clip layout

A clip is a directory with `video.mp4` and `frame_times.npy` (int64 ns, increasing),
`frame_positions.npy` (N, 3) metres, `frame_orientations.npy` (N, 4) wxyz quaternions and
`frame_speeds.npy` (N,). Manifests list `video_path label start end` rows, `end` exclusive,
relative paths against `common.data_root`. Targets are interpolated at fixed relative times, so
any frame rate works; windows without enough future are dropped at index time. Point and image
goals are sampled from the clip's own future; `route_images.npy` and `instruction_embedding.npy`
are optional sidecars. The dataset reads its goal type from the selected goal encoder.

Opt-in inputs:

- `common.ego_features=[speed,yaw_rate,past_xy]` packs per-frame signals into `ego`
  (`past_xy` is each frame's position in the newest frame — past odometry); match
  `model.modality_encoders.ego.in_dim` to the width.
- `common.use_camera=true` reads `camera_intrinsics.npy` (3, 3) or (N, 3, 3) and
  `camera_extrinsics.npy` (4, 4) or (N, 4, 4) camera-to-ego, and keeps the calibration consistent
  with crop, downscale and flips.

`dataset=torch` decodes on CPU (every goal type and input); `dataset=dali` decodes on GPU (point
goals, speed-only ego). `visnavkit-train-route route=ae|vae` trains a route-patch autoencoder on the
`route_images.npy` sidecars alone (`dataset=route`, no decode); its checkpoint seeds a policy with
`model/goal_encoder=route_image model.goal_encoder.weights=<ckpt>`. Tiny example corpora, fetched into [`assets/datasets/`](../assets/), back the tests.

`dataset=pose` puts the window on a fixed slot grid: the ego state at `common.seq_length` slots `hz`
apart ending at the current frame (`past_xy`, `yaw`, `speed`, `yaw_rate` in the current frame, packed
into `ego`), every slot's future poses (`pose_size` 2 | 3 | 5 = x, y | x, y, v | x, y, yaw, v, w) and
point / gps goals, one window every `stride_s` seconds, so every source frame rate yields the same
window. `frames: true` adds `vision` — the source frame nearest each slot when it lies within half a
slot, zeros otherwise — with `frame_mask`; `route_hw` adds `route_patch` + `route_mask` from a
`route_labels.npy` (N, h, w) uint8 class-id sidecar; `embodiment_ids` {corpus dir: id} and an
`action_bounds` JSON {corpus dir: [[lo x 5], [hi x 5]]} add `embodiment_id` and `action_bounds` (the
corpus is the clip's first directory under `data_root`). `common.uniform_t_anchors=true` puts the
`plan_len_points` anchors on a fixed rate (`plan_len_seconds / plan_len_points` s apart) instead of the
quadratic grid, in every dataset and the action space alike. `model=flowpilot_sts` trains on these
windows (`experiment=flowpilot_sts_tiny`).

## Public corpora

Each needs a one-off conversion into the clip layout; converters are not bundled, since the
sources differ too much. Each corpus keeps its own licence — VisNavKit's MIT covers this code
only.

| Corpus | Content | Licence | Source |
| --- | --- | --- | --- |
| FrodoBots-2K | ~2000 h teleoperated sidewalk driving in 10+ cities; RGB, GPS, IMU, control | CC BY-SA 4.0 | [BitRobot/FrodoBots-2K](https://huggingface.co/datasets/BitRobot/FrodoBots-2K) |
| EgoWalk | 50+ h egocentric walking, indoor and outdoor; RGB, depth, ZED poses, language goals | MIT | [EgoWalk/trajectories](https://huggingface.co/datasets/EgoWalk/trajectories) |
| NVIDIA PhysicalAI AV | ~1700 h driving in 25 countries; 306k 20 s clips, 7 cameras, LiDAR, radar | NVIDIA AV Dataset Agreement | [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) |
| OpenScene | 120+ h of nuPlan driving at 2 Hz; 8 cameras, occupancy labels | CC BY-NC-SA 4.0 + nuPlan agreement | [OpenDriveLab/OpenScene](https://github.com/OpenDriveLab/OpenScene) |
| RECON | off-road exploration with goal images | MIT | [project](https://sites.google.com/view/recon-robot/dataset) |
| SCAND | socially compliant teleoperated navigation | CC0 1.0 | [Texas Data Repository](https://doi.org/10.18738/T8/0PRYRH) |
| GoStanford2 | indoor trajectories, the ViNT release | CC BY-NC-SA 3.0 | [GO Stanford](https://cvgl.stanford.edu/gonet/dataset/), [ViNT copy](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing) |
| SACSoN / HuRoN | indoor navigation among people | MIT | [project](https://sites.google.com/view/sacson-review/huron-dataset) |

The last four are ViNT's public training set, listed in
[visualnav-transformer](https://github.com/robodhruv/visualnav-transformer).
