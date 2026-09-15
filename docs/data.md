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
goals, speed-only ego). Tiny bundled corpora in [`assets/datasets/`](../assets/) back the tests.

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
| RECON | off-road exploration with goal images | see source | [project](https://sites.google.com/view/recon-robot/dataset) |
| SCAND | socially compliant teleoperated navigation | see source | [project](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links) |
| GoStanford2 | indoor trajectories, the ViNT release | see source | [download](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing) |
| SACSoN / HuRoN | indoor navigation among people | see source | [project](https://sites.google.com/view/sacson-review/huron-dataset) |

The last four are ViNT's public training set, listed in
[visualnav-transformer](https://github.com/robodhruv/visualnav-transformer).
