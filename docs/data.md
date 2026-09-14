# Data

Prepare, inspect and measure a corpus with one entry point; everything after `cache` reads the
cached window targets instead of decoding video again:

```bash
uv run visnavkit-dataset command=preprocess                      # validate clips, write manifests
uv run visnavkit-dataset command=visualize dataset=torch         # what the policy is actually fed
uv run visnavkit-dataset command=cache     dataset=torch         # window targets, no decode
uv run visnavkit-dataset command=stats     dataset=torch name=city   # normalizer NPZ + plots
uv run visnavkit-dataset command=anchors   dataset=torch num_anchors=16
```

`stats` writes the NPZ that `model.action_decoder.normalizer.stats_path` reads and `anchors` the
one `model.action_decoder.anchors.anchors_path` reads. Both are named after `name`, so several
corpora keep separate statistics — which is what a multi-dataset run needs, since mixing
datasets must not mean mixing their normalization.

Each clip is a directory with `video.mp4` and NumPy sidecars: `frame_times.npy` (int64 ns,
strictly increasing), `frame_positions.npy` (N, 3) metres, `frame_orientations.npy` (N, 4)
wxyz quaternions, `frame_speeds.npy` (N,). Manifests list `video_path label start end` rows.
Targets are interpolated at fixed relative times, so any frame rate works. Point and image
goals are sampled from the clip's own future; `route_images.npy` and
`instruction_embedding.npy` are optional. The dataset reads the goal type from the selected
goal encoder. Details: [benchmark guide](benchmark.md#prepare-and-evaluate-real-data).

Opt-in modality inputs:

- `common.ego_features=[speed,yaw_rate,past_xy]` packs those per-frame signals into `ego`
  (`past_xy` is each observed frame's position in the newest frame, i.e. past odometry); match
  `model.modality_encoders.ego.in_dim` to its width.
- `common.use_camera=true` reads `camera_intrinsics.npy` (3, 3) or (N, 3, 3) and
  `camera_extrinsics.npy` (4, 4) or (N, 4, 4) camera-to-ego, then shifts the principal point
  by the crop, divides by the downscale and mirrors it on a flip, so the calibration always
  describes the image the policy sees.

`dataset=torch` decodes on CPU; `dataset=dali` decodes on GPU (point goals, speed-only ego).

### Public corpora

Recorded sidewalk and off-road navigation datasets this format targets. Each needs a one-off
conversion into the clip layout above; `visnavkit-dataset command=preprocess` validates the
result. Converters are not bundled — the sources differ too much to guess at.

| Corpus | Content | Licence | Source |
| --- | --- | --- | --- |
| FrodoBots-2K | ~2000 h teleoperated sidewalk driving in 10+ cities; RGB, GPS, IMU, audio, control | CC BY-SA 4.0 | [BitRobot/FrodoBots-2K](https://huggingface.co/datasets/BitRobot/FrodoBots-2K) |
| RECON | Off-road exploration with goal images | see source | [project](https://sites.google.com/view/recon-robot/dataset) |
| SCAND | Socially compliant human-teleoperated navigation | see source | [project](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links) |
| GoStanford2 | Indoor trajectories, the ViNT-modified release | see source | [download](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing) |
| SACSoN / HuRoN | Indoor navigation among people | see source | [project](https://sites.google.com/view/sacson-review/huron-dataset) |

The last four are ViNT's public training set, listed in
[visualnav-transformer](https://github.com/robodhruv/visualnav-transformer). Only FrodoBots-2K
states a licence machine-readably; for the rest, read the terms on the source page before using
them — each corpus keeps its own, and VisNavKit's MIT licence covers this code only, never the
data or any third-party weights. Tiny bundled corpora live in [`assets/datasets/`](../assets/) and
are exercised by `tests/data/test_assets.py`.
