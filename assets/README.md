# assets

Small artifacts that travel with the repository.

## `datasets/`

Tiny corpora — a handful of windows each — used to check that the policy runs on real data
rather than on random tensors. Each subdirectory is one corpus in the standard clip layout:

```text
assets/datasets/<corpus>/<clip>/video.mp4
                                frame_times.npy          (N,)   int64 nanoseconds, increasing
                                frame_positions.npy      (N, 3) metres
                                frame_orientations.npy   (N, 4) wxyz quaternions
                                frame_speeds.npy         (N,)   m/s
                                camera_intrinsics.npy    optional, (3, 3) or (N, 3, 3)
                                camera_extrinsics.npy    optional, (4, 4) or (N, 4, 4)
```

`tests/data/test_assets.py` picks up every corpus here automatically: it writes manifests,
builds the loader, and runs a forward pass through a small policy. Drop a corpus in and it is
covered; with none present the test skips.

```bash
uv run visnavkit-dataset command=preprocess common.data_root=assets/datasets/<corpus>
uv run visnavkit-dataset command=visualize dataset=torch common.data_root=assets/datasets/<corpus>
```

Keep these small — they are checked in. Anything beyond a few megabytes belongs outside the
repository, behind `VISNAVKIT_DATA_ROOT`.
