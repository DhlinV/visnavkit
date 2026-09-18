# assets

Small artifacts that travel with the repository.

## `datasets/`

Tiny corpora — three face-blurred clips from each of 11 public datasets, 11 MB — used to check
that the policy runs on real data rather than on random tensors. They are not checked in; fetch
them from [UCLA-VAIL/visnavkit-examples](https://huggingface.co/datasets/UCLA-VAIL/visnavkit-examples),
whose dataset card lists every source clip and its licence:

```bash
uv run hf download UCLA-VAIL/visnavkit-examples --repo-type dataset --local-dir assets/datasets
```

Each subdirectory is one corpus in the standard clip layout:

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

`assets/datasets/` is gitignored; anything beyond a few megabytes belongs outside the repository,
behind `VISNAVKIT_DATA_ROOT`.
