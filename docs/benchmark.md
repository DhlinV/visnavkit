# Open-loop benchmark

Full-window ONNX Runtime latency and trajectory error at the last observed frame. Native recipes
are architecture adaptations with random weights unless a checkpoint is supplied; published
graphs come from the [model catalog](models.md). Closed-loop simulation is out of scope.

```bash
uv run visnavkit-benchmark command=smoke model=gnm 'common.crop_wh=[32,32]' \
  common.downscale_factor=1 common.seq_length=2 samples=2 output_dir=outputs/benchmark/smoke
uv run visnavkit-benchmark command=evaluate adapter.name=constant_velocity \
  dataset_archive=outputs/benchmark/smoke/fixture.npz output_dir=outputs/benchmark/control
```

The first generates nonzero observations, exports, checks parity, profiles and evaluates two
synthetic trajectories; the second scores zero ADE/FDE on those constant-speed targets. Both are
labelled `pipeline_check`. Every command reads
[benchmark.yaml](../visnavkit/configs/benchmark.yaml) and shares Hydra overrides with training.

| Command | Does |
| --- | --- |
| `list` | catalog, native recipes, registered predictors |
| `download` | pinned, hash-verified published bundle |
| `fixture` | NPZ observations with analytic trajectories |
| `prepare` | NPZ shard from a validation manifest |
| `export` | native full-context FP32 graph, metadata, parity inputs |
| `profile` | latency, artifact size, declared FLOP coverage |
| `evaluate` | timestamp-based metrics and `predictions.npz` |
| `suite` | per-model profile (+ evaluate); failures recorded per model |
| `smoke` | fixture, export, profile, evaluate in one go |

## Native graphs

```bash
uv run visnavkit-benchmark command=export model=gnm checkpoint=/path/gnm.ckpt output_dir=outputs/benchmark/gnm
uv run visnavkit-benchmark command=profile model=gnm artifact=outputs/benchmark/gnm/gnm.onnx \
  runtime.warmup=10 runtime.iterations=100 output_dir=outputs/benchmark/gnm_timing
uv run visnavkit-benchmark command=suite 'suite.models=[resnet18,gnm,vint,nomad]' \
  '+suite.checkpoints.gnm=/path/gnm.ckpt' dataset_archive=outputs/benchmark/validation.npz \
  output_dir=outputs/benchmark/suite
```

A checkpoint restores its own model config, which is checked against the requested recipe. Keep
`.metadata.json` and `.inputs.npz` beside the graph; hashes are checked on reuse. The graph runs
every observed frame, makes one decision per window, and emits `trajectories (N,K,T,D)`,
`scores (N,K)` and `speed`. Goal-conditioned recipes export with their learned null goal token
(`conditioning: null_goal_token` in the metadata). Diffusion takes explicit initial noise and
times every candidate and step. Independent-sample evaluation needs batch-one exports.
`suite.fail_fast=true` stops on the first failed model; otherwise the aggregate report carries
`status: failed` per model.

## Published graphs

```bash
uv run visnavkit-benchmark command=download model_id=gnm
uv run visnavkit-benchmark command=profile source=published model_id=gnm output_dir=outputs/benchmark/published_gnm
```

Inputs are deterministic synthetic tensors unless `input_archive=/path/feeds.npz` supplies real
feeds by ONNX input name; dynamic dimensions take `+runtime.shapes.INPUT_NAME=[...]`. Published
NoMaD profiling runs the vision encoder, ten denoising calls and host DDPM updates
(`sampling.num_candidates=8`); its distance head is available through
`benchmark_nomad(..., include_distance=True)`. Published bundles still need reference parity,
preprocessing, scaling and output-timestamp validation before any model-quality comparison —
see each entry in [models.md](models.md).

## Real data

```bash
uv run visnavkit-benchmark command=prepare dataset=torch samples=100 \
  dataset_archive=outputs/benchmark/validation.npz output_dir=outputs/benchmark/prepare
uv run visnavkit-benchmark command=evaluate model=gnm checkpoint=/path/gnm.ckpt \
  dataset_archive=outputs/benchmark/validation.npz output_dir=outputs/benchmark/gnm_quality
```

`prepare` reads the training clip layout ([data guide](data.md)) with augmentation off, keeps
only windows with full future coverage, and takes the last context frame as decision time;
crop, context and targets must match the trained policy. `samples` picks the first valid windows
in manifest order (`samples=null` for all, up to 512 MiB uncompressed). Reports record manifest,
sample IDs, pose hashes, video size/mtime, loader settings and archive SHA-256. Untrained weights
on real data need `allow_untrained=true` and stay labelled pipeline checks.

Custom NPZ archives:

| Array | Shape and meaning |
| --- | --- |
| `targets` | `(N,T,2)` XY metres or `(N,T,3)` XY + speed m/s |
| `target_times_s` | `(T,)` or `(N,T)` increasing seconds after the decision |
| `sample_ids` | `(N,)` unique strings |
| `frames` | `(N,S,6,H,W)` uint8, converted to float32 `[0,1]` |
| `input__NAME` | prepared input for ONNX input `NAME`, leading sample axis |
| `current_speed` | `(N,)` ego forward speed for the constant-velocity control |
| `valid_mask` | optional `(N,T)` contiguous valid prefix |

Generic ONNX evaluation needs `adapter.prediction_times_s`, `adapter.output_name`,
`adapter.layout=NKTD|NTD` and `adapter.xy_scale` to metres; a native metadata sidecar supplies
them. Set `adapter.speed_index` only when that channel is speed.

## Protocol

Every run writes `result.json` (resolved config, Git revision, fingerprints) and `results.csv`.

- Timing is synchronous `session.run` including host I/O; session construction, preprocessing,
  decoding and warmup are excluded. Reports carry shapes, provider, threads, machine metadata,
  every latency sample, mean/std, p50/p90/p95/p99 and throughput. A CUDA request fails rather
  than falling back to CPU.
- Parameters are reported separately from ONNX initializer elements and artifact bytes. A
  multiply-accumulate is two FLOPs; unsupported operators are disclosed and incomplete totals are
  `null`.
- ADE averages displacement at the positive dataset anchors plus the requested horizon; FDE
  interpolates to that horizon. Samples without full coverage are excluded from that horizon with the count
  reported; a horizon with no coverage has `count: 0`, never zero error. Metrics include
  mean-sample and min-of-K (oracle) ADE/FDE, top-1 when scores exist, and speed MAE when both
  sides expose speed.
- Compare only the same sample IDs, target schedule, horizons, candidate count and goal/modality
  cohort; a missing modality is never silently zero-filled.

Register a predictor with `PREDICTORS.register("name")` — `predict(data, times)` returns
`(N,K,T,D)` trajectories and optional `(N,K)` scores in metres. Hydra `hooks` instantiate
observers with `on_start`, `on_result`, `on_error`. Tests cover analytic trajectories, timestamp
alignment, multimodal metrics, cached/full-context parity, runtime accounting, catalog integrity
and the smoke/control CLI, all without downloads.
