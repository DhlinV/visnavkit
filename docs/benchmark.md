# Open-loop benchmark

The benchmark measures full-window ONNX Runtime inference and trajectory error
at the final observed frame. Native recipes are architecture adaptations; the
[published model catalog](models.md) has separate provenance and input contracts.
Closed-loop simulation and trained model rankings are outside this workflow.

## Quickstart

Install with `uv sync`. CPU ONNX Runtime is included; DALI, W&B, and deployment
export slimming are optional extras (`dali`, `wandb`, and `export`). The benchmark
exporter needs no slimming extra. Native benchmark recipes default to random
weights with pretrained downloads disabled.

```bash
uv run python -m visnavkit.scripts.benchmark command=smoke model=gnm \
  common.crop_wh=[32,32] common.downscale_factor=1 common.seq_length=2 \
  samples=2 runtime.warmup=1 runtime.iterations=2 output_dir=outputs/benchmark/smoke

uv run python -m visnavkit.scripts.benchmark command=evaluate \
  adapter.name=constant_velocity \
  dataset_archive=outputs/benchmark/smoke/fixture.npz \
  output_dir=outputs/benchmark/control
```

The first command generates nonzero observations, exports a full-context policy,
checks numerical parity, profiles it, and evaluates two synthetic trajectories.
The second should produce zero ADE/FDE on those straight constant-speed targets.
Both reports are labeled `pipeline_check`. Use realistic image sizes, context,
warmup, and iteration counts for performance measurements.

`uv run visnavkit-benchmark ...` is the installed console entry point. All
commands use [benchmark.yaml](../visnavkit/configs/benchmark.yaml); Hydra overrides
are shared with training. Use separate output directories to preserve each run.

| Command | Output or behavior |
| --- | --- |
| `list` | Catalog, native recipe names, registered predictor names |
| `download` | Explicit download of a pinned, hash-verified published artifact bundle |
| `fixture` | Generated NPZ observations and analytic trajectories |
| `prepare` | Bounded NPZ shard from a validation video manifest |
| `export` | Native full-context FP32 ONNX graph, metadata, and parity inputs |
| `profile` | ONNX latency, artifact sizes, and declared FLOP coverage |
| `evaluate` | Timestamp-based trajectory metrics and `predictions.npz` |
| `suite` | Per-model profiling and optional evaluation; failures recorded individually |
| `smoke` | Native fixture, export, profile, and evaluation in one command |

## Native export and profiling

```bash
uv run python -m visnavkit.scripts.benchmark command=export model=gnm \
  output_dir=outputs/benchmark/gnm

uv run python -m visnavkit.scripts.benchmark command=profile model=gnm \
  artifact=outputs/benchmark/gnm/resnet18.onnx \
  runtime.warmup=10 runtime.iterations=100 output_dir=outputs/benchmark/gnm_timing
```

Supply `checkpoint=/path/model.ckpt` to export trained weights and compose the
checkpoint's original model configuration. Saved model configuration is checked
against the requested recipe to prevent mislabeling. Keep `.metadata.json` and
`.inputs.npz` beside the graph; metadata hashes are checked on reuse.

The exporter runs all observed frames and makes one decision per window. It
exports fixed batch and context dimensions with `trajectories (N,K,T,D)`,
`scores (N,K)`, and auxiliary `speed`. Goal-conditioned recipes are exported with
their learned null goal token (goal-free inference, as in NoMaD exploration) and the
metadata records `conditioning: null_goal_token`; prepared datasets carry no goals yet. Native diffusion has explicit initial
noise and includes every candidate and denoising step. Independent-sample
evaluation requires batch-one exports. The existing deployment export command
uses a feature buffer; those timings have a different scope.

```bash
uv run python -m visnavkit.scripts.benchmark command=suite \
  'suite.models=[resnet18,gnm,vint,nomad]' \
  common.crop_wh=[64,64] common.downscale_factor=1 common.seq_length=2 \
  runtime.warmup=2 runtime.iterations=5 output_dir=outputs/benchmark/native_suite
```

For trained suites, provide checkpoints with
`+suite.checkpoints.gnm=/path/gnm.ckpt` and equivalent entries for each recipe.
Set `dataset_archive=...` to score the same prepared split. A failed model has
`status: failed` and an error in the aggregate report; `suite.fail_fast=true`
makes such failures terminate the command. A successful suite process alone
does not mean every model succeeded.

## Published graphs

```bash
uv run python -m visnavkit.scripts.benchmark command=list
uv run python -m visnavkit.scripts.benchmark command=download model_id=gnm
uv run python -m visnavkit.scripts.benchmark command=profile source=published \
  model_id=gnm output_dir=outputs/benchmark/published_gnm
```

Downloads are explicit and retryable. Profiling uses locally downloaded graphs
and checks their pinned identities. `input_archive=/path/feeds.npz` supplies
exact ONNX input names and NumPy dtypes. Otherwise inputs are deterministic
synthetic tensors, suitable for execution checks. Dynamic dimensions require
`+runtime.shapes.INPUT_NAME=[...]`; published graphs automatically receive the
configured batch size for a dynamic leading dimension. Additional modalities
must be prepared according to the model contract for meaningful inference.

Published NoMaD profiling runs its vision encoder, ten denoising calls, and host
DDPM updates. `sampling.num_candidates=8` is the default. Prepared encoder feeds
are accepted through `input_archive`. Noise generation and metric trajectory
decoding are excluded; the optional distance head is available through the
Python `benchmark_nomad(..., include_distance=True)` API. Individual component
graphs can be profiled with `artifact=...`, but are labeled supplied-graph timing.
Generic trajectory evaluation does not implement NoMaD's full metric adapter.

The published bundles still require reference parity, preprocessing, scaling,
and output timestamp validation before model-quality comparison. See each
entry in [models.md](models.md). NavDP has no verified downloadable ONNX bundle.

## Prepare and evaluate real data

```bash
uv run python -m visnavkit.scripts.benchmark command=prepare dataset=torch \
  common.data_root=/data/nav_clips samples=100 \
  dataset_archive=outputs/benchmark/validation.npz \
  output_dir=outputs/benchmark/prepare

uv run python -m visnavkit.scripts.benchmark command=evaluate model=gnm \
  checkpoint=/path/gnm.ckpt dataset_archive=outputs/benchmark/validation.npz \
  output_dir=outputs/benchmark/gnm_quality
```

The torch loader needs FFmpeg and TorchCodec. Validation manifest rows contain
`video_path label start end`, with frame ranges `[start,end)`. Episode directories
contain `frame_positions.npy`, `frame_orientations.npy` (wxyz quaternions),
`frame_times.npy` (nanoseconds), and `frame_speeds.npy`. Relative paths resolve
against `common.data_root`. Preparation disables augmentation, keeps only valid
observation/future windows, and uses the final context frame as the decision time.
Observation cropping, context, and target settings must match the trained policy.

`samples` selects the first valid windows in manifest order (default four);
`samples=null` selects all, subject to the 512 MiB uncompressed shard limit.
Reports record the manifest, selected sample IDs, pose hashes, video size/mtime,
resolved loader settings, and archive SHA-256. Video bytes are not fully hashed.
Untrained weights on real data require `allow_untrained=true` and remain labeled
as pipeline checks.

Custom NPZ archives have the following contract:

| Array | Shape and meaning |
| --- | --- |
| `targets` | `(N,T,2)` XY meters or `(N,T,3)` XY plus speed in m/s |
| `target_times_s` | `(T,)` or `(N,T)` increasing times relative to the decision |
| `sample_ids` | `(N,)` unique strings |
| `frames` | Native `(N,S,6,H,W)` uint8, converted to float32 `[0,1]` |
| `input__NAME` | Prepared input for ONNX input `NAME`, with a leading sample axis |
| `current_speed` | `(N,)` ego forward speed for the constant-velocity control |
| `valid_mask` | Optional `(N,T)` contiguous valid prefix per sample |

Generic ONNX evaluation requires explicit `adapter.prediction_times_s`,
`adapter.output_name`, `adapter.layout=NKTD` or `NTD`, and the correct
`adapter.xy_scale` to meters. A matching native metadata sidecar supplies output
times and selection. Only specify `adapter.speed_index` when that channel is
speed; yaw and Z are not speed. For native diffusion, `input__initial_noise`
has shape `(N,K,trajectory_dimension)` and is generated from the run seed if absent.
Control predictors currently require shared one-dimensional target times.

## Measurement protocol and reports

Every CLI run writes versioned `result.json` and flattened `results.csv` through
the report hook. The JSON includes resolved configuration and Git revision/dirty
state. Exports add checkpoint/graph fingerprints and nonzero-input parity errors;
evaluation saves predictions, sample IDs, dataset fingerprint, and per-horizon counts.

- Timing covers synchronous `session.run`, including host inputs/outputs,
  transfers, and output allocation. Session construction, preprocessing, video
  decoding, input generation, and warmup are excluded. Identical feeds are reused.
- Reports include input shapes/dtypes, batch, provider, threads, machine/library
  metadata, each latency sample, mean, standard deviation, p50/p90/p95/p99, and
  throughput. CUDA requests fail if the provider is unavailable or needs CPU fallback.
- Source total/trainable parameters are separate from ONNX initializer elements
  and artifact bytes. Conv/Gemm/MatMul multiply-accumulates count as two FLOPs.
  Unsupported arithmetic and unknown shapes are disclosed; incomplete totals
  are `null`. PyTorch's registered-operator count is also only a subtotal.
- ADE averages displacement at positive dataset anchors plus the requested
  horizon. FDE interpolates to that exact horizon. Samples without coverage of
  every required point are excluded from that horizon, with the count reported.
- Metrics include mean-sample and independently minimized ADE/FDE. Min-of-K is
  an oracle metric. Top-1 needs declared scores or a single candidate. Speed MAE
  appears only when both predictions and targets expose speed.
- Each eligible sample contributes equally. Compare the same sample IDs, target
  schedule, horizons, candidate count, and compatible goal/modality cohorts.
  A horizon with no coverage has `count: 0` and absent errors, never zero error.

## Extension and verification

Register a predictor via `PREDICTORS.register("name")`. Its `predict(data, times)`
returns trajectories `(N,K,T,D)` and optional scores `(N,K)` in metric units.
Hydra `hooks` can instantiate observers implementing `on_start`, `on_result`,
and `on_error`; hook failures propagate. Catalog registration adds metadata only
and does not establish inference compatibility.

```bash
uv sync --group dev --extra export
uv run ruff check visnavkit tests
uv run pytest tests/ -q
```

Tests cover analytic trajectories, timestamp alignment, validation tails,
multimodal metrics, cached/full-context parity, runtime accounting, catalog
integrity, and the actual smoke/control CLI. DALI parity needs its optional
dependency and a CUDA GPU; scheduler reference parity needs optional Diffusers.
CI is configured to run the CPU checks and CLI tests without downloading model weights.
