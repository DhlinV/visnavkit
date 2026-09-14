<h1 align="center">
  <img src="docs/assets/logo-options/d-arrow.png" alt="" width="64" align="absmiddle"> VisNavKit
</h1>

**A composable toolkit for visual navigation policies: train, export to ONNX, benchmark.**

Vision in, trajectory out, with an **open set of extra inputs** in between. Each stage is a
Hydra group exchanging tokens of one width, so any encoder works with any goal and decoder.

```text
vision   (B, F, 3, H, W) ──▶ vision encoder ───┐
ego      (B, F, E)       ──▶ modality: ego ────┤  per-frame
K, RT    (B,F,3,3/4,4)   ──▶ modality: camera ─┤  tokens (F, K, D)
<yours>                  ──▶ modality: ... ────┘
                                               │
                                               ▼
                                        temporal encoder ──▶ action decoder ──▶ trajectory
                                          context (K, D)    ▲    plans (M, T, pose)
                                                            │
goal (one per goal encoder) ──▶ goal encoder(s) ────────────┘  goal tokens (G, D)
```

A **modality** is any non-image input: it declares the batch keys it reads and returns
per-frame tokens. Ego status and calibration ship with the repo; a spatial/depth/LiDAR stage
is a subclass plus a config entry, and the ONNX contract and feature buffer follow. Every
input but vision is optional and falls back to its encoder's learned null token.

[Architecture](docs/architecture.md) · [Benchmark guide](docs/benchmark.md) · [Model catalog](docs/models.md) · [Configs](visnavkit/configs/)

## Install

```bash
uv sync                      # CPU ONNX Runtime included; Linux resolves CUDA 13 torch wheels
uv sync --extra export       # + onnxslim for deployment graphs
uv sync --extra dali         # + NVIDIA DALI GPU video decoding (Linux)
```

The torch data loader decodes video with TorchCodec and needs **FFmpeg** on the system
(`apt install ffmpeg`). Python 3.10+.

## Quick start (no data, no downloads)

```bash
uv run visnavkit-sanity-check model=s2e model/action_decoder=flow_dit
```

Prints the composed pipeline with shapes, runs a training step, and checks that the
deployment feature-buffer path reproduces the full-window prediction; `--onnx` also exports
and verifies ONNX Runtime parity. The benchmark has an equivalent self-contained check:

```bash
uv run visnavkit-benchmark command=smoke model=gnm 'common.crop_wh=[32,32]' \
  common.downscale_factor=1 common.seq_length=2 samples=2 output_dir=outputs/benchmark/smoke
```

## Compose a policy

Pick one entry per group, or start from a recipe and override any group.

| Group | Choices |
| --- | --- |
| `model/vision_encoder` | `cnn` or `vit` with any timm backbone, plus presets: `fastvit_t8`, `fastvit_t12`, `resnet18`, `resnet50`, `efficientnet_b0`, `mobilenetv2`, `mobilenetv3`, `mobilenetv4`, `convnext_tiny`, `convnextv2_nano`, `regnety_008`, `repvit_m1`, `efficientvit_b0`, `dinov2_s`, `dinov2_b`, `dinov3_s`, `dinov3_b`, `vit_s`, `deit_s`, `eva02_s`, `siglip_b`, `clip_b` |
| `model/temporal_encoder` | `identity` (single frame), `causal`, `causal_4layer`, `bidirectional` |
| `model/goal_encoder` | `none`, `point` (distance, cos, sin), `image`, `route_image`, `instruction` (text embedding); set it to a **list** for several goals at once |
| `model/modality_encoder` | `none`, `ego` (free-form `(B, F, E)` vector), `camera` (per-frame intrinsics + extrinsics), `ego_camera`; add your own with `+model.modality_encoders.<name>=...` |
| `model/action_decoder` | `regression`, `mhp`, `anchor`, `diffusion_mlp`, `diffusion_dit`, `diffusion_unet`, `flow_mlp`, `flow_dit`, `flow_unet`, `anchor_diffusion_dit`, `anchor_flow_dit` |

Knobs that cut across the groups:

- `model.vision_encoder.token_mode=global|patch|fused` with `patch_grid=[4,4]` — one token
  per frame, or a pooled spatial grid for the temporal encoder and decoder to attend over.
- `model.action_decoder.action_space.kind=waypoint|velocity` (ego-frame poses, or unicycle
  speed and yaw rate integrated back to poses) and `.normalizer.mode=none|meanstd|minmax`.
- `model.vision_encoder.speed_head=true` — per-frame speed regression. A per-recipe
  auxiliary signal, off by default, adding a `speed` output to the exported graph.

```bash
uv run visnavkit-train dataset=torch model=vint \
  model/vision_encoder=dinov3_s model/goal_encoder=point model/action_decoder=flow_dit \
  model.vision_encoder.token_mode=fused common.data_root=/data/nav_clips
```

Generative decoders are a denoiser (`mlp`, `dit`, `unet`) times a scheduler (`ddim`, `flow`),
optionally anchored; the named entries are those combinations. Scheduler knobs follow their
reference implementations — `scheduler.time_sampling=beta` (openpi pi0), `.shift=3` (diffusers
SD3/Flux). Every decoder emits the same flat `[mean, log_scale, confidence]` layout per mode,
so metrics, export and the benchmark never change.

### Recipes

Paper-named recipes are architecture adaptations to this repo's data contract (single RGB
frames, fixed-horizon x/y/v targets, goals sampled from the episode's own future) — not
reproductions or checkpoint-compatible replacements.

| Recipe | Vision | Temporal | Goal | Decoder |
| --- | --- | --- | --- | --- |
| `base`, `mimic` | FastViT-T8 | causal x1 | none | MHP |
| `resnet18` | ResNet18 | single frame | none | regression |
| `gnm` | MobileNetV2 | single frame | image (stacked with observation) | regression |
| `vint` | EfficientNet-B0 | causal x4 | image (stacked) | regression |
| `nomad` | EfficientNet-B0 | causal x4 | image, 50% goal dropout | diffusion U-Net |
| `citywalker` | frozen DINOv2 ViT-S | causal x4 | point | regression |
| `s2e` | frozen DINOv3 ViT-S | causal x1 | point, 50% goal dropout | MHP |
| `dinov2`, `dinov3` | frozen DINO ViT-S | causal x1 | none | MHP |
| `flowpilot` | FastViT-T8 + speed head | causal x1 | point | anchored flow DiT, Beta(1.5, 1) times |
| `diffusion`, `flow_dit`, `anchor` | FastViT-T8 | causal x1 | none | diffusion MLP, flow DiT, anchor |

## Data

Each clip is a directory with `video.mp4` and NumPy sidecars: `frame_times.npy` (int64 ns,
strictly increasing), `frame_positions.npy` (N, 3) metres, `frame_orientations.npy` (N, 4)
wxyz quaternions, `frame_speeds.npy` (N,). Manifests list `video_path label start end` rows.
Targets are interpolated at fixed relative times, so any frame rate works. Point and image
goals are sampled from the clip's own future; `route_images.npy` and
`instruction_embedding.npy` are optional. The dataset reads the goal type from the selected
goal encoder. Details: [benchmark guide](docs/benchmark.md#prepare-and-evaluate-real-data).

Opt-in modality inputs:

- `common.ego_features=[speed,yaw_rate]` packs those per-frame signals into `ego`; match
  `model.modality_encoders.ego.in_dim` to its width.
- `common.use_camera=true` reads `camera_intrinsics.npy` (3, 3) or (N, 3, 3) and
  `camera_extrinsics.npy` (4, 4) or (N, 4, 4) camera-to-ego, then shifts the principal point
  by the crop, divides by the downscale and mirrors it on a flip, so the calibration always
  describes the image the policy sees.

`dataset=torch` decodes on CPU; `dataset=dali` decodes on GPU (point goals, speed-only ego).

## Train, export, benchmark

```bash
uv run visnavkit-train dataset=torch model=base common.data_root=/data/nav_clips
uv run visnavkit-export checkpoint=logs/baseline/.../last.ckpt output=outputs/policy.onnx
uv run visnavkit-benchmark command=export model=resnet18 output_dir=outputs/benchmark/resnet18
```

`ema=default` keeps an exponential moving average of the weights (diffusers `EMAModel`
schedule, as in diffusion policy and NoMaD), which validation, checkpoints and export then
use. Training records Git provenance (`strict_git=true` requires a clean tree) and keeps
experiment overrides in [`configs/experiment/`](visnavkit/configs/experiment/).

The deployment graph takes `vision`, `feature_buffer`, then one `goal` input per goal
encoder, one per modality key, and `noise` — whichever the recipe uses — and returns `plan`,
`feat_out`, plus `speed` with the auxiliary head. Explicit noise makes generative policies
deterministic and lets export verify parity. The benchmark exporter runs the full window
instead and outputs `trajectories`, `scores` (plus `speed`).

## Development

```bash
uv run pytest tests/ -q
uv run ruff check visnavkit tests
uv run python -m visnavkit.scripts.smoke_forward model=nomad
```

Layout: [`models/vision`](visnavkit/models/vision/) · [`models/temporal`](visnavkit/models/temporal/) ·
[`models/goal`](visnavkit/models/goal/) · [`models/modality`](visnavkit/models/modality/) ·
[`models/action`](visnavkit/models/action/) ·
[`models/policy.py`](visnavkit/models/policy.py) · [`data/`](visnavkit/data/) ·
[`benchmark/`](visnavkit/benchmark/) · [`evaluation/`](visnavkit/evaluation/) · [`scripts/`](visnavkit/scripts/)

Use it from another project with `uv add --editable /path/to/visnavkit`; the configs
ship inside the package as `visnavkit.configs`.

<details>
<summary>Research references</summary>

- **GNM**: A General Navigation Model to Drive Any Robot (ICRA 2023) — [arXiv:2210.03370](https://arxiv.org/abs/2210.03370), [code](https://github.com/robodhruv/drive-any-robot)
- **ViNT**: A Foundation Model for Visual Navigation (CoRL 2023) — [arXiv:2306.14846](https://arxiv.org/abs/2306.14846), [code](https://github.com/robodhruv/visualnav-transformer)
- **NoMaD**: Goal Masked Diffusion Policies for Navigation and Exploration (ICRA 2024) — [arXiv:2310.07896](https://arxiv.org/abs/2310.07896), [code](https://github.com/robodhruv/visualnav-transformer)
- **ViKiNG**: Vision-Based Kilometer-Scale Navigation with Geographic Hints (RSS 2022) — [arXiv:2202.11271](https://arxiv.org/abs/2202.11271)
- **CityWalker**: Learning Embodied Urban Navigation from Web-Scale Videos (CVPR 2025) — [code](https://github.com/ai4ce/CityWalker)
- **S2E**: From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning — [arXiv:2507.22028](https://arxiv.org/abs/2507.22028), [code](https://github.com/VAIL-UCLA/S2E)
- **FlowPilot**: From Imitation to Alignment: Human-Preference Flow Policies for Long-Horizon Sidewalk Navigation (CoRL 2026) — [arXiv:2606.12603](https://arxiv.org/abs/2606.12603), [project](https://vail.cs.ucla.edu/FlowPilot), [code](https://github.com/VAIL-UCLA/FlowPilot)
- **NWM**: Navigation World Models (CVPR 2025) — [arXiv:2412.03572](https://arxiv.org/abs/2412.03572)
- **MIMIC**: Learning Sidewalk Autopilot from Multi-Scale Imitation with Corrective Behavior Expansion (ICRA 2026) — [arXiv:2603.22527](https://arxiv.org/abs/2603.22527), [code](https://github.com/VAIL-UCLA/MIMIC)
- **Diffusion Policy** (RSS 2023) — [arXiv:2303.04137](https://arxiv.org/abs/2303.04137); **DiT** — [arXiv:2212.09748](https://arxiv.org/abs/2212.09748); **Flow matching** — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **MetaUrban** (ICLR 2025) — [code](https://github.com/metadriverse/metaurban); **SidewalkBench** — [arXiv:2606.16953](https://arxiv.org/abs/2606.16953)

</details>
