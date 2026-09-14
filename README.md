<h1 align="center">
  <img src="docs/assets/logo-options/d-arrow.png" alt="" width="64" align="absmiddle"> VisNavKit
</h1>

**A composable toolkit for visual navigation policies: train, export to ONNX, benchmark.**

Vision in, trajectory out — and an **open set of extra inputs** in between. Each stage is a
Hydra group and every stage exchanges tokens of one width, so any encoder works with any
goal and any decoder.

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
per-frame tokens that join the vision tokens. Ego status and camera calibration ship with
the repo; a spatial/depth/LiDAR/IMU stage is a subclass plus a config entry, and the policy,
the ONNX contract and the deployment feature buffer follow on their own. Everything except
vision is optional — an absent input falls back to that encoder's learned null token.

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

This prints the composed pipeline with tensor shapes, runs a training step, and checks
that the deployment feature-buffer path reproduces the full-window prediction. Add
`--onnx` to export the deployment graph and verify ONNX Runtime parity. The open-loop
benchmark has an equivalent self-contained check:

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

Three more knobs cut across the groups:

- **Tokens per frame**: `model.vision_encoder.token_mode=global|patch|fused` with
  `patch_grid=[4,4]`. Global is one token per frame; patch and fused keep a pooled
  spatial grid for the temporal encoder and decoder to attend over.
- **Action space**: `model.action_decoder.action_space.kind=waypoint|velocity`
  (ego-frame poses, or unicycle speed and yaw rate integrated back to poses), and
  `model.action_decoder.normalizer.mode=none|meanstd|minmax` with `stats_path` for
  normalized targets.
- **Auxiliary speed head**: `model.vision_encoder.speed_head=true` regresses each frame's
  speed. It is a per-recipe training signal, off by default, and adds a `speed` output to
  the exported graph.

```bash
uv run visnavkit-train dataset=torch model=vint \
  model/vision_encoder=dinov3_s model/goal_encoder=point model/action_decoder=flow_dit \
  model.vision_encoder.token_mode=fused common.data_root=/data/nav_clips
```

Generative decoders are a denoiser (`mlp`, `dit`, `unet`) times a scheduler (`ddim`,
`flow`), optionally seeded from a set of anchor trajectories; the named entries above are
just those combinations. The scheduler knobs follow their reference implementations, for
example `model.action_decoder.scheduler.time_sampling=beta` (openpi pi0) or `.shift=3`
(diffusers SD3/Flux) for flow matching. Every decoder emits the same flat `[mean, log_scale, confidence]`
layout per mode, so metrics, export, and the benchmark never change.

### Recipes

Recipes named after papers are architecture adaptations to this repo's data contract
(single RGB frames, fixed-horizon x/y/v targets, goals sampled from the episode's own future),
not reproductions or checkpoint-compatible replacements.

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
| `diffusion`, `flow_dit`, `anchor` | FastViT-T8 | causal x1 | none | diffusion MLP, flow DiT, anchor |

## Data

Each clip is a directory with `video.mp4` and NumPy sidecars: `frame_times.npy` (int64
ns, strictly increasing), `frame_positions.npy` (N, 3) metres, `frame_orientations.npy`
(N, 4) wxyz quaternions, `frame_speeds.npy` (N,). Manifests list `video_path label start end`
rows. Targets are interpolated at fixed relative times, so any frame rate works. Point and
image goals are sampled from the clip's own future; route images and instruction
embeddings come from optional `route_images.npy` and `instruction_embedding.npy` sidecars.
The dataset reads the goal type from the selected goal encoder. Details: [benchmark guide](docs/benchmark.md#prepare-and-evaluate-real-data).

The other two inputs are opt-in:

- `common.ego_features=[speed,yaw_rate]` packs those per-frame signals into `ego`; the
  encoder is width-agnostic, so any dataset-defined vector works as long as
  `model.modality_encoders.ego.in_dim` matches.
- `common.use_camera=true` reads `camera_intrinsics.npy` (3, 3) or (N, 3, 3) and
  `camera_extrinsics.npy` (4, 4) or (N, 4, 4) camera-to-ego. The dataset shifts the
  principal point by the crop, divides by the downscale, and mirrors it with a horizontal
  flip, so the calibration always describes the image the policy sees.

`dataset=torch` decodes on CPU; `dataset=dali` decodes on GPU (point goals, speed-only ego).

## Train, export, benchmark

```bash
uv run visnavkit-train dataset=torch model=base common.data_root=/data/nav_clips
uv run visnavkit-export checkpoint=logs/baseline/.../last.ckpt output=outputs/policy.onnx
uv run visnavkit-benchmark command=export model=resnet18 output_dir=outputs/benchmark/resnet18
```

Add `ema=default` to keep an exponential moving average of the weights (diffusers
`EMAModel` schedule, as in diffusion policy and NoMaD); validation, checkpoints and
export then use the averaged policy. Training records Git provenance (`strict_git=true`
requires a clean tree) and keeps experiment overrides in
[`configs/experiment/`](visnavkit/configs/experiment/). The
deployment graph takes `vision`, `feature_buffer`, and, when the recipe needs them, one
`goal` input per goal encoder, one input per key its modality encoders read, and `noise`; it returns
`plan`, `feat_out` and, with the auxiliary head, `speed`. Explicit noise makes generative
policies deterministic and lets export verify parity. The benchmark exporter runs the
full window instead and outputs `trajectories`, `scores` (plus `speed`).

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
- **S2E**: From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning — [arXiv:2507.22028](https://arxiv.org/abs/2507.22028), [project](https://vail-ucla.github.io/S2E/)
- **NWM**: Navigation World Models (CVPR 2025) — [arXiv:2412.03572](https://arxiv.org/abs/2412.03572)
- **MIMIC**: Learning Sidewalk Autopilot from Multi-Scale Imitation with Corrective Behavior Expansion (ICRA 2026) — [arXiv:2603.22527](https://arxiv.org/abs/2603.22527)
- **Diffusion Policy** (RSS 2023) — [arXiv:2303.04137](https://arxiv.org/abs/2303.04137); **DiT** — [arXiv:2212.09748](https://arxiv.org/abs/2212.09748); **Flow matching** — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **MetaUrban** (ICLR 2025) — [code](https://github.com/metadriverse/metaurban); **SidewalkBench** — [arXiv:2606.16953](https://arxiv.org/abs/2606.16953)

</details>
