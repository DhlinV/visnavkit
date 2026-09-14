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

[Architecture](docs/architecture.md) · [Data](docs/data.md) · [Models & weights](docs/models.md) · [Benchmark](docs/benchmark.md) · [Configs](visnavkit/configs/)

## Roadmap

*Adapted* means the architecture composes, trains and exports here; *reproduced* means matching
the paper's reported numbers, which needs the corpus and a training run.

| | Progress |
| --- | --- |
| Architectures adapted | `██████████` 11/11 |
| Corpus tooling: preprocess, cache, stats, anchors, visualize | `██████████` 5/5 |
| Export, ONNX parity, open-loop benchmark | `██████████` 3/3 |
| Multi-dataset training: per-corpus statistics, then a loader that mixes them | `█████░░░░░` 1/2 |
| Dataset converters: FrodoBots-2K, EgoWalk, NVIDIA PhysicalAI AV, OpenScene, RECON, SCAND, GoStanford2, SACSoN | `░░░░░░░░░░` 0/8 |
| Reproducing GNM / ViNT / NoMaD | `░░░░░░░░░░` 0/3 |
| Reproducing CityWalker / MBRA / NavDP | `░░░░░░░░░░` 0/3 |
| Reproducing S2E / SocialNav / InternVLA-N1 | `░░░░░░░░░░` 0/3 |
| Reproducing MIMIC / FlowPilot | `░░░░░░░░░░` 0/2 |
| Released VisNavKit checkpoints, one per recipe | `░░░░░░░░░░` 0/11 |

## Install

```bash
uv sync                      # CPU ONNX Runtime included; Linux resolves CUDA 13 torch wheels
uv sync --extra export       # + onnxslim for deployment graphs
uv sync --extra dali         # + NVIDIA DALI GPU video decoding (Linux)
```

The torch data loader decodes video with TorchCodec and needs **FFmpeg** on the system
(`apt install ffmpeg`). Python 3.10+.

Point the configs at your corpus once, instead of passing it to every command:

```bash
echo 'export VISNAVKIT_DATA_ROOT=/data/nav_clips' >> ~/.bashrc && source ~/.bashrc
```

`common.data_root` reads it (`${oc.env:VISNAVKIT_DATA_ROOT,/data/nav_clips}`), so
`common.data_root=...` on the command line stays available as an override.

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
| `model/vision_encoder` | `cnn` or `vit` with any timm backbone; 22 presets from `fastvit_t8` to `dinov3_b` in [`configs/model/vision_encoder/`](visnavkit/configs/model/vision_encoder/) |
| `model/temporal_encoder` | `identity` (single frame), `causal`, `causal_4layer`, `bidirectional` |
| `model/goal_encoder` | `none`, `point` (distance, cos, sin), `gps` (local x/y metres), `image`, `route_image`, `instruction` (text embedding), `gps_route_image`; set it to a **list** for any other combination |
| `model/modality_encoder` | `none`, `ego` (free-form `(B, F, E)` vector), `camera` (per-frame intrinsics + extrinsics), `ego_camera`; add your own with `+model.modality_encoders.<name>=...` |
| `model/action_decoder` | `regression`, `mhp`, `anchor`, `diffusion_mlp`, `diffusion_dit`, `diffusion_unet`, `flow_mlp`, `flow_dit`, `flow_unet`, `anchor_diffusion_dit`, `anchor_flow_dit` |

Knobs that cut across the groups: `vision_encoder.token_mode=global|patch|fused`,
`action_decoder.action_space.kind=waypoint|velocity`, `vision_encoder.speed_head=true` for the
auxiliary speed loss, and a per-signal `normalizer` (`none|meanstd|minmax|scale`) on the
supervision targets and on each input encoder. All four are explained in the
[architecture guide](docs/architecture.md).

```bash
uv run visnavkit-train dataset=torch model=vint \
  model/vision_encoder=dinov3_s model/goal_encoder=gps model/action_decoder=flow_dit \
  model.vision_encoder.token_mode=fused
```

Generative decoders are a denoiser (`mlp`, `dit`, `unet`) times a scheduler (`ddim`, `flow`),
optionally anchored; the named entries are those combinations. Every decoder emits the same flat
`[mean, log_scale, confidence]` layout per mode, so metrics, export and the benchmark never
change.

### Recipes

Paper-named recipes are architecture adaptations to this repo's data contract (single RGB
frames, fixed-horizon x/y/v targets, goals sampled from the episode's own future) — not
reproductions or checkpoint-compatible replacements.

| Recipe | Vision | Temporal | Goal | Decoder |
| --- | --- | --- | --- | --- |
| `gnm` | MobileNetV2 | single frame | image (stacked with observation) | regression |
| `vint` | EfficientNet-B0 | causal x4, 512-d, 4 heads | image (stacked) | regression |
| `nomad` | EfficientNet-B0 | causal x4, 256-d | image, 50% goal dropout | diffusion U-Net, 10 steps, 8 candidates |
| `citywalker` | DINOv2 ViT-B (frozen) | causal x16, 768-d | point + past odometry | regression |
| `mbra` | EfficientNet-B0 | causal x4, 1024-d, 4 heads | gps | regression |
| `navdp` | DINOv2 ViT-S | causal x2, 384-d | point | diffusion DiT (384, x16), 10 steps, 16 candidates |
| `s2e` | EfficientNet-B0 | causal x4, 768-d | point, 50% goal dropout | anchor, 8 anchors |
| `socialnav` | FastViT-T8 | causal x1 | instruction (VLM prior) | flow DiT (1536, x12), 5 steps |
| `internvla_n1` | DINOv2 ViT-S | causal x1 | instruction (System 2 latent, 4 tokens) | flow DiT (384, x12), 10 steps |
| `mimic` | DINOv3 ViT-S | causal x4, 512-d | point + camera token | anchor, 64 anchors |
| `flowpilot` | FastViT-MA36 + speed head | causal x1, 1280-d | gps, 90% goal dropout | anchored flow DiT, 64 anchors, Beta(1.5, 1) times |

`model=base` is the bare skeleton these inherit — the stage wiring with no paper attached.
Anything that only reselects one group is an override, not a recipe:
`model/action_decoder=flow_dit`, `model/vision_encoder=dinov3_s`.

## Data

Each clip is a directory with `video.mp4` and four NumPy sidecars (times, positions,
orientations, speeds); manifests list `video_path label start end`. Targets are interpolated at
fixed relative times, so any frame rate works.

```bash
uv run visnavkit-dataset command=preprocess                    # validate clips, write manifests
uv run visnavkit-dataset command=visualize dataset=torch       # what the policy is actually fed
uv run visnavkit-dataset command=stats     dataset=torch name=city   # normalizer NPZ + plots
uv run visnavkit-dataset command=anchors   dataset=torch num_anchors=16
```

Sidecar layout, the opt-in ego and calibration inputs, and the public corpora this format
targets (FrodoBots-2K, EgoWalk, NVIDIA PhysicalAI AV, OpenScene, RECON, SCAND, GoStanford2,
SACSoN/HuRoN) with their licences:
[data guide](docs/data.md).

VisNavKit ships no trained policies — the recipes are architectures. What loads, and where each
paper publishes its own weights: [model catalog](docs/models.md).

## Train, export, benchmark

```bash
uv run visnavkit-train dataset=torch model=mimic
uv run visnavkit-export checkpoint=logs/baseline/.../last.ckpt output=outputs/policy.onnx
uv run visnavkit-benchmark command=export model=gnm output_dir=outputs/benchmark/gnm
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
uv run visnavkit-sanity-check model=nomad      # architecture, parameter counts, shapes
```

Layout: [`models/vision`](visnavkit/models/vision/) · [`models/temporal`](visnavkit/models/temporal/) ·
[`models/goal`](visnavkit/models/goal/) · [`models/modality`](visnavkit/models/modality/) ·
[`models/action`](visnavkit/models/action/) ·
[`models/policy.py`](visnavkit/models/policy.py) · [`data/`](visnavkit/data/) ·
[`benchmark/`](visnavkit/benchmark/) · [`evaluation/`](visnavkit/evaluation/) · [`scripts/`](visnavkit/scripts/)

Use it from another project with `uv add --editable /path/to/visnavkit`; the configs
ship inside the package as `visnavkit.configs`.

## Credits

VisNavKit is built on the following amazing open-source projects:

- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) Training loop, checkpointing and callbacks.
- [Hydra](https://github.com/facebookresearch/hydra) + [OmegaConf](https://github.com/omry/omegaconf) Composable configuration for every stage.
- [timm](https://github.com/huggingface/pytorch-image-models) Every vision backbone, pretrained and feature-ready.
- [TorchCodec](https://github.com/pytorch/torchcodec) and [NVIDIA DALI](https://github.com/NVIDIA/DALI) CPU and GPU video decoding.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [OnnxSlim](https://github.com/inisis/OnnxSlim) Deployment graphs, parity checks and benchmarks.
- [uv](https://github.com/astral-sh/uv) + [Ruff](https://github.com/astral-sh/ruff) Environments, linting and formatting.

The following repositories greatly inspire VisNavKit:

- [diffusers](https://github.com/huggingface/diffusers) — typed outputs, denoiser and scheduler as separate objects, the EMA schedule.
- [openpi](https://github.com/Physical-Intelligence/openpi) — flow-matching conventions and its Beta time sampling.
- [LeRobot](https://github.com/huggingface/lerobot) — one training forward, one deployment predict.
- [diffusion_policy](https://github.com/real-stanford/diffusion_policy) — the conditional 1D U-Net denoiser.
- [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) — the GNM / ViNT / NoMaD recipes.
- [openpilot](https://github.com/commaai/openpilot) — quadratically spaced trajectory anchors.

Thanks to the maintainers of these projects for their contribution to the community!

## Research references

Navigation policies, oldest first. Every one but ViKiNG has a recipe in the table above.

- **ViKiNG**: Vision-Based Kilometer-Scale Navigation with Geographic Hints (RSS 2022) — [arXiv:2202.11271](https://arxiv.org/abs/2202.11271)
- **GNM**: A General Navigation Model to Drive Any Robot (ICRA 2023) — [arXiv:2210.03370](https://arxiv.org/abs/2210.03370), [code](https://github.com/robodhruv/drive-any-robot)
- **ViNT**: A Foundation Model for Visual Navigation (CoRL 2023) — [arXiv:2306.14846](https://arxiv.org/abs/2306.14846), [code](https://github.com/robodhruv/visualnav-transformer)
- **NoMaD**: Goal Masked Diffusion Policies for Navigation and Exploration (ICRA 2024) — [arXiv:2310.07896](https://arxiv.org/abs/2310.07896), [code](https://github.com/robodhruv/visualnav-transformer)
- **CityWalker**: Learning Embodied Urban Navigation from Web-Scale Videos (CVPR 2025) — [arXiv:2411.17820](https://arxiv.org/abs/2411.17820), [code](https://github.com/ai4ce/CityWalker)
- **MBRA**: Learning to Drive Anywhere with Model-Based Reannotation (RA-L 2025) — [arXiv:2505.05592](https://arxiv.org/abs/2505.05592), [code](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA)
- **NavDP**: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance (ICRA 2026) — [arXiv:2505.08712](https://arxiv.org/abs/2505.08712), [code](https://github.com/InternRobotics/NavDP)
- **S2E**: From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning (ICLR 2026) — [arXiv:2507.22028](https://arxiv.org/abs/2507.22028), [code](https://github.com/VAIL-UCLA/S2E)
- **SocialNav**: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation (CVPR 2026 oral) — [arXiv:2511.21135](https://arxiv.org/abs/2511.21135), [code](https://github.com/AMAP-EAI/SocialNav)
- **InternVLA-N1**: Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-Language Navigation (ICLR 2026) — [arXiv:2512.08186](https://arxiv.org/abs/2512.08186), [code](https://github.com/InternRobotics/InternNav)
- **MIMIC**: Learning Sidewalk Autopilot from Multi-Scale Imitation with Corrective Behavior Expansion (ICRA 2026) — [arXiv:2603.22527](https://arxiv.org/abs/2603.22527), [code](https://github.com/VAIL-UCLA/MIMIC)
- **FlowPilot**: From Imitation to Alignment: Human-Preference Flow Policies for Long-Horizon Sidewalk Navigation (CoRL 2026) — [arXiv:2606.12603](https://arxiv.org/abs/2606.12603), [code](https://github.com/VAIL-UCLA/FlowPilot), [project](https://vail.cs.ucla.edu/FlowPilot)

World models, generative building blocks, simulators and benchmarks:

- **NWM**: Navigation World Models (CVPR 2025) — [arXiv:2412.03572](https://arxiv.org/abs/2412.03572), [code](https://github.com/facebookresearch/nwm)
- **Diffusion Policy** (RSS 2023, IJRR 2025) — [arXiv:2303.04137](https://arxiv.org/abs/2303.04137), [code](https://github.com/real-stanford/diffusion_policy); **DiT** (ICCV 2023) — [arXiv:2212.09748](https://arxiv.org/abs/2212.09748), [code](https://github.com/facebookresearch/DiT); **Flow matching** (ICLR 2023) — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **MetaUrban** (ICLR 2025) — [arXiv:2407.08725](https://arxiv.org/abs/2407.08725), [code](https://github.com/metadriverse/metaurban); **SidewalkBench** (CoRL 2026) — [arXiv:2606.16953](https://arxiv.org/abs/2606.16953)

## Citation

If VisNavKit helps your work, please consider citing it:

```bibtex
@Misc{visnavkit2026,
  author       = {Honglin He},
  title        = {{VisNavKit}: a composable toolkit for visual navigation policies},
  howpublished = {\url{https://github.com/DhlinV/visnavkit}},
  year         = {2026},
}
```

GitHub's *Cite this repository* button reads [`CITATION.cff`](CITATION.cff), which carries the
same entry. Please also cite the work a recipe adapts: [`CITATION.bib`](CITATION.bib) has one
per paper listed above, taken from Google Scholar's BibTeX export, never transcribed.
