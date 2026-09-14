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
| `model/vision_encoder` | `cnn` or `vit` with any timm backbone, plus presets: `fastvit_t8`, `fastvit_t12`, `resnet18`, `resnet50`, `efficientnet_b0`, `mobilenetv2`, `mobilenetv3`, `mobilenetv4`, `convnext_tiny`, `convnextv2_nano`, `regnety_008`, `repvit_m1`, `efficientvit_b0`, `dinov2_s`, `dinov2_b`, `dinov3_s`, `dinov3_b`, `vit_s`, `deit_s`, `eva02_s`, `siglip_b`, `clip_b` |
| `model/temporal_encoder` | `identity` (single frame), `causal`, `causal_4layer`, `bidirectional` |
| `model/goal_encoder` | `none`, `point` (distance, cos, sin), `gps` (local x/y metres), `image`, `route_image`, `instruction` (text embedding), `gps_route_image`; set it to a **list** for any other combination |
| `model/modality_encoder` | `none`, `ego` (free-form `(B, F, E)` vector), `camera` (per-frame intrinsics + extrinsics), `ego_camera`; add your own with `+model.modality_encoders.<name>=...` |
| `model/action_decoder` | `regression`, `mhp`, `anchor`, `diffusion_mlp`, `diffusion_dit`, `diffusion_unet`, `flow_mlp`, `flow_dit`, `flow_unet`, `anchor_diffusion_dit`, `anchor_flow_dit` |

Knobs that cut across the groups:

- `model.vision_encoder.token_mode=global|patch|fused` with `patch_grid=[4,4]` — one token
  per frame, or a pooled spatial grid for the temporal encoder and decoder to attend over.
- `model.action_decoder.action_space.kind=waypoint|velocity` (ego-frame poses, or unicycle
  speed and yaw rate integrated back to poses).
- Normalization is per signal, not global: supervision targets use
  `model.action_decoder.normalizer`, input signals their own encoder's `normalizer`
  (`gps`, ego vectors). Each is a `Normalizer` with `mode=none|meanstd|minmax|scale` —
  `scale` needs no corpus, the other two are fit or loaded from an NPZ.
- `model.vision_encoder.speed_head=true` — per-frame speed regression. A per-recipe
  auxiliary signal, off by default, adding a `speed` output to the exported graph.

```bash
uv run visnavkit-train dataset=torch model=vint \
  model/vision_encoder=dinov3_s model/goal_encoder=gps model/action_decoder=flow_dit \
  model.vision_encoder.token_mode=fused
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
| `gnm` | MobileNetV2 | single frame | image (stacked with observation) | regression |
| `vint` | EfficientNet-B0 | causal x4 | image (stacked) | regression |
| `nomad` | EfficientNet-B0 | causal x4 | image, 50% goal dropout | diffusion U-Net |
| `citywalker` | DINOv2 ViT-S | causal x4 | point | regression |
| `mbra` | EfficientNet-B0 | causal x4 | point | regression |
| `navdp` | DINOv2 ViT-S | causal x4 | point | diffusion DiT |
| `s2e` | DINOv3 ViT-S | causal x1 | point, 50% goal dropout | MHP |
| `socialnav` | FastViT-T8 | causal x1 | instruction (VLM prior) | flow DiT |
| `internvla_n1` | DINOv2 ViT-S | causal x1 | instruction (System 2 latent) | flow DiT (384, x12) |
| `mimic` | FastViT-T8 | causal x1 | none | MHP |
| `flowpilot` | FastViT-T8 + speed head | causal x1 | point | anchored flow DiT, Beta(1.5, 1) times |

`model=base` is the bare skeleton these inherit — the stage wiring with no paper attached.
Anything that only reselects one group is an override, not a recipe:
`model/action_decoder=flow_dit`, `model/vision_encoder=dinov3_s`.

## Data

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
goal encoder. Details: [benchmark guide](docs/benchmark.md#prepare-and-evaluate-real-data).

Opt-in modality inputs:

- `common.ego_features=[speed,yaw_rate]` packs those per-frame signals into `ego`; match
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

| Corpus | Content | Source |
| --- | --- | --- |
| FrodoBots-2K | ~2000 h teleoperated sidewalk driving in 10+ cities; RGB, GPS, IMU, audio, control | [BitRobot/FrodoBots-2K](https://huggingface.co/datasets/BitRobot/FrodoBots-2K) |
| RECON | Off-road exploration with goal images | [project](https://sites.google.com/view/recon-robot/dataset) |
| SCAND | Socially compliant human-teleoperated navigation | [project](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links) |
| GoStanford2 | Indoor trajectories, the ViNT-modified release | [download](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing) |
| SACSoN / HuRoN | Indoor navigation among people | [project](https://sites.google.com/view/sacson-review/huron-dataset) |

The last four are ViNT's public training set, listed in
[visualnav-transformer](https://github.com/robodhruv/visualnav-transformer). Tiny bundled
corpora live in [`assets/datasets/`](assets/) and are exercised by `tests/data/test_assets.py`.

## Pretrained weights

VisNavKit ships no trained policies; the recipes are architectures, not checkpoints. What it
does load:

| Weights | How |
| --- | --- |
| timm backbone (ImageNet, DINOv2/v3, CLIP, SigLIP, ...) | `model.vision_encoder.pretrained=true`, the default |
| A vision encoder you trained | `model.vision_encoder.weights=/path/encoder.pt` |
| A full VisNavKit checkpoint | `pretrained.ckpt_path=/path/last.ckpt`, with `pretrained.strict=false` to take the stages that match |
| Published navigation ONNX exports | `visnavkit-benchmark` downloads the pinned zoo — see [model catalog](docs/models.md) |

Upstream releases (GNM/ViNT/NoMaD, CityWalker, S2E, MBRA, SocialNav, InternVLA-N1) publish their
own checkpoints, linked in the references below. They are **not** loadable into these recipes:
the recipes adapt the architectures to this repo's data contract, so the tensors do not line up.
Treat them as references and as benchmark baselines, not as initialization.

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

## Citation

If VisNavKit helps your work, please consider citing it:

```bibtex
@Misc{visnavkit2026,
  author       = {DhlinV},
  title        = {{VisNavKit}: a composable toolkit for visual navigation policies},
  howpublished = {\url{https://github.com/DhlinV/visnavkit}},
  year         = {2026},
}
```

Please also cite the work a recipe adapts — [`CITATION.bib`](CITATION.bib) carries an entry for
every paper below, exported from arXiv rather than transcribed.

## Research references

Navigation policies, oldest first. Every one but ViKiNG has a recipe in the table above.

- **ViKiNG**: Vision-Based Kilometer-Scale Navigation with Geographic Hints (RSS 2022) — [arXiv:2202.11271](https://arxiv.org/abs/2202.11271)
- **GNM**: A General Navigation Model to Drive Any Robot (ICRA 2023) — [arXiv:2210.03370](https://arxiv.org/abs/2210.03370), [code](https://github.com/robodhruv/drive-any-robot)
- **ViNT**: A Foundation Model for Visual Navigation (CoRL 2023) — [arXiv:2306.14846](https://arxiv.org/abs/2306.14846), [code](https://github.com/robodhruv/visualnav-transformer)
- **NoMaD**: Goal Masked Diffusion Policies for Navigation and Exploration (ICRA 2024) — [arXiv:2310.07896](https://arxiv.org/abs/2310.07896), [code](https://github.com/robodhruv/visualnav-transformer)
- **CityWalker**: Learning Embodied Urban Navigation from Web-Scale Videos (CVPR 2025) — [arXiv:2411.17820](https://arxiv.org/abs/2411.17820), [code](https://github.com/ai4ce/CityWalker)
- **MBRA**: Learning to Drive Anywhere with Model-Based Reannotation (RA-L 2025) — [arXiv:2505.05592](https://arxiv.org/abs/2505.05592), [code](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA)
- **NavDP**: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance — [arXiv:2505.08712](https://arxiv.org/abs/2505.08712), [code](https://github.com/InternRobotics/NavDP)
- **S2E**: From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning (ICLR 2026) — [arXiv:2507.22028](https://arxiv.org/abs/2507.22028), [code](https://github.com/VAIL-UCLA/S2E)
- **SocialNav**: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation — [arXiv:2511.21135](https://arxiv.org/abs/2511.21135), [code](https://github.com/AMAP-EAI/SocialNav)
- **InternVLA-N1**: Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-and-Language Navigation — [arXiv:2512.08186](https://arxiv.org/abs/2512.08186), [code](https://github.com/InternRobotics/InternNav)
- **MIMIC**: Learning Sidewalk Autopilot from Multi-Scale Imitation with Corrective Behavior Expansion (ICRA 2026) — [arXiv:2603.22527](https://arxiv.org/abs/2603.22527), [code](https://github.com/VAIL-UCLA/MIMIC)
- **FlowPilot**: From Imitation to Alignment: Human-Preference Flow Policies for Long-Horizon Sidewalk Navigation (CoRL 2026) — [arXiv:2606.12603](https://arxiv.org/abs/2606.12603), [code](https://github.com/VAIL-UCLA/FlowPilot), [project](https://vail.cs.ucla.edu/FlowPilot)

World models, generative building blocks, simulators and benchmarks:

- **NWM**: Navigation World Models (CVPR 2025) — [arXiv:2412.03572](https://arxiv.org/abs/2412.03572), [code](https://github.com/facebookresearch/nwm)
- **Diffusion Policy** (RSS 2023) — [arXiv:2303.04137](https://arxiv.org/abs/2303.04137), [code](https://github.com/real-stanford/diffusion_policy); **DiT** (ICCV 2023) — [arXiv:2212.09748](https://arxiv.org/abs/2212.09748), [code](https://github.com/facebookresearch/DiT); **Flow matching** (ICLR 2023) — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **MetaUrban** (ICLR 2025) — [arXiv:2407.08725](https://arxiv.org/abs/2407.08725), [code](https://github.com/metadriverse/metaurban); **SidewalkBench** — [arXiv:2606.16953](https://arxiv.org/abs/2606.16953)
