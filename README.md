# navigators

Navigation imitation-learning models: hydra configs, Lightning training, DALI/torch dataloaders, ONNX export.

## Setup

```bash
uv sync
```

## Use as a library

```bash
uv pip install -e /path/to/navigators   # or: uv add --editable ../navigators
```

Configs ship inside the package (`navigators.configs`), so hydra composition works from an install too.

## Layout

```
navigators/
├── configs/       # hydra configs (dataset/, model/, optimizer/, metrics/, experiment/)
├── data/          # datasets + lightning datamodules (DALI and torch loaders)
├── models/        # E2EModel, ActionDecoder, LitModel
│   ├── encoders/     # vision encoders (timm backbone, DINOv2/v3)
│   ├── temporal_encoders/  # temporal fusion over the frame sequence
│   ├── heads/        # plan heads (MHP, waypoint, diffusion), pose_head
│   ├── layers/       # res blocks
│   └── losses/       # laplace NLL
├── evaluation/    # metrics + metric calculators (ADE/FDE)
├── scripts/       # entry points: train, export, smoke_forward, benchmark_dataloader
└── utils/
```

## Smoke test

```bash
uv run python -m navigators.scripts.smoke_forward
uv run pytest tests/
```

## Dataloaders

Two interchangeable datamodules over the same `path label start end` file_list format:
`dataset=dali` (GPU decode, NVIDIA DALI) and `dataset=torch` (CPU decode, torchcodec).

```bash
uv run python -m navigators.scripts.benchmark_dataloader --batches 50 common.data_root=/data/nav_clips
```

## Train / export

```bash
uv run python -m navigators.scripts.train experiment=<name>   # needs a dataset config first
uv run python -m navigators.scripts.export checkpoint=<ckpt> output=<out.onnx>
```

## Model zoo

Recipes in `configs/model/` compose a vision encoder, a temporal encoder, and a plan
head via hydra `_target_` overrides — select one with `model=<recipe>`. The reference-work
recipes are adaptations to this repo's goal-free, fixed-horizon data contract, not
reproductions.

| recipe | encoder | temporal_encoder | head |
|---|---|---|---|
| `base` | FastViT-T8 | causal transformer ×1 | MHP (Laplace NLL) |
| `gnm` | MobileNetV2 | none | waypoint regression |
| `vint` | EfficientNet-B0 | transformer ×4 | waypoint regression |
| `nomad` | EfficientNet-B0 | transformer ×4 | diffusion policy |
| `citywalker` | DINOv2 ViT-S (frozen) | transformer ×4 | waypoint regression |
| `s2e` | DINOv3 ViT-S (frozen) | causal transformer ×1 | MHP |
| `mimic` | = base | = base | = base |
| `dinov2` / `dinov3` | DINO ViT-S (frozen) | = base | = base |
| `diffusion` | = base | = base | diffusion policy |

## Reference works

Navigation foundation models and imitation learning:

- **GNM**: A General Navigation Model to Drive Any Robot (ICRA 2023) — [arXiv:2210.03370](https://arxiv.org/abs/2210.03370), [code](https://github.com/robodhruv/drive-any-robot)
- **ViNT**: A Foundation Model for Visual Navigation (CoRL 2023) — [arXiv:2306.14846](https://arxiv.org/abs/2306.14846), [code](https://github.com/robodhruv/visualnav-transformer)
- **NoMaD**: Goal Masked Diffusion Policies for Navigation and Exploration (ICRA 2024) — [arXiv:2310.07896](https://arxiv.org/abs/2310.07896), [code](https://github.com/robodhruv/visualnav-transformer)
- **ViKiNG**: Vision-Based Kilometer-Scale Navigation with Geographic Hints (RSS 2022) — [arXiv:2202.11271](https://arxiv.org/abs/2202.11271)
- **CityWalker**: Learning Embodied Urban Navigation from Web-Scale Videos (CVPR 2025) — [code](https://github.com/ai4ce/CityWalker)
- **S2E**: From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning — [arXiv:2507.22028](https://arxiv.org/abs/2507.22028), [project](https://vail-ucla.github.io/S2E/)
- **NWM**: Navigation World Models (CVPR 2025) — [arXiv:2412.03572](https://arxiv.org/abs/2412.03572)
- **MIMIC**: Learning Sidewalk Autopilot from Multi-Scale Imitation with Corrective Behavior Expansion (ICRA 2026) — [arXiv:2603.22527](https://arxiv.org/abs/2603.22527)

Simulation and benchmarks for urban micromobility:

- **MetaUrban**: An Embodied AI Simulation Platform for Urban Micromobility (ICLR 2025 Spotlight) — [code](https://github.com/metadriverse/metaurban)
- **SidewalkBench**: Benchmarking Visual Navigation on Urban Sidewalks — [arXiv:2606.16953](https://arxiv.org/abs/2606.16953)
