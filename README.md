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
├── models/        # E2EModel, VisionEncoder, ActionDecoder, LitModel
│   ├── heads/     # plan/pose heads
│   ├── layers/    # summarizer, res blocks
│   └── losses/    # laplace NLL
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
