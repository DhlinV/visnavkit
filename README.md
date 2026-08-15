# navigators

Navigation IL models. Structure and proven components ported from `driving-model-track` (autopilot).

## Setup

```bash
uv sync
```

## Use as a library

```bash
uv pip install -e /path/to/navigators   # or: uv add --editable ../navigators
```

Configs ship inside the package (`navigators.configs`), so hydra composition works from an install too.

## Smoke test

```bash
uv run python -m navigators.smoke_forward
uv run pytest tests/
```

## Dataloaders

Two interchangeable datamodules over the same `path label start end` file_list format:
`dataset=dali` (GPU decode, NVIDIA DALI) and `dataset=torch` (CPU decode, torchcodec).

```bash
uv run python -m navigators.benchmark_dataloader --batches 50 common.data_root=/data/nav_clips
```

## Train / export

```bash
uv run navigators/train.py experiment=<name>   # needs a dataset config first
uv run python -m navigators.export checkpoint=<ckpt> output=<out.onnx>
```
