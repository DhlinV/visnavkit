<h1 align="center">
  <img src="docs/assets/logo-options/d-arrow.png" alt="" width="64" align="absmiddle"> VisNavKit
</h1>

**Train, export, and benchmark visual navigation policies.**

VisNavKit combines vision encoders, temporal encoders, and policy heads through
Hydra configuration. Train on video and ego poses, export policies to ONNX, and
measure inference latency and open-loop trajectory error.

[Benchmark guide](docs/benchmark.md) · [Model catalog](docs/models.md) · [Configurations](visnavkit/configs/)

## Quick start

From a local checkout, install dependencies with `uv`:

```bash
uv sync
```

Run a small benchmark with generated data and random weights—no dataset or
pretrained weights required:

```bash
uv run visnavkit-benchmark command=smoke model=gnm \
  'common.crop_wh=[32,32]' common.downscale_factor=1 common.seq_length=2 \
  samples=2 runtime.warmup=1 runtime.iterations=2 output_dir=outputs/benchmark/smoke
```

This checks PyTorch/ONNX numerical parity, measures CPU inference, and scores
synthetic trajectories. Results are saved under `outputs/benchmark/smoke/`,
including `result.json`, `results.csv`, the ONNX model, and predictions.
**This is a pipeline check; measuring policy quality requires trained weights
and a real evaluation split.** See the [benchmark guide](docs/benchmark.md) for
dataset preparation, profiling, and evaluation.

## Train and export

Prepare `train.txt` and `val.txt` manifests with rows of `video_path label start end`,
plus the videos and pose arrays described in the
[data format](docs/benchmark.md#prepare-and-evaluate-real-data). Set
`common.data_root` to your dataset directory. The torch loader uses CPU video
decoding and requires FFmpeg; TorchCodec is included in the dependencies.

```bash
uv run visnavkit-train dataset=torch model=base experiment=baseline \
  common.data_root=/data/nav_clips
```

Select a model with `model=<recipe>`. Keep experiment-specific overrides in
[`visnavkit/configs/experiment/`](visnavkit/configs/experiment/) and select them
with `experiment=<name>`.

Export a trained checkpoint for deployment:

```bash
uv sync --extra export
uv run visnavkit-export checkpoint=/path/to/model.ckpt output=outputs/policy.onnx
```

Deployment export uses a feature buffer to reuse past frame features. For
full-window inference measurements, use the
[benchmark exporter](docs/benchmark.md#native-export-and-profiling).

<details>
<summary>Optional: GPU video decoding with DALI</summary>

On an NVIDIA GPU, install DALI and use `dataset=dali` when training. Both loaders
share the same manifests and target format.

```bash
uv sync --extra dali
uv run visnavkit-benchmark-data --batches 50 common.data_root=/data/nav_clips
```

To benchmark only the torch loader, add `--datasets torch` before `--batches`;
the DALI extra is then unnecessary.

</details>

## Model recipes

[Local recipes](visnavkit/configs/model/) share a goal-free, fixed-horizon data
format. Recipes named after research models are architecture adaptations, not
paper reproductions or compatible replacements for the original checkpoints.
The [model catalog](docs/models.md) tracks published ONNX bundles separately,
including validation still needed before comparing policy quality.

| Recipe | Vision encoder | Temporal encoder | Policy head |
| --- | --- | --- | --- |
| `base`, `mimic` | FastViT-T8 | 1-layer causal transformer | MHP |
| `resnet18` | ResNet18 | None | Waypoint regression |
| `gnm` | MobileNetV2 | None | Waypoint regression |
| `vint` | EfficientNet-B0 | 4-layer transformer | Waypoint regression |
| `nomad` | EfficientNet-B0 | 4-layer transformer | Diffusion |
| `citywalker` | Frozen DINOv2 ViT-S | 4-layer transformer | Waypoint regression |
| `dinov2` | Frozen DINOv2 ViT-S | 1-layer causal transformer | MHP |
| `dinov3`, `s2e` | Frozen DINOv3 ViT-S | 1-layer causal transformer | MHP |
| `diffusion` | FastViT-T8 | 1-layer causal transformer | Diffusion |

MHP is a multi-hypothesis prediction head trained with Laplace negative
log-likelihood.

## Development

```bash
uv run python -m visnavkit.scripts.smoke_forward
uv run pytest tests/
```

To use VisNavKit from another project, install it as an editable dependency:

```bash
uv add --editable /path/to/visnavkit
```

Hydra configs ship with the package as `visnavkit.configs`.

Source: [data loaders](visnavkit/data/) · [models](visnavkit/models/) ·
[benchmarks](visnavkit/benchmark/) · [metrics](visnavkit/evaluation/) ·
[CLI scripts](visnavkit/scripts/)

<details>
<summary>Research references</summary>

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

</details>
