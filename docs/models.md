# Model catalog

Research identity is recorded separately from this repository's architecture adaptations.
**Downloadable ONNX artifacts are not yet validated open-loop policy adapters.**

## Published exports

The [UCLA-VAIL model zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public) holds
the published ONNX exports and wrappers. The catalog pins revision
`9c1f523ef8dddbccb6dee1d3d588112f754bc8f2` (verified 2026-09-07); sizes and SHA-256 digests are
in [`catalog.py`](../visnavkit/benchmark/catalog.py). The downloader takes only `.onnx` and
`.onnx.data`, verifies length and digest, and installs atomically; no remote Python is imported.

```python
from visnavkit.benchmark.catalog import download_model, get_model, list_models

for model in list_models():
    print(model.id, model.status, model.download_bytes)
paths = download_model("gnm", "artifacts/models")  # explicit network download
```

`register_model(ModelSpec(...))` adds an entry without validating inference.

| ID | Source | Published export | Local recipe |
| --- | --- | --- | --- |
| `gnm` | [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | image-goal graph; six RGB frames, 64×85 | image-goal adaptation, no temporal-distance head |
| `vint` | [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | image-goal graph; six RGB frames, 64×85 | image-goal adaptation, no temporal-distance head |
| `nomad` | [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | three component graphs; four RGB frames, 96×96 | 1D U-Net denoiser with goal dropout; sampler and normalization differ |
| `citywalker` | [ai4ce/CityWalker](https://github.com/ai4ce/CityWalker) | five RGB frames, 350×630, past coordinates, point goal | frozen DINOv2-B, 16-layer stack; past odometry as `past_xy`; causal, not bidirectional |
| `s2e` | [VAIL-UCLA/S2E](https://github.com/VAIL-UCLA/S2E) | BC Web100 variant; eleven RGB frames, 256×256, point goal | ICLR 2026 camera-ready: DINOv3, 6 layers, 64 k-means anchors; no RL stage |
| `mimic` | [VAIL-UCLA/MIMIC](https://github.com/VAIL-UCLA/MIMIC) | goal-free variant; sixteen RGB frames, 288×512; batch one | DINOv3-S, goal and camera tokens, 64-anchor decoder |
| `navdp` | [InternRobotics/NavDP](https://github.com/InternRobotics/NavDP) | none verified; checkpoints by author form | point-goal diffusion DiT; RGB only, no privileged critic |
| `mbra` | [MBRA / LogoNav](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA) | six RGB frames, 96×96, point-goal pose | LogoNav: EfficientNet-B0, local GPS goal, regression; heading omitted |

Resolution is `(height, width)`. A paper name in Hydra establishes no checkpoint compatibility;
published exports stay `artifacts_available` until reference parity is verified, and NavDP is
`unavailable` (downloading raises `ModelUnavailableError`).

## Wrapper audit

What each published bundle needs before its numbers mean anything:

- **GNM, ViNT**: the zoo wrappers feed random goal images; use a real goal image and label
  random-goal ablations. Cadence and metric spacing must match the checkpoint.
  [GNM](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/gnm/gnm.py),
  [ViNT](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/vint.py)
- **NoMaD**: the denoiser graph is one component; full-policy timing includes the encoder, the
  ten DDPM steps and decoding, with a fixed candidate count and seed.
  [metadata](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/NoMaD_GL_Official/model_info.yaml)
- **CityWalker**: the wrapper ignores `goal_xy`, randomizes the last coordinate row and can
  mutate the past-trajectory array; an adapter needs explicit odometry, goal, coordinate
  conversion and the original normalization.
  [wrapper](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/CityWalker_PG_Official/inference.py),
  [config](https://github.com/ai4ce/CityWalker/blob/main/config/citywalk_2000hr.yaml)
- **S2E**: BC weights, not the RL policy. Wrapper comments disagree with its code, which scales
  output XY by 0.25 and encodes (clipped distance / 200, cos, sin) — audit findings, not
  validated conversions.
  [wrapper](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/S2E/inference.py)
- **MIMIC**: the goal-free graph differs from the paper's DINOv3 + goal/camera model; the wrapper
  has a stale docstring and imports `urbansim`. Output times are
  `[1,2,4,6,7,8,10,12,14,15,17,19,21,23,25]/5` s — keep them and declare truncation; scaling is
  unvalidated.
  [metadata](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/MIMIC/model_info.yaml),
  [wrapper](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/MIMIC/inference.py)
- **NavDP**: RGB-D, diffusion sampling and critic selection must be preserved; upstream defaults
  are eight 224×224 frames plus depth. [agent](https://github.com/InternRobotics/NavDP/blob/master/baselines/navdp/policy_agent.py)

`output_names` are source-reported hints; only MIMIC has source-verified output timestamps.
`output_contract_verified=False` means the metre conversion is pending.

Keep RGB goal-free, image-goal, point-goal-with-odometry and RGB-D policies in separate cohorts,
report full-policy latency apart from component timing, and artifact bytes apart from parameter
count. The zoo is Apache-2.0 over the original terms: visualnav-transformer MIT, CityWalker
Apache-2.0, NavDP CC-BY-NC-SA-4.0
([zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public#license),
[NavDP](https://github.com/InternRobotics/NavDP#-license)).

## Weights

VisNavKit ships no trained policies. What loads:

| Weights | How |
| --- | --- |
| timm backbone (ImageNet, DINOv2/v3, CLIP, SigLIP, ...) | `model.vision_encoder.pretrained=true`, the default |
| a vision encoder you trained | `model.vision_encoder.weights=/path/encoder.pt` |
| a full VisNavKit checkpoint | `pretrained.ckpt_path=/path/last.ckpt`; `pretrained.strict=false` takes the stages that match |
| published ONNX exports | `visnavkit-benchmark command=download`, catalogued above |

Upstream checkpoints are **not** loadable into the recipes — the adapted architectures do not
line up tensor for tensor. Treat them as baselines and follow each project's licence.

| Recipe | Released weights | Notes |
| --- | --- | --- |
| `gnm`, `vint`, `nomad` | [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | one checkpoint per model |
| `citywalker` | [ai4ce/CityWalker](https://github.com/ai4ce/CityWalker) | — |
| `mbra` | [Learning-to-Drive-Anywhere-with-MBRA](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA) | LogoNav image-goal and GPS-goal |
| `navdp` | [InternRobotics/NavDP](https://github.com/InternRobotics/NavDP) | by author form |
| `s2e` | [VAIL-UCLA/S2E](https://github.com/VAIL-UCLA/S2E) | BC weights; the RL stage is unreleased |
| `socialnav` | [AMAP-EAI/SocialNav](https://github.com/AMAP-EAI/SocialNav) | Qwen2-VL and Qwen2.5-VL Brains; dataset unreleased |
| `internvla_n1` | [InternRobotics/InternVLA-N1](https://huggingface.co/InternRobotics/InternVLA-N1) | `-System2`, `-DualVLN`, `-Preview`, `-wo-dagger` |
| `mimic` | [VAIL-UCLA/MIMIC](https://github.com/VAIL-UCLA/MIMIC), [zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public) | inference and augmentation code, goal-free ONNX; training code and checkpoints planned |
| `flowpilot` | [VAIL-UCLA/FlowPilot](https://github.com/VAIL-UCLA/FlowPilot) | repository is still a placeholder |
