# Model catalog and integration status

The catalog records each model's research identity separately from this repository's
trainable architecture adaptations. **Downloadable ONNX artifacts are not yet
validated open-loop policy adapters.**

## Release provenance

The [UCLA-VAIL model zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public)
holds the published ONNX exports and their wrappers. The catalog pins revision
`9c1f523ef8dddbccb6dee1d3d588112f754bc8f2`, verified on 2026-09-07; per-file sizes
and SHA-256 digests are in [`catalog.py`](../visnavkit/benchmark/catalog.py), taken
from the [pinned manifest](https://huggingface.co/api/models/UCLA-VAIL/Navigation-Model-Zoo-Public/tree/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2?recursive=true&expand=false).

The downloader takes only `.onnx` and `.onnx.data`, streams to a temporary file, checks
length and SHA-256, and installs atomically, so an interrupted bundle can be retried.
Keep external data beside its graph. No remote Python is imported.

```python
from dataclasses import asdict
from visnavkit.benchmark.catalog import download_model, get_model, list_models

for model in list_models():
    print(model.id, model.status, model.download_bytes)

print(asdict(get_model("mimic")))
paths = download_model("gnm", "artifacts/models")  # explicit network download
```

`MODELS` is read-only; `register_model(ModelSpec(...))` adds an entry
(case-insensitive duplicate protection) without validating inference, so a custom
entry can describe your own model files. Built-in `Artifact` URLs point at the pinned
zoo release.

## Research identity and supported artifacts

| ID | Primary source | Published export | Local recipe status |
| --- | --- | --- | --- |
| `gnm` | [GNM / official code](https://github.com/robodhruv/visualnav-transformer) | Image-goal graph; six RGB frames, 64×85 | `gnm` is an image-goal architecture adaptation (no temporal-distance head) |
| `vint` | [ViNT / official code](https://github.com/robodhruv/visualnav-transformer) | Image-goal graph; six RGB frames, 64×85 | `vint` is an image-goal architecture adaptation (no temporal-distance head) |
| `nomad` | [NoMaD / official code](https://github.com/robodhruv/visualnav-transformer) | Three component graphs; four RGB frames, 96×96 | `nomad` uses a 1D U-Net denoiser with goal dropout; sampler and normalization differ |
| `citywalker` | [CityWalker](https://github.com/ai4ce/CityWalker) | Five RGB frames, 350×630, past coordinates and point-goal | `citywalker` follows the paper's frozen DINOv2-B and 16-layer stack; past odometry arrives as the `past_xy` ego feature; attention stays causal, not bidirectional |
| `s2e` | [S2E](https://github.com/VAIL-UCLA/S2E) | BC Web100 variant; eleven RGB frames, 256×256, point-goal | `s2e` follows the ICLR 2026 camera-ready: DINOv3 encoder, 6-layer stack, 64 k-means anchors; no RL stage |
| `mimic` | [VAIL-UCLA/MIMIC](https://github.com/VAIL-UCLA/MIMIC) | Goal-free variant; sixteen RGB frames, 288×512; fixed batch one | `mimic` follows the paper: DINOv3-S, goal and camera tokens, 64-anchor decoder; the published export is its goal-free variant |
| `navdp` | [NavDP](https://github.com/InternRobotics/NavDP) | No verified public ONNX bundle; checkpoint access via author form | `navdp` is a point-goal diffusion-DiT adaptation; RGB only (the depth branch belongs in a modality encoder) and no privileged critic |
| `mbra` | [MBRA / LogoNav](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA) | Six RGB frames, 96×96, point-goal pose | `mbra` is the LogoNav architecture (EfficientNet-B0 + local GPS goal + regression, 1024-d x4); its goal pose omits the paper's heading; the reannotation pipeline is a data stage |

Input resolution is `(height, width)`. A paper name in Hydra establishes no checkpoint
compatibility: size, FLOP and timing results carry `architecture_adaptation` provenance,
and published exports stay `published_export_variant` / `artifacts_available` until
reference parity and semantic adapters are verified. NavDP is
`external_integration_pending` / `unavailable`; downloading it raises `ModelUnavailableError`.

## Required contract checks

- **GNM and ViNT:** original implementations have separate image-goal encoders.
  The zoo wrappers supply random goal images and camera-specific preprocessing.
  Use a real goal image for image-goal evaluation, labeling random-goal ablations
  explicitly. Observation cadence and metric spacing must match the checkpoint
  and dataset. [GNM implementation](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/gnm/gnm.py),
  [ViNT implementation](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/vint.py)
- **NoMaD:** a denoiser graph is only one component of inference. Include visual
  encoding, the complete diffusion schedule and decoding in full-policy timing;
  label component timings separately. Fix candidate count and seed. The published
  wrapper uses ten DDPM steps and eight candidates.
  [Pinned metadata](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/NoMaD_GL_Official/model_info.yaml)
- **CityWalker:** the published wrapper ignores `goal_xy` and randomizes its last
  coordinate row; it can mutate the provided past-trajectory array. An adapter
  needs explicit past odometry and goal, coordinate conversion, and original
  normalization. Official training uses frozen DINOv2-B and sixteen attention
  layers. [Pinned wrapper](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/CityWalker_PG_Official/inference.py),
  [official configuration](https://github.com/ai4ce/CityWalker/blob/main/config/citywalk_2000hr.yaml)
- **S2E:** released weights are BC, not the RL-finetuned policy. Wrapper comments
  conflict with executed scaling and goal encoding. Code multiplies output XY
  by 0.25 and encodes clipped distance divided by 200, cosine of bearing, and
  sine of bearing. These are audit findings, not validated metric conversions.
  [Release scope](https://github.com/VAIL-UCLA/S2E),
  [pinned wrapper](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/S2E/inference.py)
- **MIMIC:** the full paper uses DINOv3, goal/camera tokens and horizon-specific
  anchor supervision; the published goal-free graph differs. Its wrapper has a
  stale docstring for another graph and imports `urbansim`. Output timestamps
  are `[1,2,4,6,7,8,10,12,14,15,17,19,21,23,25]/5` seconds. Preserve those times
  and declare truncation. Output scaling still requires reference validation.
  [Paper](https://arxiv.org/html/2603.22527v1),
  [pinned metadata](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/MIMIC/model_info.yaml),
  [pinned wrapper](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public/blob/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2/MIMIC/inference.py)
- **NavDP:** preserve RGB-D, diffusion sampling and critic selection. Upstream
  defaults are eight 224×224 RGB frames plus depth; these dimensions do not
  identify a validated export. Checkpoint access and export remain pending.
  [Repository](https://github.com/InternRobotics/NavDP),
  [agent input processing](https://github.com/InternRobotics/NavDP/blob/master/baselines/navdp/policy_agent.py)

`output_names` are source-reported hints, and NoMaD's is empty because its components
have separate interfaces. Only MIMIC has source-verified output timestamps.
`output_scale_to_meters=None` with `output_contract_verified=False` means the conversion
is pending: inspect the real tensor metadata and check nonzero-input parity before
claiming more.

## Fair comparisons and licensing

Keep RGB goal-free, RGB image-goal, RGB point-goal with odometry and RGB-D in separate
cohorts; a missing modality is never silently zero-filled. Report full-policy latency
apart from component timing, and artifact bytes apart from training parameter count —
shared diffusion parameters count once, denoiser FLOPs repeat for every executed step.

The zoo declares Apache-2.0 while preserving original model terms. GNM, ViNT and
NoMaD's source repository is MIT; CityWalker is Apache-2.0; NavDP's code is
CC-BY-NC-SA-4.0. Downloading does not relicense upstream code or weights.
[Zoo terms](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public#license),
[visualnav-transformer license](https://github.com/robodhruv/visualnav-transformer/blob/main/LICENSE),
[CityWalker license](https://github.com/ai4ce/CityWalker/blob/main/LICENSE),
[NavDP terms](https://github.com/InternRobotics/NavDP#-license)

## Pretrained weights

VisNavKit ships no trained policies; the recipes are architectures, not checkpoints. What it
does load:

| Weights | How |
| --- | --- |
| timm backbone (ImageNet, DINOv2/v3, CLIP, SigLIP, ...) | `model.vision_encoder.pretrained=true`, the default |
| A vision encoder you trained | `model.vision_encoder.weights=/path/encoder.pt` |
| A full VisNavKit checkpoint | `pretrained.ckpt_path=/path/last.ckpt`, with `pretrained.strict=false` to take the stages that match |
| Published navigation ONNX exports | `visnavkit-benchmark` downloads the pinned zoo, catalogued above |

### Upstream checkpoints

Where each paper publishes its own weights. They are **not** loadable into these recipes — the
recipes adapt the architectures to this repo's data contract, so the tensors do not line up.
Treat them as baselines to compare against, not as initialization, and follow each project's
licence.

| Recipe | Released weights | Variants |
| --- | --- | --- |
| `gnm`, `vint`, `nomad` | [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | one checkpoint per model |
| `citywalker` | [ai4ce/CityWalker](https://github.com/ai4ce/CityWalker) | — |
| `mbra` | [Learning-to-Drive-Anywhere-with-MBRA](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA) | LogoNav image-goal and GPS-goal |
| `navdp` | [InternRobotics/NavDP](https://github.com/InternRobotics/NavDP) | checkpoint access by author form |
| `s2e` | [VAIL-UCLA/S2E](https://github.com/VAIL-UCLA/S2E) | BC weights only; the RL stage is unreleased |
| `socialnav` | [AMAP-EAI/SocialNav](https://github.com/AMAP-EAI/SocialNav) | Qwen2-VL and Qwen2.5-VL Brains; the dataset stays unreleased |
| `internvla_n1` | [InternRobotics/InternVLA-N1](https://huggingface.co/InternRobotics/InternVLA-N1) | `-System2`, `-DualVLN`, `-Preview`, `-wo-dagger` |
| `mimic` | [VAIL-UCLA/MIMIC](https://github.com/VAIL-UCLA/MIMIC), [UCLA-VAIL zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public) | inference and augmentation code plus the goal-free ONNX export; training code and torch checkpoints planned |
| `flowpilot` | [VAIL-UCLA/FlowPilot](https://github.com/VAIL-UCLA/FlowPilot) | repository is still a placeholder |

`visnavkit-benchmark` downloads the zoo exports catalogued above; the caveats on each are in the contract checks.
