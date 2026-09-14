# Model catalog and integration status

The catalog identifies GNM, ViNT, NoMaD, CityWalker, S2E, MIMIC by Honglin He at
VAIL-UCLA, NavDP, and MBRA. It records research identity separately from this
repository's trainable architecture adaptations. **Downloadable ONNX artifacts
are not yet validated open-loop policy adapters.**

## Release provenance

The [UCLA-VAIL model zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public)
contains published ONNX exports and accompanying wrappers. The catalog pins
revision `9c1f523ef8dddbccb6dee1d3d588112f754bc8f2`, verified on 2026-09-07.
Per-file byte sizes and SHA-256 digests are in
[`catalog.py`](../visnavkit/benchmark/catalog.py). LFS digests come from the
[pinned public manifest](https://huggingface.co/api/models/UCLA-VAIL/Navigation-Model-Zoo-Public/tree/9c1f523ef8dddbccb6dee1d3d588112f754bc8f2?recursive=true&expand=false).
The 69,632-byte NoMaD distance-head external data file is an ordinary Git blob;
its SHA-256 was calculated directly from that revision's bytes.

The downloader retrieves only `.onnx` and `.onnx.data` artifacts. It streams into
temporary files, checks length and SHA-256, and atomically installs each completed
file. Interrupted bundles can be retried: already verified files are reused.
Keep external data beside its associated graph. No remote Python is imported.

```python
from dataclasses import asdict
from visnavkit.benchmark.catalog import download_model, get_model, list_models

for model in list_models():
    print(model.id, model.status, model.download_bytes)

print(asdict(get_model("mimic")))
paths = download_model("gnm", "artifacts/models")  # explicit network download
```

The `MODELS` mapping is read-only. `register_model(ModelSpec(...))` adds metadata
with case-insensitive duplicate protection. Registration does not validate
inference. Custom entries without artifacts can describe user-provided model
files. Built-in `Artifact` URLs refer to the pinned zoo release.

## Research identity and supported artifacts

| ID | Primary source | Published export | Local recipe status |
| --- | --- | --- | --- |
| `gnm` | [GNM / official code](https://github.com/robodhruv/visualnav-transformer) | Image-goal graph; six RGB frames, 64×85 | `gnm` is an image-goal architecture adaptation (no temporal-distance head) |
| `vint` | [ViNT / official code](https://github.com/robodhruv/visualnav-transformer) | Image-goal graph; six RGB frames, 64×85 | `vint` is an image-goal architecture adaptation (no temporal-distance head) |
| `nomad` | [NoMaD / official code](https://github.com/robodhruv/visualnav-transformer) | Three component graphs; four RGB frames, 96×96 | `nomad` uses a 1D U-Net denoiser with goal dropout; sampler and normalization differ |
| `citywalker` | [CityWalker](https://github.com/ai4ce/CityWalker) | Five RGB frames, 350×630, past coordinates and point-goal | `citywalker` uses point-goal conditioning; omits past-odometry input |
| `s2e` | [S2E](https://github.com/VAIL-UCLA/S2E) | BC Web100 variant; eleven RGB frames, 256×256, point-goal | `s2e` is a DINOv3 + point-goal + MHP adaptation; no RL stage |
| `mimic` | [MIMIC, Honglin He et al.](https://vail-ucla.github.io/MIMIC/) | Goal-free variant; sixteen RGB frames, 288×512; fixed batch one | `mimic` equals `base`; `model/action_decoder=anchor` adds an anchor decoder |
| `navdp` | [NavDP](https://github.com/InternRobotics/NavDP) | No verified public ONNX bundle; checkpoint access via author form | No local substitution |
| `mbra` | [Model-Based Reannotation](https://model-base-reannotation.github.io/) | Six RGB frames, 96×96, point-goal pose | No local substitution |

Input resolution is `(height, width)`. Native size/FLOP/timing results must carry
`architecture_adaptation` provenance; a paper name in Hydra does not establish
checkpoint compatibility. Published exports have `published_export_variant`
provenance and `artifacts_available` status until reference parity and semantic
adapters are verified. NavDP is `external_integration_pending` / `unavailable`;
attempting to download it raises `ModelUnavailableError` with the reason.

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

`output_names` are source-reported hints. NoMaD components have separate
interfaces, so its aggregate output-name tuple is empty. Only MIMIC currently has
source-verified output timestamps. `output_scale_to_meters=None` and
`output_contract_verified=False` flag conversions as pending. Inspect actual
tensor metadata and verify nonzero-input numerical parity before assigning a
stronger status.

## Fair comparisons and licensing

Keep RGB goal-free, RGB image-goal, RGB point-goal with odometry, and RGB-D in
explicit cohorts. Missing modalities cannot be replaced silently with zeros.
Separate full-policy latency from component timing, and artifact bytes/exported
initializer elements from training parameter count. Shared diffusion parameters
count once, while denoiser FLOPs repeat for every executed step. Unsupported
operations remain visible in compute reports.

The zoo declares Apache-2.0 while preserving original model terms. GNM, ViNT and
NoMaD's source repository is MIT; CityWalker is Apache-2.0; NavDP's code is
CC-BY-NC-SA-4.0. Downloading does not relicense upstream code or weights.
[Zoo terms](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public#license),
[visualnav-transformer license](https://github.com/robodhruv/visualnav-transformer/blob/main/LICENSE),
[CityWalker license](https://github.com/ai4ce/CityWalker/blob/main/LICENSE),
[NavDP terms](https://github.com/InternRobotics/NavDP#-license)
