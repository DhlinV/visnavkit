# Architecture

A policy takes vision, any number of extra input *modalities*, and any number of goals,
connected by tokens of width `feat_size`. Hydra instantiates each stage from its own config
group; `NavigationPolicy` owns the wiring, `LitModel` the optimization and metrics. The set
of stages is deliberately open: the policy knows vision by name and everything else through
the modality and goal contracts. The decomposition follows
[diffusers](https://github.com/huggingface/diffusers) (typed outputs, denoiser and
scheduler as separate objects) and [LeRobot](https://github.com/huggingface/lerobot)
(one training forward, one deployment predict).

```text
visnavkit/models/
├── policy.py            NavigationPolicy: forward (training) / predict (deployment)
├── outputs.py           VisionOutput, PlanOutput, PolicyOutput dataclasses
├── lit_model.py         Lightning module: targets, losses, metrics, optimizer
├── vision/              one RGB frame -> tokens
│   ├── base.py          frame normalization, token modes, projection, optional heads
│   ├── speed_head.py    optional auxiliary per-frame speed regression
│   ├── timm_cnn.py      any features_only timm backbone (ResNet, EfficientNet, ConvNeXt, FastViT, ...)
│   └── timm_vit.py      any timm ViT (DINOv2/v3, DeiT, EVA-02, SigLIP, CLIP, ...)
├── modality/            any non-image input -> per-frame tokens
│   ├── base.py          input_names contract, null token, modality dropout
│   ├── vector.py        free-form per-frame vector (ego state, IMU, odometry, ...)
│   ├── camera.py        pinhole intrinsics + extrinsics
│   └── none.py          empty slot
├── temporal/            tokens across frames -> context
│   ├── base.py          transformer over (frames x tokens), reductions
│   ├── causal.py / bidirectional.py / identity.py
├── goal/                goal specification(s) -> goal tokens
│   ├── base.py          null token, goal dropout
│   ├── none.py / point.py / image.py / route.py / instruction.py
└── action/              context (+ goal) tokens -> trajectories
    ├── base.py          conditioning, packing to the flat layout, losses
    ├── spaces.py        ActionSpace: waypoint | velocity
    ├── normalizer.py    ActionNormalizer buffers (none | meanstd | minmax)
    ├── anchors.py       AnchorSet: arc fan or NPZ vocabulary
    ├── regression.py / mhp.py / anchor.py
    ├── generative.py    GenerativeDecoder = denoiser x scheduler (x anchors)
    ├── denoisers/       mlp.py, dit.py, unet.py
    └── schedulers/      ddim.py, flow.py
```

## Tensor contracts

| Stage | Input | Output |
| --- | --- | --- |
| Vision encoder | `(N, 3, H, W)` RGB frame in [0, 1] | `VisionOutput(tokens (N, Kv, D), speed (N, 1) or None)` |
| Modality encoder | the batch keys in its `input_names`, each `(B, F, ...)`, or `None` | `(B, F, Km, D)`; `Km = 0` disables the slot |
| Temporal encoder | `(B, F, K, D)`, `K = Kv + sum(Km)` | `(B, F', K, D)`, `F' = F` for `reduction=none`, else 1 |
| Goal encoder(s) | one goal batch each (see below) or `None` | `(N, G, D)` concatenated; `G = 0` for `none` |
| Action decoder | context `(N, K, D)`, goal `(N, G, D)`, optional noise | `PlanOutput(plans (N, M * (2 * T * P + 1)), ...)` |

- **Tokens per frame** `K`: the vision encoder's `token_mode=global` (1), `patch`
  (`gh * gw` pooled patches) or `fused` (1 + grid), plus whatever the modalities add. Patch
  tokens carry a learned position embedding; the temporal encoder shares its frame position
  across the `K` tokens and expands the causal mask. `K * D` is also one deployment
  feature-buffer slot, so a new modality widens the buffer and the export graph by itself.
- **Modalities** are an open `{name: encoder}` mapping (`model/modality_encoder`, or
  `+model.modality_encoders.<name>=...`). An encoder's `input_names` are at once the
  dataset keys, the `forward`/`predict` keywords and the ONNX input names — the whole
  interface. Distinctness is enforced at construction. Ships with `VectorEncoder` (any
  free-form per-frame vector; `key` selects the batch entry, `common.ego_features` fills
  it) and `PinholeCameraEncoder`, which normalizes intrinsics by the resolution the policy
  is running at, so weights transfer across crops and downscales.
- **Optional inputs**: every non-vision encoder has a learned null token, so a missing
  goal, ego status or calibration still yields well-formed tokens. `p_drop` substitutes it
  for a fraction of training samples, so one set of weights works both ways (goal-free
  exploration, NoMaD style).
- **Goal batches**: point goals are per frame, `(B, F, 3)` as (distance, cos, sin) in each
  frame's ego frame; image `(B, 3, h, w)`, route image `(B, C, h, w)` and instruction
  `(B, E)` describe the whole window. `goal_encoder` may be a **list**, in which case
  `goal` and the dataset's `goal_type` are lists in the same order.
- **Auxiliary heads**: per-frame speed regression is a recipe-specific signal
  (`vision_encoder.speed_head=true`), not part of the contract. Without it
  `VisionOutput.speed` is None, there is no `vision_*` loss and no `speed` export output.
- **Conditioning**: the decoder concatenates context and goal tokens with a type embedding
  and pools them with one learned attention query (or the mean); a lone context token is
  passed through unchanged. DiT denoisers also cross-attend to the raw tokens.
- **Flat layout**: every decoder packs `[mu, log_scale, confidence_logit]` per mode in pose
  space; `parse_plan_output` yields trajectories, scales, confidences and the best plan.
  Regression and generative decoders emit uniform confidences, anchor decoders logits.

## Action spaces and normalization

`ActionSpace(kind, pose_size, plan grid)` converts dataset poses `(N, T, P)` to the
predicted quantity and back. `waypoint` is the identity. `velocity` derives unicycle
(speed, yaw rate) per anchor segment and integrates them back with the same grid, so
losses act on commands while metrics and export always see poses. `ActionNormalizer`
holds affine statistics as buffers (fit from targets or loaded from NPZ) so they travel
with the checkpoint and the ONNX graph. Generative decoders expect roughly unit-scale
targets; use `meanstd` or `minmax` once a corpus exists.

## Generative decoders

`GenerativeDecoder(denoiser, scheduler, anchors=None)`:

- **Schedulers** work on continuous time `t in [0, 1]` (1 = noise). `DDIMScheduler`
  trains with DDPM noise prediction and samples with deterministic DDIM. Its `clip_sample`
  clamp bounds the `1 / sqrt(alpha_bar)` term near `t = 1` and is expressed in *normalized*
  action units (diffusers clips at 1.0 for `[-1, 1]` data); with `normalizer.mode=none`
  those units are metres, so the clamp has to exceed the longest plan — `GenerativeDecoder`
  warns about that combination. `FlowMatchingScheduler` uses linear interpolation, a
  velocity target and Euler steps; `time_sampling` is `uniform`, `logit_normal` (SD3) or
  `beta` (openpi pi0's `Beta(1.5, 1)`, weighted toward the noisy end), and `shift` applies
  the diffusers `FlowMatchEulerDiscreteScheduler` time shift so more of the step budget
  lands at high noise.
- **Denoisers** share `forward(x_t, t, cond, tokens)`: `MLPDenoiser` (flat),
  `DiTDenoiser` (adaLN-Zero self-attention over anchor steps, cross-attention to context
  and goal tokens), `UNet1DDenoiser` (diffusion-policy FiLM U-Net).
- **Anchors** turn the problem into per-anchor residual generation: one denoised
  trajectory per anchor, confidences from an anchor classifier (`num_modes = K`).
- Training skips sampling (`plans` stay zero; the loss uses `cond`/`tokens`). Inference
  takes explicit `noise (N, M, T, A)`, so the same noise gives the same plan in PyTorch
  and ONNX.

## Weight averaging

`ema=default` adds `EMACallback` (`utils/ema.py`), the diffusers `EMAModel` /
diffusion-policy schedule `1 - (1 + step / inv_gamma) ** -power` capped at `decay`. It
averages every floating-point tensor, swaps the average in for validation, and writes it
into the checkpoint's `state_dict`, so metrics, export and the benchmark all see the
averaged policy; the online weights ride along under `ema_online_state_dict` so a resume
matches the restored optimizer state. Denoising decoders benefit most; off by default.

## Deployment versus benchmark export

`NavigationPolicy.predict(frame, feature_buffer, goal=None, noise=None, **modality_inputs)`
encodes one frame and reuses buffered past tokens `(B, history, K * D)`; the newest frame's
modality inputs are passed without a frame axis (`ego (B, E)`, `intrinsics (B, 3, 3)`, ...).
`scripts/export.py` traces it with presence-driven inputs — `vision`, `feature_buffer`, one
`goal` input per goal encoder, one per modality key, `noise` — and verifies ONNX Runtime parity
(enforced for checkpoints, reported for untrained pipeline checks). `benchmark/export.py`
traces the full observation window with fixed shapes and outputs `trajectories`, `scores`
(plus `speed` when the recipe has the auxiliary head) for latency and open-loop
measurements. Both flip `reduction=none` to `last`, precompute
ViT position embeddings for the export resolution, and disable the MHA fast path.

## Add a component

1. Subclass the stage's base (`BaseVisionEncoder._encode`, `BaseTemporalEncoder`,
   `BaseGoalEncoder.encode`, `BaseModalityEncoder.encode`, `BaseActionDecoder.decode/loss`,
   or a denoiser with the shared signature) in the matching package. A new input — a
   spatial raster, depth, LiDAR, an IMU window — is a `BaseModalityEncoder` with its own
   `input_names`; nothing in `policy.py` changes.
2. Add a yaml to the matching `configs/model/<group>/` directory with an explicit
   `_target_`; interpolate `feat_size` from `${model.feat_size}`. Modality files carry a
   `# @package model.modality_encoders` header and write one named slot, so options
   compose.
3. Run `uv run visnavkit-sanity-check model/<group>=<name> --onnx`, then add the entry to
   the group test in `tests/models/test_model_configs.py`.
