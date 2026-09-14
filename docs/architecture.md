# Architecture

A policy is four stages connected by tokens of width `feat_size`. Hydra instantiates
each stage from its own config group; `NavigationPolicy` connects them and `LitModel`
handles optimization and metrics. The decomposition follows
[diffusers](https://github.com/huggingface/diffusers) (typed outputs, denoiser and
scheduler as separate objects) and [LeRobot](https://github.com/huggingface/lerobot)
(one training forward, one deployment predict).

```text
visnavkit/models/
├── policy.py            NavigationPolicy: forward (training) / predict (deployment)
├── outputs.py           VisionOutput, PlanOutput, PolicyOutput dataclasses
├── lit_model.py         Lightning module: targets, losses, metrics, optimizer
├── vision/              frames -> tokens
│   ├── base.py          pair preparation, token modes, projection, speed head
│   ├── timm_cnn.py      any features_only timm backbone (ResNet, EfficientNet, ConvNeXt, FastViT, ...)
│   └── timm_vit.py      any timm ViT (DINOv2/v3, DeiT, EVA-02, SigLIP, CLIP, ...)
├── temporal/            tokens across frames -> context
│   ├── base.py          transformer over (frames x tokens), reductions
│   ├── causal.py / bidirectional.py / identity.py
├── goal/                goal specification -> goal tokens
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
| Vision encoder | `(N, 6, H, W)` previous/current RGB pair in [0, 1] | `VisionOutput(tokens (N, K, D), pose (N, 1), prev_img_mask (N,))` |
| Temporal encoder | `(B, F, K, D)` | `(B, F', K, D)`, `F' = F` for `reduction=none`, else 1 |
| Goal encoder | goal batch (see below) or `None` | `(N, G, D)`; `G = 0` for `none` |
| Action decoder | context `(N, K, D)`, goal `(N, G, D)`, optional noise | `PlanOutput(plans (N, M * (2 * T * P + 1)), ...)` |

- **Tokens per frame** `K`: `token_mode=global` (1), `patch` (`gh * gw` pooled patches),
  `fused` (1 + grid). Patch tokens carry a learned position embedding; the temporal
  encoder shares its frame position across the `K` tokens and expands the causal mask.
- **Pair fusion**: `pair_mode=early` feeds the 6-channel stack to the backbone;
  `late` runs a shared 3-channel backbone on both frames and concatenates embeddings
  (default for pretrained ViTs).
- **Goal batches**: point goals are per frame, `(B, F, 3)` as (distance, cos, sin) in
  each frame's ego frame; image `(B, 3, h, w)`, route image `(B, C, h, w)` and
  instruction `(B, E)` goals describe the whole window. `p_drop` replaces a sample's goal
  tokens with the learned null token during training; passing `goal=None` at inference
  uses the same null token (goal-free exploration, NoMaD style).
- **Conditioning**: the decoder concatenates context and goal tokens with a type
  embedding and pools them with one learned attention query (or the mean). One context
  token with no goal is passed through unchanged, so goal-free single-token recipes cost
  nothing extra. DiT denoisers also cross-attend to the raw tokens.
- **Flat layout**: every decoder packs `[mu, log_scale, confidence_logit]` per mode in
  pose space; `parse_plan_output` yields trajectories, scales, confidences and the best
  plan. Regression and generative decoders emit uniform confidences; anchor decoders
  emit classifier logits.

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
averages every floating-point tensor of the module, swaps the average in for validation,
and writes it into the checkpoint's `state_dict`, so monitored metrics, export and the
benchmark all see the averaged policy. The online weights ride along under
`ema_online_state_dict`, so a resumed run continues from the weights the restored
optimizer state belongs to. Denoising decoders benefit most; it is off by default.

## Deployment versus benchmark export

`NavigationPolicy.predict(frame, feature_buffer, goal=None, noise=None)` encodes one
frame and reuses buffered past tokens `(B, history, K * D)`; `scripts/export.py`
traces it with presence-driven inputs and verifies ONNX Runtime parity (enforced for
checkpoints, reported for untrained pipeline checks). `benchmark/export.py` traces the
full observation window with fixed shapes and outputs `trajectories`, `scores`, `speed`
for latency and open-loop measurements. Both flip `reduction=none` to `last`, precompute
ViT position embeddings for the export resolution, and disable the MHA fast path.

## Add a component

1. Subclass the stage's base (`BaseVisionEncoder._encode`, `BaseTemporalEncoder`,
   `BaseGoalEncoder.encode`, `BaseActionDecoder.decode/loss`, or a denoiser with the
   shared signature) in the matching package.
2. Add a yaml to the matching `configs/model/<group>/` directory with an explicit
   `_target_`; interpolate `feat_size` from `${model.feat_size}`.
3. Run `uv run visnavkit-sanity-check model/<group>=<name> --onnx`, then add the entry to
   the group test in `tests/models/test_model_configs.py`.
