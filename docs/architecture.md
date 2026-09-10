# Model architecture

VisNavKit composes a per-frame spatial encoder, a temporal encoder, and an action
decoder. Hydra builds these components; `E2EModel` connects them and `LitModel`
handles optimization and metrics. The organization follows
[Diffusers' model families](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)
and [composition of reusable components](https://huggingface.co/docs/diffusers/main/en/using-diffusers/write_own_pipeline),
with explicit configuration choices as in
[LeRobot's policy construction](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/factory.py).
Hydra remains the factory, so no second registration framework is needed.

```text
visnavkit/models/
├── spatial_encoders/
│   └── vision_encoders/
│       ├── base.py               # RGB-pair preparation, projections, pose/loss contract
│       ├── timm.py               # General feature-pyramid backbone adapter
│       ├── vit_dino.py           # Shared DINO frame-pair encoding
│       ├── vit_dinov2.py
│       ├── vit_dinov3.py
│       ├── vit_fastvit.py
│       ├── cnn_resnet.py
│       ├── cnn_efficientnet.py
│       └── cnn_mobilenet.py
├── temporal_encoders/
│   ├── base.py                   # Shared transformer and time reductions
│   ├── causal.py
│   ├── bidirectional.py
│   └── identity.py
├── action_decoders/
│   ├── base.py                   # ActionDecoder: temporal encoder + trajectory head
│   ├── mhp.py                    # PlanHead: multi-hypothesis Laplace regression
│   ├── waypoint.py               # WaypointHead: deterministic regression
│   ├── diffusion.py              # DiffusionPlanHead: DDPM training / DDIM sampling
│   └── outputs.py                # Shared trajectory output parsing
├── heads/pose_head.py            # Auxiliary spatial supervision
├── layers/                      # Reusable neural-network layers
├── losses/                      # Loss primitives
├── compatibility.py             # Known checkpoint/config import migrations
├── e2e_model.py
└── lit_model.py
```

The tree omits package initializers and compatibility modules. FastViT is a
hybrid backbone; it uses the feature-pyramid adapter while keeping its own
`vit_fastvit` family entry. `spatial_encoders` leaves a place for additional
spatial modalities without mixing them into vision implementations.

## Choose components

| Hydra group | Choices |
| --- | --- |
| `vision_encoder` | `vit_fastvit`, `vit_dinov2`, `vit_dinov3`, `cnn_resnet`, `cnn_efficientnet`, `cnn_mobilenet`, `timm` |
| `temporal_encoder` | `causal`, `causal_4layer`, `bidirectional`, `identity` |
| `action_decoder` | `mhp`, `waypoint`, `diffusion` |

Each group has shared defaults in `base.yaml`. They populate the existing
`model.modules.vision_encoder` and `model.modules.action_decoder` structure.
The action decoder group selects the trajectory head; the temporal group
populates `model.modules.action_decoder.temporal_encoder`. Configs stay under
`visnavkit/configs/` and ship in the wheel.

```bash
# Existing recipe with independently selected temporal and action components.
uv run python -m visnavkit.scripts.smoke_forward model=vint \
  temporal_encoder=identity action_decoder=mhp \
  'common.crop_wh=[64,64]' common.downscale_factor=1 common.seq_length=3

# DINOv3 + full-window attention + diffusion, without pretrained downloads.
uv run python -m visnavkit.scripts.smoke_forward vision_encoder=vit_dinov3 \
  temporal_encoder=bidirectional action_decoder=diffusion \
  'common.crop_wh=[64,64]' common.downscale_factor=1 common.seq_length=3
```

Change a backbone variant through
`model.modules.vision_encoder.backbone_name=...`. Use its matching family, or
`vision_encoder=timm` for another timm feature-pyramid backbone, setting
`out_indices` and `act_layer` as required. Transformer depth, hidden sizes, and
other component settings remain ordinary Hydra overrides. Research recipe names
retain their original architecture adaptations; see the [model catalog](models.md).

## Tensor and training contracts

Print the selected model's pipeline and run its checks with:

```bash
uv run visnavkit-sanity-check --onnx vision_encoder=vit_dinov2 \
  temporal_encoder=bidirectional action_decoder=diffusion
```

The script draws the configured classes and tensor shapes as ASCII, checks
training forward/backward and feature-buffer equivalence, and optionally checks
ONNX Runtime outputs. Use ordinary Hydra overrides to select components or
change the small default input window. It never downloads pretrained weights.

- Vision inputs are floating-point previous/current RGB pairs: `[B, 6, H, W]`.
  Outputs include `feat_out: [B, D]`, `pose: [B, 1]`, and `prev_img_mask: [B]`.
  Pyramid backbones encode stacked pairs; DINO shares a three-channel backbone
  between the two frames and concatenates their pooled embeddings. DINO inputs
  are padded to patch boundaries and frozen backbones stay in evaluation mode.
- Temporal inputs are `[B, S, D]`. `reduction=none` returns all tokens; `last`,
  `avg`, and `sum` return `[B, D]`. Causal attention sees only preceding/current
  frames; bidirectional attention sees the whole supplied window; identity
  applies only the requested reduction.
- All trajectory heads return `plans` with flat per-mode layout
  `[mu, log_scale, confidence_logit]`, totaling `M * (2 * T * P + 1)` values.
  `parse_plan_output` provides trajectories, confidences, scales, and the best
  trajectory. Waypoint and diffusion retain this layout for common metrics and
  export. Diffusion uses noise prediction during training and sampling in eval.
- `E2EModel` accepts frame sequences `[B, S, 6, H, W]` and returns `vision` and
  `action` dictionaries. Causal/identity training defaults to one action per
  frame. Reduced outputs train against the final frame's future trajectory;
  auxiliary pose supervision still covers every frame.

Bidirectional defaults to `reduction=last`. Using `none` with per-frame online
action supervision exposes earlier decisions to later observations. Reserve
that combination for intentional offline sequence tasks. `avg` and `sum` also
use the final decision target when training through `LitModel`.

## Export and compatibility

Deployment accepts one current RGB pair plus a feature buffer. Full-window
benchmark export encodes the complete observed sequence. Both consume the same
component outputs; diffusion benchmark export takes explicit initial noise for
numerical parity. `none` becomes `last` for deployment.

DINOv2 export precomputes its position embeddings for the requested resolution
using the same antialiased interpolation as eager inference. The exporters use
the prepared encoder copy, preserving the original model and checkpoint shapes.
This avoids an unsupported runtime interpolation operation in ONNX.

Old encoder, temporal encoder, action decoder, and action-head import paths are
small reexports for saved Hydra configs. Module attribute names and state-dict
keys are preserved, including unwrapped single-layer temporal transformers.
Benchmark checkpoint matching recognizes known equivalent old/new targets but
still rejects architecture changes. New configs and imports use canonical paths.

## Add a family

1. Add an implementation to its component package and export its public class.
   Reuse `BaseVisionEncoder` for shared vision processing, `TimmVisionEncoder`
   for feature pyramids, or `BaseTemporalEncoder` for transformer variants.
2. Add a Hydra group config with the existing tensor contract and explicit
   `_target_`. Add a model recipe only when it represents a useful combination.
3. Verify forward, loss/backward, and export behavior. Preserve state-dict keys
   for existing families and test any intentional migration.

`tests/models/` covers component behavior, checkpoint compatibility, attention
direction, training targets, and feature-buffer parity. Benchmark tests exercise
ONNX export and numerical parity.
