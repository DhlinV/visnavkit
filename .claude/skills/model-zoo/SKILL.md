---
name: model-zoo
description: Use or extend the reference-model recipes (GNM, ViNT, NoMaD, CityWalker, S2E, MIMIC, DINO encoders, diffusion head). Use when training a paper-style baseline or adding one.
---

# Model zoo

Every recipe in `visnavkit/configs/model/` composes four swappable groups — vision
encoder x temporal encoder x goal encoder x action decoder — purely via yaml
(`defaults: [base|<parent>, {override <group>: <choice>}, _self_]`). No recipe has its own class.

| recipe | vision | temporal | goal | action decoder |
|---|---|---|---|---|
| base | fastvit_t8 (pair-stacked) | causal x1 | none | mhp |
| mimic | dinov3_s | causal x4 (512-d) | point p_drop 0.5 + camera | anchor (64) |
| resnet18 | resnet18 | identity | none | regression |
| gnm | mobilenetv2 | identity | image (stack_observation) | regression |
| vint | efficientnet_b0 | causal_4layer (512-d) | image (stack_observation) | regression |
| nomad | efficientnet_b0 | causal_4layer (256-d) | image, p_drop 0.5 | diffusion_unet, 10 steps |
| citywalker | dinov2_b (frozen) | causal x16 (768-d) | point + ego past_xy | regression |
| s2e | dinov3_s (frozen) | causal x6 (768-d) | point, p_drop 0.55 | anchor (64, k-means) |
| dinov2 / dinov3 | DINO ViT-S (frozen) | causal x1 | none | mhp |
| diffusion / flow_dit / anchor | fastvit_t8 | causal x1 | none | diffusion_mlp / flow_dit / anchor |

All recipes are ADAPTATIONS to this repo's data contract (single RGB frames, fixed-horizon
x/y/v targets, goals sampled from the clip's own future) — describe them as "-style", never as
reproductions. Paper components that don't transfer (temporal-distance heads, topological
graphs, past-odometry inputs, RL post-training) are intentionally omitted.

## Run one
```bash
uv run visnavkit-train experiment=<name> dataset=<dali|torch> model=<recipe>
```
(or set the model inside the experiment file).

## Add one
1. New components only if yaml can't express it — see /new-encoder and /new-head.
2. `configs/model/<name>.yaml` inheriting the closest parent recipe (chained defaults work,
   e.g. nomad -> vint -> base) with `override <group>: <choice>` lines.
3. Head-comment cites the paper (arXiv id) and names what was adapted or omitted.
4. Add `<name>` to `tests/models/test_model_configs.py` (recipe table) and `test_smoke_forward.py`.
5. Add a row to this table and the README recipe table.
