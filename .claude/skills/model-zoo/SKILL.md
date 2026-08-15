---
name: model-zoo
description: Use or extend the reference-model recipes (GNM, ViNT, NoMaD, CityWalker, S2E, MIMIC, DINO encoders, diffusion head). Use when training a paper-style baseline or adding one.
---

# Model zoo

Every recipe in `navigators/configs/model/` composes three swappable parts — vision
encoder x temporal_encoder x plan head — purely via yaml (`defaults: [base|<parent>, _self_]`
plus `_target_` overrides). No recipe has its own model class.

| recipe | encoder | temporal_encoder | head |
|---|---|---|---|
| base | FastViT-T8 (pair-stacked) | causal transformer x1 | PlanHead (MHP, Laplace NLL) |
| gnm | MobileNetV2 | none (`num_layers: 0`) | WaypointHead |
| vint | EfficientNet-B0 | transformer x4 | WaypointHead |
| nomad | EfficientNet-B0 | transformer x4 | DiffusionPlanHead |
| citywalker | DINOv2 ViT-S (frozen) | transformer x4 | WaypointHead |
| s2e | DINOv3 ViT-S (frozen) | causal transformer x1 | PlanHead |
| mimic | = base | = base | = base |
| dinov2 / dinov3 | DINO ViT-S (frozen) | = base | = base |
| diffusion | = base | = base | DiffusionPlanHead |

All recipes are ADAPTATIONS to this repo's data contract (goal-free, fixed horizon,
prev+cur frame pair, x/y/v targets) — describe them as "-style", never as reproductions.
Paper components that don't transfer (goal images/coordinates, goal masking, topological
graphs, RL post-training) are intentionally omitted.

## Run one
```bash
uv run python -m navigators.scripts.train experiment=<name> dataset=<dali|torch> model=<recipe>
```
(or set the model inside the experiment file).

## Add one
1. New components only if yaml can't express it — see /new-encoder and /new-head.
2. `configs/model/<name>.yaml` inheriting the closest parent recipe (chained defaults work,
   e.g. nomad -> vint -> base).
3. Head-comment cites the paper (arXiv id) and names what was adapted or omitted.
4. Add `<name>` to the zoo parametrization in `tests/models/test_smoke_forward.py`.
5. Add a row to this table and the README zoo table.
