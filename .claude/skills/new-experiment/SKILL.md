---
name: new-experiment
description: Create and run a training experiment in navigators. Use when trying a new hyperparameter, model tweak, or ablation.
---

# New experiment

Experiments are self-contained hydra files with the highest precedence. Recipe files
(model/, optimizer/, dataset/ yamls) are never edited for an experiment.

## Steps
1. Copy `navigators/configs/experiment/baseline.yaml` to `configs/experiment/<name>.yaml`.
2. Set `exp_name: <name>` and put ALL overrides in this one file, e.g.:
   ```yaml
   # @package _global_
   exp_name: lr_sweep_1e3
   optimizer:
     lr: 1e-3
   ```
3. Smoke it before training (config composes + forward runs):
   ```bash
   uv run python -m navigators.scripts.smoke_forward <name>
   ```
4. Commit the experiment file — `train.py` refuses to run on a dirty tree (untracked files count).
5. Train:
   ```bash
   uv run python -m navigators.scripts.train experiment=<name> dataset=<dali|torch>
   ```
   Checkpoints monitor `val/action_reg` (min); logs go to wandb project `navigators`.

## Afterwards
- Improvement over baseline: fold the winning overrides into `configs/train.yaml` (or the recipe file they belong to) and DELETE the experiment file.
- No improvement: keep the file in git as a record, or delete it — never leave stale half-adopted overrides.
