# Open-loop visual navigation benchmark plan

Scope: reproducible comparison of model size, compute, ONNX Runtime inference latency,
and open-loop trajectory quality. Closed-loop simulation is outside this milestone.

## Execution schedule

Work proceeds in dependency order; these are delivery gates, not calendar estimates.

1. **Contract and source audit.** Verify official sources for GNM, ViNT, NoMaD,
   CityWalker, S2E, MIMIC (Honglin He, VAIL-UCLA), and NavDP. Record modality,
   conditioning, checkpoint provenance, and implementation status. Keep official
   model results separate from this repository's architecture adaptations.
2. **Correctness cleanup.** Correct timestamp-based trajectory interpolation,
   reject invalid pose data, index only windows with valid future coverage, align
   loader path handling, retain validation tails, and fix horizon-specific metrics.
   Preserve causal attention when selecting the last token for export.
3. **Benchmark core.** Add reusable ONNX export, deterministic sample inputs,
   strict execution-provider selection, warmup and timed runs, parameter and
   artifact sizes, FLOP accounting with declared coverage, machine metadata, and
   versioned JSON/CSV results. Exclude decoding/export/session creation from timed
   inference. Report mean, median, p90/p95/p99 latency and throughput with batch,
   resolution, context length, precision, sample count, and denoising steps.
4. **Open-loop evaluation.** Evaluate a fixed dataset split at the last observed
   frame using actual target timestamps. Report top-1 (only when a selector is
   defined), mean-sample, min-of-K ADE/FDE, speed error when available, and counts.
   Include stationary/constant-velocity predictors and a generated fixture.
   Store checkpoint/data/config fingerprints; label random-weight and synthetic
   results as pipeline checks rather than model-quality benchmarks.
5. **Model integration.** Support the requested models through a catalog with
   explicit official/adaptation status and input requirements. Integrate verified
   public implementations without silently substituting another architecture.
   Record unsupported exports or unavailable artifacts with actionable reasons.
   Begin runnable architecture comparisons with existing recipes, adding simple
   ResNet behavior cloning as an additional baseline. Do not invent trained results.
6. **Public workflow.** Add quickstart, model/protocol documentation, offline smoke
   data, regression tests and CI. Verify the actual CLI, exported-model numerical
   parity, and saved reports. Update this plan with completed and pending gates.

## Comparison rules

- Official checkpoints and controlled retraining are separate tracks. Existing
  paper-inspired recipes are adaptations, not reproductions.
- Goal-free, point-goal, image-goal, route-conditioned, and RGB-D inputs must be
  disclosed. Scores across incompatible tasks are not pooled into one ranking.
- Store total/trainable parameters separately from serialized ONNX size. Count
  multiply-add as two FLOPs; report unsupported operations rather than implying
  a partial estimate is a complete FLOP count.
- Distinguish full-context inference from cached-feature streaming. Every diffusion
  sample and denoising iteration contributing to a prediction belongs in timing.
- CPU timing includes ONNX Runtime input/output processing through `session.run`.
  A CUDA request must not silently become a CPU benchmark. State transfer scope.
- Use real nonzero inputs for parity. For stochastic models, expose initial noise
  or otherwise identify the limits of reproducibility and parity checks.
- No benchmark is a trained-model result without a supplied checkpoint and a real
  evaluation split. No missing metric or unsupported export is recorded as zero.

## Initial environment

The inspected machine has an NVIDIA RTX 5080 (16 GB), but the installed ONNX Runtime
exposes CPU and Azure providers only. Initial executable timing checks use CPU.
Real dataset/checkpoint locations and the preferred comparison track are pending
user input; work independent of those choices can proceed.

## Acceptance checks

- Analytic straight/turning trajectories, irregular frame times, final-anchor
  coverage, invalid quaternions, loader ranges, and partial validation batches.
- Hand-computed horizon-specific multimodal metrics and sample-weighted aggregation.
- Full-sequence versus cached-context parity, including multilayer transformers.
- ONNX numerical parity with fixed nonzero inputs; strict provider failure behavior.
- A generated-data command, an export/profile command, and open-loop evaluation
  produce parseable reports without credentials or pretrained-weight downloads.
- Model catalog documents each requested model's verified source and actual support.

## Delivery status — 2026-09-07

- Source audit and pinned artifact provenance are recorded in [models.md](models.md).
- Data correctness, native export/parity, provider checks, latency accounting,
  partial FLOPs, prepared shards, controls, and open-loop metrics are implemented.
- Native recipes and ResNet18 run. Published graphs support profiling; NoMaD
  includes its normalized-action sampler. Published metric adapters/reference
  parity and NavDP artifacts remain pending.
- [Commands and protocol](benchmark.md), README quickstart, CLI regression
  coverage, and CPU CI configuration are added. Hosted CI has not run here.

Local validation: **91 passed, 1 skipped** with
`uv run --no-sync pytest tests/ -q`. The skip is optional Diffusers scheduler
reference parity. Ruff, lockfile consistency, and whitespace checks pass.
The GNM CLI smoke had maximum trajectory parity error approximately `2.4e-6`.
Standalone fixture generation and the CLI constant-velocity control also passed.

Continuation fixes: published NoMaD forwards supplied encoder input archives;
native artifact reuse rejects ONNX hashes that differ from their metadata.
Regression tests cover both cases. Real trained-policy comparisons still require
a fixed evaluation split, matching checkpoints/modalities, and validated published
preprocessing/output contracts. Random-weight scores remain pipeline checks.

## Package migration validation — 2026-09-08

The package, Hydra targets, console commands, tests, and documentation now use
`visnavkit`. A source search found no remaining old package references.

- Regression suite: **91 passed, 1 skipped** (optional Diffusers reference parity).
- Ruff, offline lockfile consistency, and whitespace checks passed.
- The wheel built with all 19 YAML configs and four console entry points.
  Outside the checkout, training/export config composition, dataloader help,
  and benchmark fixture generation passed using the wheel contents.
- The installed `visnavkit-benchmark` smoke command exported GNM, profiled it,
  and evaluated two synthetic samples. Maximum trajectory parity error was
  `2.384185791015625e-6`; the report is
  `outputs/benchmark/visnavkit_smoke/result.json` (a pipeline check).
- Setup guidance now documents the DALI extra; training guidance reflects
  optional `strict_git=true` enforcement and the default CSV logger.

The trained-policy comparison prerequisites listed above remain pending.
