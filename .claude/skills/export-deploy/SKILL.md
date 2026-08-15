---
name: export-deploy
description: Export a navigators checkpoint to ONNX and debug export failures. Use when producing a deployment model or when torch.onnx.export breaks.
---

# Export / deploy

```bash
uv run python -m navigators.scripts.export checkpoint=<ckpt> output=<out.onnx>   # checkpoint=null: untrained pipeline check
```

## What the export path does (scripts/export.py)
1. Flips `temporal_encoder.reduction` `none -> last` (training predicts per token, deploy wants the last one only).
2. Disables the MHA fastpath (`torch.backends.mha.set_fastpath_enabled(False)`) — the fused aten op is not exportable.
3. `reparameterize_model`: calls `.reparameterize()` on any module that has it (fastvit), then folds Linear->BatchNorm1d pairs.
4. Export signature: `model(x, fb)` with `x (1, 6, h, w)` and `fb (1, seq_step*seq_len - seq_step, feat_size)`; outputs ordered `plan, pose, feat_out, *head_outputs` — `get_export_output_names()` must match the forward tuple exactly.
5. onnxslim -> optional fp16 (`half=true`, io kept fp32) -> `enforce_output_order` (index-based runtimes rely on it) -> ONNX Runtime sanity check printing speed/logits/best_plan.

## Adding an exported head
- Give the head an `output_names` attribute and export-mode forward.
- List it in `export_heads` in `configs/export.yaml`; heads missing from the model are silently skipped.
- The output count check in export.py will fail loudly if names and tensors diverge.

## Debugging export failures
- Unsupported aten op: usually a training-only branch still active — check `reduction` flip happened and model is in `eval()`.
- Shape mismatch at ORT sanity check: `feature_idxs` gathering in `E2EModel` must match the deploy feature-buffer layout (newest last).
- fp16 NaNs: re-run with `half=false` to isolate; keep io types fp32 (already default).
- Always finish with the SANITY CHECK block printing plausible values; paste it in the PR/report.
