---
name: export-deploy
description: Export a visnavkit checkpoint to ONNX and debug export failures. Use when producing a deployment model or when torch.onnx.export breaks.
---

# Export / deploy

```bash
uv run visnavkit-export checkpoint=<ckpt> output=<out.onnx>   # checkpoint=null: untrained pipeline check (parity reported, not enforced)
```

## What `scripts/export.py` does
1. Restores the checkpoint's model/preprocessing config (`prepare_export_config`), flips `temporal_encoder.reduction` `none -> last`.
2. `LitModel.load_from_checkpoint(..., weights_only=False)`, then `reparameterize_model`: `.reparameterize()` (FastViT) + Linear->BatchNorm1d folding; `vision_encoder.prepare_for_export((h, w))` precomputes ViT position embeddings.
3. Builds example inputs from `policy.example_inputs(...)`; input names are presence-driven: `input (1,6,h,w)`, `feature_buffer (1, seq_step*(seq_len-1), K*feat_size)`, then `goal` if the recipe has a goal encoder and `noise (1, M, T, A)` for generative decoders. Outputs: `plan, pose, feat_out, *heads`.
4. Traces `policy.predict` with the MHA fastpath disabled, slims (onnxslim, optional extra), converts to fp16 (`half=true`, io kept fp32), enforces output order, and runs ONNX Runtime parity on the same nonzero inputs.

## Debugging
- Unsupported aten op: usually a training-only branch; confirm `eval()` and the reduction flip. New denoisers must avoid data-dependent control flow (fixed `sample_steps`).
- Shape mismatch at parity: `feature_idxs` gathering in `NavigationPolicy.predict` must match the buffer layout (newest last, width `K * feat_size`).
- Goal/noise missing in the graph: check `policy.export_input_names()`; None inputs are dropped by design.
- fp16 NaNs: rerun with `half=false` to isolate.
- Paste the SANITY CHECK block (speed, logits, best_plan) in the PR/report.
