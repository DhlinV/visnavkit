---
name: oss-audit
description: Audit visnavkit for open-source readiness before publishing or after large changes. Use before tagging a release or making the repo public.
---

# Open-source audit

This repo is intended to be public. Run this before release and after big merges.

## Scrub checklist
```bash
# personal names, emails, private paths, internal repo names
grep -rniE "todo\([a-z]+\)|@(gmail|comma)|/home/|driving-model-track" \
  visnavkit tests README.md pyproject.toml --include="*.py" --include="*.yaml" --include="*.toml" --include="*.md"
```
- No person names in TODOs (plain `TODO:` only), no private repo mentions, no absolute home paths.
- `CLAUDE.md` / `.claude/skills/` may reference internal workflow — confirm with the owner whether they ship.
- No committed data, checkpoints, or onnx (`.gitignore` covers `*.ckpt`, `*.onnx`, data/, logs/, wandb/).
- `data_root` and file_list values in configs are placeholders, not real infra paths.

## Hygiene checklist
- LICENSE file exists and pyproject `license` field matches (blocker if missing).
- README commands actually run: copy-paste each one.
- `uv sync` from scratch works on a clean clone (lockfile committed and consistent).

## Full verification
```bash
uv run ruff check .
uv run pytest tests/ -q
uv run visnavkit-sanity-check --onnx
uv run visnavkit-export checkpoint=null output=/tmp/audit.onnx
```
All four must pass. Report findings as a list with file:line references; fix only after approval.
