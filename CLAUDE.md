# Golden rules

[DON'T BE ADHD, BE EXTREMELY CONCISE, CLEAR THINKING PATH, USE MINIMAL CODE]

- **Concise everywhere.** Replies lead with the answer or the exact command plus one short
  why. Docs and code comments say the contract and the non-obvious reason, nothing else.
- **Nothing is fixed.** The stage set is open by design: vision is the only input the policy
  knows by name, and anything else (ego state, calibration, a spatial/depth/LiDAR stage) is a
  `BaseModalityEncoder` plus a config entry. Leave that space; never hardcode a new input into
  `policy.py`.
- **Paper tricks are opt-in.** Model-specific auxiliary losses — the per-frame speed head is
  the current one — stay configurable per recipe, never baked into a base class, a loss dict
  or the export graph.
- **Focus ONLY on the asked task.** No scope creep, no unsolicited refactors, no drive-by
  "improvements".
- **MINIMAL CODE, always**: smallest viable diff; reuse proven components over new abstractions.
- **Review means review**: when asked for concerns/analysis, do NOT change code until
  explicitly told "do it" / "let's do this".
- **Clear thinking path**: state the diagnosis in 1-2 lines before the fix, so the reasoning
  is checkable.
- **Verify before claiming done**: smoke-test the forward/export paths or dry-run the actual
  command; report real shapes/losses.
- **Follow the source.** Recipes named after papers cite them and match the published code
  where it is public; where it is not, say so rather than inventing architecture.

See [AGENTS.md](AGENTS.md) for the same rules plus the commands and layout.

# Repo conventions

- `uv run` for everything; training records Git provenance, and `strict_git=true` requires a
  clean tree (untracked files count); after pushing, give the exact training/export command.
- Installable lib (`uv pip install -e .`); configs ship in the wheel via `visnavkit.configs`
  package-data, so keep all yamls under `visnavkit/configs/`.
