# Response rules

[DON'T BE ADHD, BE EXTREMELY CONCISE, CLEAR THINKING PATH, USE MINIMAL CODE]

- Focus ONLY on the asked task. No scope creep, no unsolicited refactors, no drive-by "improvements".
- MINIMAL CODE, always: smallest viable diff; reuse proven components (Summarizer, existing heads) over new abstractions.
- Review means review: when asked for concerns/analysis, do NOT change code until explicitly told "do it" / "let's do this".
- Extremely concise replies: lead with the answer or the exact command; one short why.
- Clear thinking path: state the diagnosis in 1-2 lines before the fix, so the reasoning is checkable.
- Verify before claiming done: smoke-test the forward/export paths or dry-run the actual command; report real shapes/losses.
- Repo conventions: `uv run` for everything; `train.py` requires a clean git tree (untracked files count), so every new file must be committed; after pushing, give the exact training/export command.
- This repo is an installable lib (`uv pip install -e .`); configs ship inside the wheel via `navigators.configs` package-data, so keep all yamls under `navigators/configs/`.
- Proven components are ported from `~/projects/driving-model-track` (autopilot). Prefer porting from there over writing new ones.
