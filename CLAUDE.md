# CLAUDE.md — 3D Medical Segmentation (Uncertainty & Explainability)

## Project
3D tumor segmentation (MRI/CT) with pixel-level uncertainty quantification (entropy maps, calibration error) and spatial-attention-based interpretability. Deep learning course deliverable — 100-mark project + research paper. Solo, ~2–3 month timeline, dedicated GPU available. Full context: `PROJECT_GUIDE.md`.

## Scope
`src/`, `configs/`, `tests/`, and `paper/` are the graded deliverable — treat them with full rigor. `demo/` is a separate, optional showcase track: don't let it pull dependencies or time from the rest of the repo, and don't touch it unless explicitly asked to.

## Stack
- Python 3.11+, PyTorch, MONAI for the segmentation/uncertainty/metrics stack
- Config via YAML (or Hydra) — no hardcoded hyperparameters in source
- pytest for tests, ruff + mypy for lint/types, vulture for dead-code checks
- MLflow or W&B for experiment tracking

## Working rules
- Don't restate errors in the code unless explicitly asked to point them out
- Flag ambiguous requirements or missing decisions instead of silently assuming — ask, don't guess
- Every new module needs at least one test before it's "done"
- Run `ruff check .`, `mypy src/`, `pytest`, `vulture src/` before calling anything complete. Report pass/fail only — keep the full checklist output out of the response unless it's asked for
- No new dependency without a one-line reason in the commit message
- Prefer extending an existing module over creating a new one with overlapping responsibility
- Normal git workflow — commit and push as you go; don't batch unrelated changes into one commit

## Do not
- Do not touch `demo/` unless explicitly asked
- Do not add frontend/visualization dependencies to the core `src/` dependency tree
