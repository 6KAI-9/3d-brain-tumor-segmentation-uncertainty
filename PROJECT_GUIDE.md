# 3D Medical Segmentation — Uncertainty & Explainability
**Project guide** · Deep learning course project (100 marks) + research paper

---

## 1. What this project is

- **Brief:** "Explainable AI & Uncertainty Estimation in 3D Medical Segmentation"
- **Task:** 3D tumor segmentation from MRI/CT
- **Uncertainty:** pixel-level quantification, evaluated via calibration error and entropy maps
- **Interpretability:** spatial-attention-based
- **Evaluation:** Dice score, calibration error, entropy maps
- **Constraints:** solo project, ~2–3 month (full-semester) timeline, dedicated GPU available
- **Deliverables:** codebase + research paper

---

## 2. Scope boundaries — read this before touching the frontend

Two tracks. Keep them structurally and temporally separate — don't let Track B eat Track A's time or dependency tree.

### Track A — Core (graded — most of the time budget goes here)
Data pipeline → segmentation model → uncertainty method → interpretability method → evaluation/ablations → paper. This is what a 100-mark rubric and a research paper are actually judged on. None of it needs a frontend.

### Track B — Showcase (optional, time-boxed)
An interactive demo / portfolio-grade frontend. Doesn't move the grade. Good for a viva demo and as a standalone portfolio piece — bad as a place to spend the week Track A needs. Start Track B only once Track A has: a trained baseline, a working uncertainty pass, and at least one interpretability visualization.

---

## 3. Repo verdicts — the links from planning

| Repo | What it actually is (verified) | Verdict |
|---|---|---|
| `usestrix/strix` | Autonomous AI pentesting agent (39k★, active) — finds exploitable vulnerabilities in *deployed, networked* apps via live attack simulation | Skip for the core repo — wrong threat model. Only relevant if Track B ships as a public web app, and even then `bandit` + Dependabot cover the realistic risk with far less setup |
| `Nutlope/hallmark` | "Anti-AI-slop" design skill for Claude Code/Cursor/Codex — structural and typographic rules against generic AI-generated UI | Useful for Track B only, if pursued. Small/new (37★) |
| `asgeirtj/system_prompts_leaks` | Crowd-sourced archive of extracted system prompts from various AI products (58.6k★) | Not related to this project — drop it |
| `bradautomates/claude-video` | Claude Code plugin: downloads a video, extracts frames + transcript, hands both to Claude (12.6k★, active) | Actually useful — this is how to get the Instagram reference clip in front of Claude Code, since a direct fetch of Instagram is blocked (see §5) |
| `Robbyant/lingbot-map` | Research repo: feed-forward 3D scene reconstruction from streaming video — robotics/outdoor-scene domain, arXiv paper (12.1k★) | Wrong domain. This is SLAM-style reconstruction of rooms/streets, not medical volume segmentation. Skip |
| `Shubhamsaboo/awesome-llm-apps` | Curated list of example LLM applications (RAG, agents, etc.) | Browsing material only, not a dependency |
| `openai/codex-plugin-cc` | Official OpenAI plugin — call Codex from inside Claude Code for review or delegation (`/codex:review`, `/codex:adversarial-review`) (26.7k★, active) | Genuinely useful: a second model reviewing Claude Code's own output is a real check against fluff. Needs a ChatGPT/OpenAI account |
| `addyosmani/agent-skills` | 24 production-engineering skills for coding agents: spec-first development, TDD, code review, security-and-hardening, doubt-driven-development, git workflow (80.5k★, active) | **The single best fit for "no fluff, no vulnerabilities, proper structure."** Install this one for certain |
| `ruucm/shadergradient` | React/three.js library for animated gradient backgrounds (~1.9k★, active, v2 in dev) | Fine for Track B |
| `paper-design/liquid-logo` | Small demo: turns a logo into a liquid-metal shader effect ("just for fun" per its own README) | Fine for Track B as a fun extra, not infrastructure |
| `dashersw/liquid-glass-js` | Vanilla WebGL "Apple Liquid Glass" effect library (~500★) | Fine for Track B, but niche/small — budget time for rough edges |
| `pmndrs/react-three-fiber` | The standard React renderer for three.js (31k★, mature, huge ecosystem: drei, xr, postprocessing) | Solid, safe choice if Track B happens |
| `nextlevelbuilder/ui-ux-pro-max-skill` | Design-intelligence skill/CLI: 84 styles, 192 palettes, font pairings, UX rules across tech stacks (110k★, active) | Good for Track B — keeps Claude Code from defaulting to generic layouts |

---

## 4. What that list is missing

None of the 13 links above are built for the actual scientific deliverable. These are:

| Need | Tool | Why |
|---|---|---|
| Segmentation backbone + training loop | **MONAI** (PyTorch) | Purpose-built for exactly this: 3D medical segmentation networks, NIfTI/DICOM transforms, Dice/Hausdorff metrics, MC-dropout uncertainty utilities, and `blend_images` for overlaying segmentation + entropy maps. Most of §1's evaluation criteria come out of the box |
| Uncertainty | MC dropout or a small deep ensemble, via MONAI | Directly matches "pixel-level uncertainty… entropy maps" |
| Interpretability | Attention U-Net (or an attention module bolted onto another backbone) + attention-map visualization; Captum as a secondary check | Matches "spatial-attention-based interpretability" as stated in the brief |
| Experiment tracking | MLflow or Weights & Biases | Reproducibility — needed for the paper's results tables |
| Scientific 3D viewer (if wanted) | **niivue** | WebGL2 medical volume viewer built for exactly this (NIfTI + segmentation overlays); used by research groups at Oxford, Boston Children's Hospital, etc. A better fit for showing actual segmentation results than a generic three.js scene |
| Fast interactive demo | Gradio or Streamlit → Hugging Face Spaces (free) | The standard, low-effort way segmentation/CV papers ship a "try it yourself" demo — hours, not weeks |
| Code review beyond Claude Code itself | `openai/codex-plugin-cc` adversarial review, or the `code-review-and-quality` skill from `agent-skills` | A second reviewer catches what one model misses |
| Dead-code / fluff detection | `vulture` (Python) | Directly targets "ghost code" |
| Lint / format / types | `ruff`, `mypy` | Fast, standard, zero-config-friendly |
| Dependency & security scanning | GitHub Dependabot (built in, free) + `bandit` | Matches the actual risk in a solo research repo — unlike an autonomous pentest agent built for live networked apps |

---

## 5. Frontend stack, if Track B happens

- **Core:** react-three-fiber + drei (helpers) + ShaderGradient for backgrounds
- **Motion/design:** Motion Primitives (Framer-Motion-based components), Haikei for background SVG blobs, realtimecolors.com for palette testing
- **Aesthetic guardrail:** `hallmark` and/or `ui-ux-pro-max-skill` as Claude Code skills, so output doesn't default to generic "AI slop" layouts
- **The part that actually shows your results:** niivue, not a generic 3D scene
- **Deploy:** Vercel (frontend) + Hugging Face Spaces or Cloud Run (inference backend — see §6)

**Instagram reference clip:** a direct fetch is blocked — Instagram disallows automated access, no way around that. Install `bradautomates/claude-video` in Claude Code and run `/watch <reel URL> what's the visual/motion language here?` — that gets the actual frames and motion cues into Claude Code's context, rather than a secondhand description.

---

## 6. Google AI Studio — what it can and can't do here

As of the mid-2026 "Build" mode expansion, AI Studio genuinely does generate a full-stack app (React + Node.js, with Firebase for auth/data) and deploy it to Cloud Run for free (Starter Tier, no card needed), with GitHub export. That part is real and current.

What it is *not* built for: serving a custom trained PyTorch/MONAI model. Build mode's backend is oriented around calling the Gemini API, not hosting arbitrary ML models. Two clean uses of it here:
- A Gemini-powered natural-language layer on top of the results ("explain this segmentation/uncertainty map in plain language") — a genuinely good, free fit
- Skip it for model-serving; that's a small FastAPI service around the MONAI model, hosted on Hugging Face Spaces (simplest) or Cloud Run directly

---

## 7. Workflow: Claude chat / Cowork / Claude Code / Obsidian / GitHub

- **Claude chat (here):** architecture decisions, this guide, the paper outline, "should I do X or Y" calls
- **Cowork:** literature review, drafting/organizing the paper, synthesizing experiment results into tables/figures, orchestrating multi-step non-coding work. It runs on the same agent architecture as Claude Code but is explicitly the non-developer tool — don't hand it the model code
- **Claude Code:** all the actual coding — model, training loop, tests, git commits/branches/pushes, running experiments, fixing CI
- **The bridge between Cowork and Code:** a shared project folder plus `CLAUDE.md`. They don't auto-sync with each other — that's the realistic picture, not a seamless built-in handoff. The shared folder and this guide are the connective tissue
- **Obsidian:** point a vault at (or symlink) the repo's `docs/` folder — it's just markdown, so it's version-controlled for free through the same git repo
- **GitHub:** source of truth. Claude Code pushes directly (git access is already set up). Add GitHub Actions running `ruff`, `mypy`, `pytest`, `vulture` on every push, plus Dependabot for dependency alerts — both free, both built in, no marketplace app required

---

## 8. Repository structure

```
project/
├── src/
│   ├── data/            # MONAI transforms, dataset loading
│   ├── models/           # segmentation architecture (attention U-Net etc.)
│   ├── uncertainty/       # MC dropout / ensemble inference
│   ├── explainability/    # attention map extraction + visualization
│   ├── training/          # training loop
│   └── evaluation/        # Dice, calibration error, entropy maps
├── configs/               # YAML/Hydra configs — no hardcoded hyperparameters
├── notebooks/             # exploration only, nothing load-bearing
├── tests/                 # pytest — one new test per new module, minimum
├── docs/                  # mirrors/symlinks into the Obsidian vault
├── demo/                  # Track B only — kept out of the core dependency tree
├── paper/                 # paper source
├── .github/workflows/     # CI: lint, type-check, tests, dead-code check
├── CLAUDE.md
└── README.md
```

---

## 9. Roadmap (rough — pin down the real deadline and re-map these)

Assuming ~12 working weeks:

1. **Week 1** — finalize dataset, finalize architecture + uncertainty method, repo scaffolding
2. **Weeks 2–4** — data pipeline + baseline segmentation model (Dice working end-to-end, no uncertainty/interpretability yet)
3. **Weeks 5–6** — add uncertainty quantification (MC dropout, calibration error, entropy maps)
4. **Weeks 6–7** — add spatial-attention interpretability + visualization
5. **Week 8** — full evaluation + ablations (with/without uncertainty, with/without attention)
6. **Weeks 9–10** — paper writing
7. **Weeks 10–11 (parallel, optional)** — Track B demo, only once 1–6 are solid
8. **Weeks 12+** — buffer, polish, viva prep

---

## 10. Code-quality guardrails

- `ruff check`, `mypy`, `pytest`, `vulture src/` before any task counts as done — Claude Code reports pass/fail only, not the full output, per standing preference
- No hardcoded hyperparameters — everything through `configs/`
- No new module without at least one test
- Run `/codex:adversarial-review` (from `codex-plugin-cc`) or the `code-review-and-quality` skill (from `agent-skills`) before merging anything non-trivial
- No new dependency without a one-line reason in the commit message

---

## 11. Open questions — confirm before Claude Code starts building

- Exact deadline (to convert §9's week numbers into real dates)
- Dataset: which one (BraTS, Medical Segmentation Decathlon, something course-provided)?
- Uncertainty method: MC dropout, deep ensemble, or evidential deep learning?
- Base architecture: Attention U-Net, or attention added to a different backbone?
- Is a live/deployed demo actually part of the grading, or purely a personal extra?
