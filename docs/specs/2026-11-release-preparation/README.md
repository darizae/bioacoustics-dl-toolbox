# Bioacoustics DL Toolbox — coordinated November 2026 specification series

**Plans, not shipped changes.** Checked baseline: `main@22c1b342918b33c62288f23b660aa716f3e361e6`, current package and GitHub Release `v0.2.0`. BioCall Studio depends on `bioacoustics-dl-toolbox[rf]==0.2.0`; the new headless rendering API, when accepted, must be released before BioCall Studio integrates a pinned wheel.

Execute in order:
1. [01 — Extract headless spectrogram rendering](01-headless-spectrogram.md) — standalone NumPy render contract and public API, no Qt or Studio state.
2. [02 — Brand-forward README and documentation](02-readme-and-docs.md) — human-facing Spectral Grid assets, concise examples, accurate GPL badge.
3. [03 — Toolbox release](03-release.md) — verify base unit suite, build and publish a versioned distribution/tag before downstream integration.

**Proposed next tag: `v0.3.0`**, because a new public rendering API is additive functionality, not a patch fix. If the feature is rejected or not shipped, choose the correct smaller release separately, without claiming `v0.3.0` functionality.

**Cross-repository dependency:** [alp-ui specs](https://github.com/darizae/alp-ui/tree/main/docs/project/specs/2026-11-release-preparation) proceed independently; [BioCall Studio upstream integration spec](https://github.com/darizae/biocall-studio/blob/main/docs/specs/release-preparation-2026-11/01-upstream-ownership.md) is downstream of this toolbox's *released* `v0.3.0` (or the version actually published). One final BioCall Studio spec alone runs its complete own base suite against both latest approved released libraries.

**Execution:** GitHub API: review/code/docs/PR/release metadata; ChatGPT VM: deterministic CPU unit tests, Ruff, build, install-wheel smoke, checksums; native Mac/Windows: **not required** for this headless toolbox stage. No physical playback, new Cocoa captures, soak, huge real-data benchmark or performance qualification. Preserve GPL-3.0-only/ANIMAL-SPOT notices, established `trained_model.pt` compatibility, current ML feature transforms, and the existing published `v0.2.0` tag.

## Branch integration, versioning and synchronization

Use [VERSION_CONTROL.md](VERSION_CONTROL.md). **First merge the existing [toolbox spec PR #2](https://github.com/darizae/bioacoustics-dl-toolbox/pull/2) into `main` after review**. T1 is `feat/toolbox-01-headless-spectrogram` → tested PR → merge `main`; T2 is `docs/toolbox-02-human-readme` from post-T1 main → merge `main`; T3 is `release/toolbox-v0.3.0` → review/bump/notes PR → merge `main` → own tests/build on exact merged SHA → annotated tag `v0.3.0` → GitHub Release tested assets. No downstream Studio import until the version is actually published. Library tracks are parallel **only across repositories**, never across sequential T1–T3 steps.
