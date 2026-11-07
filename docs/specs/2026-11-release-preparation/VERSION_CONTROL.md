# Git and release delivery — Bioacoustics DL Toolbox

This is the version-control contract for [specs 01–03](README.md). Initial GitHub spec series: [PR #2](https://github.com/darizae/bioacoustics-dl-toolbox/pull/2) on `spec/2026-11-release-preparation`. The existing published `v0.2.0` remains immutable. No new version/tag exists merely because the spec is committed.

## Bootstrap and sequential merges

1. Review and merge the **existing spec PR #2 into `main` first**. Check this contract is present and record its real merged-main SHA. Keep PR #2 and its branch intact until review; do not recreate them.
2. Implement each spec on a fresh focused branch from the **current reviewed `main`**, not from an unmerged previous task. Make a new implementation PR, run owning tests, review/approve, merge and note the new `main` SHA before moving to the next. Existing matching implementation branches take precedence over creating duplicates. Do not force-push or move historical tags.

| Spec | Proposed branch | Merge gate |
|---|---|---|
| [01 — NumPy spectrogram](01-headless-spectrogram.md) | `feat/toolbox-01-headless-spectrogram` | Merge public API and deterministic CPU tests, including optional-extra-free import, to `main`. |
| [02 — Spectral Grid README](02-readme-and-docs.md) | `docs/toolbox-02-human-readme` | Base on merged 01, verify genuine branding, GPL/NOTICE links and examples; merge before release. |
| [03 — tagged release](03-release.md) | `release/toolbox-v0.3.0` | Base on merged 02, set version `0.3.0` and truthful `RELEASE_NOTES.md`; review and merge release-prep PR **before** selecting tag SHA. |

No forcing an unneeded API: if 01 determines no extraction should ship, explicitly revise 03's version decision before creating a release PR/tag. A no-change assessment can be recorded in the owning PR.

## Release SHA and downstream handoff

- After the release-prep PR merges, freeze `TOOLBOX_RELEASE_SHA = main HEAD`. Verify `python -m pytest`, Ruff, source/wheel build, installed-wheel headless API import **without** optional visualization extras, license/ANIMAL-SPOT notice, and artifact SHA-256 **at that exact commit**.
- Create an immutable annotated tag **`v0.3.0`** at that `main` commit and publish matching GitHub Release wheel/sdist artifacts according to [releasing](../../RELEASING.md). The toolbox currently publishes on GitHub, **not automatically on PyPI**. An unmerged PR SHA or mutable `main` branch URL is not a released dependency.
- If anything changes after QA, merge a focused fix to `main`, **re-freeze and rerun** wheel/tests/hashes before tagging. No post-tag force updates. Record the release tag target SHA, exact wheel hashes, artifact download URLs and installed import examples. Handoff to [BioCall Studio 01](https://github.com/darizae/biocall-studio/blob/spec/2026-11-release-preparation/docs/specs/release-preparation-2026-11/01-upstream-ownership.md) only **after** publication.
- Toolbox 01–03 may run independently/in parallel with the *other repository's* alp-ui 01–03, but **not concurrently with each other**. Its own deterministic unit suite is independent of BioCall's one final full suite.

**Execution:** GitHub API (branch/PR/main synchronization), ChatGPT VM/CI (CPU base tests, build, checksum), release-capable Git/CI runner when annotated tags or uploads aren't available through current connector. No native installer or long-running real-data tests in this repo.
