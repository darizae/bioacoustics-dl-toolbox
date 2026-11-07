# 03 — Verify and publish the new toolbox GitHub release

**Baseline:** `v0.2.0` exists as an annotated tag/GitHub Release. With the additive public headless NumPy spectrogram API in spec 01, propose `v0.3.0`, version `0.3.0` in `pyproject.toml`. If extraction was correctly rejected/no API shipped, explicitly decide a different version; do not tag absent features.

## Ordered steps

1. Merge/review specs 01 and 02 implementation. Inspect current `main` and real tag/release list, verify no version collision, and update `RELEASE_NOTES.md` with verified changes. Leave `Unreleased` honest.
2. Run toolbox **own** ordinary deterministic CPU `python -m pytest`, `python -m ruff check src tests`, `python -m ruff format --check src tests`. Add no soak/real-data benchmark/GUI audio test. Check no incompatible model preprocessing, `trained_model.pt` or RF contract changes.
3. Build sdist/wheel (`python -m build`), inspect contents, import the **installed** wheel in a clean environment, including `bioacoustics_dl_toolbox.visualization.spectrogram` without optional visualization extras. Check `project.version`, GPL-3.0-only `LICENSE` and adapted ANIMAL-SPOT `NOTICE.md`, and record SHA-256 of tested artifacts.
4. Release from an exact reviewed `main` commit with an immutable **annotated** `v0.3.0` tag; publish a GitHub Release containing the **same** wheel/sdist and concise release notes. Follow `docs/RELEASING.md`; do not push to PyPI without an explicit new policy. If GitHub API lacks annotated-tag/upload capability, use a Git-capable VM or permitted CI for that action; record genuine result, never simulate.
5. Verify public artifact download, tag-commit mapping, wheel SHA, public imports and actual release URL. Hand BioCall Studio its released version, commit SHA and artifact checksums so its adapter can depend on **released**, not editable/sibling-only, code.

## Gate

**BioCall Studio integration spec 01 does not complete before this toolbox release exists.** Documentation work in `alp-ui` may run in parallel; the **one final Studio suite** runs later in BioCall Studio after both libraries have released. No macOS/Windows native work or long-running test required here.

## Branch, main sync and immutable released wheel

Follow [VERSION_CONTROL.md](VERSION_CONTROL.md). Create `release/toolbox-v0.3.0` from the reviewed **post-T2 `main`**, update `pyproject.toml` version and `RELEASE_NOTES.md`, and **merge this release-preparation PR to `main` first**. Only then freeze `TOOLBOX_RELEASE_SHA = main HEAD`, run toolbox unit/Ruff/build/import tests on that exact SHA, and prepare the wheel/sdist digests. With owner approval, create the annotated **`v0.3.0`** tag at that same merged-`main` commit and publish those exact artifacts on GitHub. If a fix changes main SHA, rerun owning tests/build/hash before tagging. Handoff immutable tag target SHA and public release artifact hashes to BioCall Studio 01; do not run the full Studio suite here.
