# 02 — Human-oriented Spectral Grid README, accurate samples and light docs

## Authentic assets already present

Keep the dark/light-aware lockup in `assets/branding/png/lockup.png` and `lockup-inverse.png`, the original Spectral Grid mark in `assets/branding/png/mark.png`, and vector/icon provenance in `assets/branding/README.md`. Avoid AI-generated placeholder GUI screenshots: this is a **headless library**, not BioCall Studio's GUI. The existing `assets/branding/previews/lockups-light-and-dark.png` is a branding preview, **not** evidence of a completed spectrogram demo. If a tiny actual output example materially improves understanding, generate it via the real public function using deterministic test audio and label its synthetic origin; otherwise no new images are necessary.

## Rewrite priorities

Lead with who benefits, a concise “audio → spectral features → train → sliding-window detection” story and a short real working API example. Explain and visually distinguish the new *researcher-inspection spectrogram raster* from the existing model-training spectrogram input, since their numerical contracts are not interchangeable. Show a compact import and NumPy array/axes example once the release API exists; use only real callable signatures. Keep an optional `[rf]` install example, API scope links, `trained_model.pt` compatibility, and concise `NOTICE.md` attribution.

Use a small license shield (GPL-3.0-only) linked to the actual `LICENSE`, and a truthful Python/version/release status badge only if its target exists. Do not conflate toolbox GPL terms with MIT `alp-ui` or Studio's undecided redistributable license. Reuse relative asset links, visible lockup at appropriate width, readable code samples and descriptive alt text.

## Scope of markdown reduction

There are only a handful of documentation files (`README.md`, `AGENTS.md`, `NOTICE.md`, `RELEASE_NOTES.md`, `assets/branding/README.md`, `docs/RELEASING.md`). **Do not invent Markdown debt** or delete legally necessary attribution. Slim duplicate README/release process prose and avoid duplicated roadmap/history; retain short agent routing, public package API and science boundaries, and the current GPL/release contract. Historical published release notes remain factual; do not rewrite them to make the new version appear published.

## Execution and acceptance

**API/VM:** edit README/docs, verify local image links, command examples, badge links, package metadata and imported API, run scoped documentation checks. **Native:** none. Human-readable landing page, authentic brand, truthful examples and license, compact docs with no broken GitHub paths. Do **not** update release notes to call the new feature shipped before spec 03.

## Branch and integration handoff

Follow [VERSION_CONTROL.md](VERSION_CONTROL.md). Start `docs/toolbox-02-human-readme` from the **post-T1 merged `main`** so its example uses the actual public NumPy API, not anticipated signatures. Keep brand/license/attribution accurate, validate examples and links, open a documentation PR and merge to `main` **before** T3. Record actual merged-`main` SHA; do not publish a README-only version or quietly modify the already released `v0.2.0` tag.
