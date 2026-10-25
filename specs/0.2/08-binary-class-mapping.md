# SPEC-08 — Eliminate silent binary class-map collapse

**Status:** Planned · **Priority:** P1 · **Dependencies:** SPEC-03 · **Target:** v0.2.0

## Problem / evidence

In `SpectrogramDataset.__init__()` (`audio/datasets.py`), a two-class dataset with no `positive_class` or `class_to_index` uses the ANIMAL-SPOT default `"target" → 1; everything else → 0`. For `classes=["sparrow", "crow"]`, **both** names map to index 0, making binary classification incorrect without an error. Existing users with `target` must remain compatible.

## Required implementation

1. Validate that `classes` contains **distinct**, nonempty names, and that binary mode has exactly two. Validate explicit `class_to_index` has exactly those names as keys and bijective indices `{0,1}`; reject missing/extra keys and duplicate/out-of-range indices. Explicit mapping always wins.
2. If `positive_class` is supplied, require it exists in the two classes; map it to `1` and the other class to `0`. Do not silently ignore misspellings. If an explicit mapping is also supplied, either reject conflicting `positive_class` or document that explicit mapping takes precedence with conflict validation; recommended: reject a contradictory pair.
3. If neither setting exists and exactly one class is named `target`, preserve the legacy positive-class convention **regardless of class-list order**.
4. Otherwise create a **bijective class-list-order mapping** `{classes[0]:0, classes[1]:1}` and clearly log/document the deterministic fallback (no more silent class collapse). Do not issue a broad deprecation of legacy `target` mapping.
5. If a label extractor returns an unknown class, raise an actionable `ValueError` mentioning the label and declared classes; do not silently assign a default. Keep `class_dist_dict`, `get_class_name_from_index`, trainer summaries, and future checkpoint class order consistent.
6. For **multilabel**, the class order is governed by SPEC-03 and SPEC-04 and must not use binary `target` special-casing even with C=2.

## Unit tests only

File: `tests/test_class_mapping.py`. Instantiate datasets with synthetic filenames and monkeypatch the spectrogram reader if feature access is needed; no real audio files.

- `classes=["sparrow", "crow"]` maps to `{sparrow:0, crow:1}`, never `{0,0}`.
- `classes=["target", "noise"]` and reversed order preserve `{target:1, noise:0}` by default.
- Explicit `positive_class="sparrow"` maps it to 1; explicit mapping preserves caller choice.
- Reject nonexistent positive class, repeated declared class, duplicate indices, incomplete mapping, out-of-range index, and unknown extracted label.
- Multi-label C=2 is not affected by legacy `target` convention.

## Acceptance criteria

- [ ] Valid binary mappings are bijections over `{0,1}`.
- [ ] Legacy `target` preference remains correct.
- [ ] Non-target class pairs are no longer silently collapsed.
- [ ] Validation errors reveal configuration problems early.
- [ ] All added tests are **unit tests**.

## Documentation and release notes

**README surgical update after implementation:** Add one sentence under "Classification tasks" explaining that explicit `class_to_index` or `positive_class` is recommended when class polarity matters, while legacy `target` mappings are retained.

**Release-notes entry (copy only after completion)** → [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md), **Unreleased → Fixed**:

> **SPEC-08:** Fixed silent binary-label collapse for datasets without a `target` class; explicit or deterministic bijective class mappings now validate early.

Under **Migration notes**, mention that two-class datasets without a `target` or explicit mapping will use class-list order instead of mapping both labels to zero. Mark Implemented with tests and commit.
