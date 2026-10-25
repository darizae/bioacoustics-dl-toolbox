# SPEC-01 — Correct `CsvSplit` partitioning and CSV reuse

**Status:** Implemented (local working copy; commit pending) · **Priority:** P0 · **Dependencies:** None · **Target:** v0.2.0

## Problem / evidence

`src/bioacoustics_dl_toolbox/audio/datasets.py` — `CsvSplit.split_fn()` passes **independent** fractional indices to `numpy.split` instead of cumulative cut points. For 100 files, `{"train": .70, "val": .15, "test": .15}` gives cuts `[70, 15, 15]`: validation/test may be empty and samples omitted. `_split_with_seed()` uses `if self.seed:`, so seed `0` is treated as unset and `random.seed()` mutates process-global RNG state. `load()` also has fragile relative-path concatenation when reloading CSV files; protect CSV reuse while fixing splitting. `DatabaseCsvSplit` has a separate tape-aware algorithm and is not the primary target.

## Required implementation

1. Validate `split_fracs`: nonempty mapping, unique names, finite fractions in `[0,1]`, total approximately 1.0. Preserve insertion-order partition names. Reject duplicate input file identifiers explicitly or define/document their semantics; the recommended policy is to reject duplicates before sampling.
2. For `k` partitions and `N` items, compute **exactly `k-1`** cumulative integer boundaries: `cut[i] = floor(N * sum(fractions[:i+1]))`. Last partition receives the remainder; every input item appears **exactly once**. For common exact multiples, 100 at 70/15/15 yields 70/15/15.
3. Use a local `random.Random(seed)` instance for deterministic shuffling including `seed=0`, avoiding global RNG mutation. A `None` seed remains nondeterministic. Reusing the same split instance must not cause unintended re-shuffles after caching.
4. Preserve relative file paths and directory hierarchy when saving/loading CSVs, avoid double-prefixing a **relative** `working_dir`, and validate CSV rows. Don't alter the `CsvSplit.load(split, files=...)` public signature; existing valid CSVs should continue to load where possible.
5. If a requested positive-fraction split is necessarily empty for a tiny dataset, emit a clear `ValueError` on CSV generation (consistent with the current nonempty-CSV guard), not a misleading malformed partition. Zero-fraction splits may be empty by design.
6. Avoid changing `DatabaseCsvSplit`'s tape grouping; any shared helper refactor must preserve its group exclusivity. If local RNG is introduced, route it through the tape-aware path too so its seed remains reproducible without process-global RNG mutation.

## Unit tests only

File: `tests/test_csv_split.py`. Use arrays of in-memory identifiers and isolated `tmp_path` CSV fixtures, never recordings.

- `test_fractional_boundaries_allocate_every_item_once`: 100 unique items yield 70/15/15 and disjoint union of 100.
- `test_small_dataset_boundaries_do_not_drop_items`: small `N` distributes all items, or raises the clearly specified empty-positive-split error when CSV persistence requires it.
- `test_seed_zero_is_reproducible`: two instances at seed 0 yield same assignments.
- `test_global_random_state_unchanged`: splitting does not advance the global PRNG state.
- `test_invalid_fraction_inputs_raise`: NaN, negative, non-unit total, empty mapping, duplicate files.
- `test_relative_workdir_csv_round_trip`: save then load relative paths with nested directories using only `tmp_path` and fake filenames; no audio I/O.
- `test_tape_split_unchanged`: use a tiny synthetic set of ANIMAL-SPOT-pattern filenames to validate that one tape never spans multiple partitions; mock randomness if necessary. This is a **unit test of `DatabaseCsvSplit`**, not an end-to-end test.

## Acceptance criteria

- [x] Every input is assigned once and only once for a successful split; expected sizes for 100/70:15:15 are correct.
- [x] `seed=0` reproducible; no global RNG mutation.
- [x] CSV reload is path-correct and idempotent for valid input.
- [x] Tape-aware assignment still keeps tapes together.
- [x] All listed **unit tests** pass; no integration tests introduced.

## Documentation and release notes

**README surgical update after implementation:** Under Development, mention deterministic/cumulative CSV split behavior in one sentence only if it affects documented usage. Do not add a new tutorial.

**Release-notes entry (copy only after completion)** → [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md), **Unreleased → Fixed**:

> **SPEC-01:** Fixed `CsvSplit` boundaries so train/validation/test partitions cover every input exactly once, including deterministic handling of seed `0` and reliable CSV reuse.

Mark this spec Implemented with commit and test-file references only after verification.

## Implementation verification

- Unit tests: [`tests/test_csv_split.py`](../../tests/test_csv_split.py) (plus existing [`tests/test_checkpoints.py`](../../tests/test_checkpoints.py)).
- Local test command: `PYTHONPATH=/tmp/test_shims:src python -m pytest -q` — uses a temporary external, import-only `resampy` stub because dependencies cannot be installed in the offline environment. The stub is not part of the repository.
- `ruff` lint and format verification remains pending because `ruff` is unavailable offline.
- Commit hash: pending; the supplied ZIP archive does not contain Git metadata.
