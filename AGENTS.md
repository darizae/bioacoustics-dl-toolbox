# Repository instructions

- For work targeting v0.2.0, **first read [`0.2_VERSION.md`](0.2_VERSION.md)** and the relevant numbered spec in `specs/0.2/`. Document completed changes in [`RELEASE_NOTES.md`](RELEASE_NOTES.md); planned items are not shipped.
- Add **unit tests only** for v0.2.0 work. Do not add integration or end-to-end tests.
- Keep this package independent of Alpaca Desktop and `alpaca-pipelines` workflow,
  persistence, and UI contracts.
- Preserve the documented `trained_model.pt` checkpoint format unless a versioned
  replacement is introduced with a **unit-tested** loader compatibility path.
- Run `python -m pytest` and `python -m ruff check src tests` before pushing.
- This project is GPL-3.0-only because it contains adapted ANIMAL-SPOT code.

