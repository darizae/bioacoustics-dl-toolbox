# Repository instructions

- Keep this package independent of Alpaca Desktop and `alpaca-pipelines` workflow,
  persistence, and UI contracts.
- Preserve the documented `trained_model.pt` checkpoint format unless a versioned
  replacement is introduced with a loader compatibility test.
- Run `python -m pytest` and `python -m ruff check src tests` before pushing.
- This project is GPL-3.0-only because it contains adapted ANIMAL-SPOT code.

