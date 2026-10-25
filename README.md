# Bioacoustics DL Toolbox

Reusable PyTorch audio transforms, ResidualEncoder classifiers, metrics, training,
and checkpoint serialization for bioacoustic research. This package contains no
Alpaca Desktop or `alpaca-pipelines` workflow, persistence, UI, or run-management
code.

The repository was extracted with history from
[`alpaca-pipelines`](https://github.com/darizae/alpaca-pipelines) at source commit
`64f650645c373a4d1b9b47ea3ad48ec2d8bf3f16`. Much of the implementation is
adapted from [ANIMAL-SPOT](https://github.com/ChristianBergler/ANIMAL-SPOT), an
animal-independent ResNet-based bioacoustic classification framework. See
[`NOTICE.md`](NOTICE.md) for attribution and the research citation.

## Roadmap and release notes

- [**v0.2.0 implementation plan and agent onboarding**](0.2_VERSION.md) — implementation in progress; only completed workstreams are listed under Unreleased in the release notes.
- [**v0.2.0 workstream specifications**](specs/0.2/) — one spec per change, with unit-test acceptance criteria.
- [**Release notes**](RELEASE_NOTES.md) — ongoing change log, including explicitly marked planned work.

## Install locally

```bash
python -m pip install -e /path/to/bioacoustics-dl-toolbox
```

RF feature extraction is optional:

```bash
python -m pip install -e '/path/to/bioacoustics-dl-toolbox[rf]'
```

TensorBoard summaries and plotting are optional:

```bash
python -m pip install -e '/path/to/bioacoustics-dl-toolbox[summaries,visualization]'
```

## Checkpoint contract

`save_model` writes the established six-key `trained_model.pt` payload:

- `encoderConfig`
- `classifierConfig`
- `spectrogramConfig`
- `encoderState`
- `classifierState`
- `classes`

`load_model` uses CPU placement and PyTorch's weights-only loader. Consumers own
workflow provenance and any additional metadata such as sequence length,
normalization, and training metrics.

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```

`CsvSplit` now uses cumulative partition boundaries and local seeded randomness, including seed `0`, while preserving file paths when reusing split CSVs.

## License

GPL-3.0-only. See [`LICENSE`](LICENSE).

