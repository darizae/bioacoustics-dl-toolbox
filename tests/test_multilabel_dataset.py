"""Isolated multi-hot dataset contracts for SPEC-04 (no audio files required)."""

import pytest
import torch

from bioacoustics_dl_toolbox.audio import datasets
from bioacoustics_dl_toolbox.config import TaskConfig


@pytest.fixture
def make_dataset(monkeypatch):
    monkeypatch.setattr(datasets, "AsyncFileReader", lambda: object())

    def build(*, labels, classes=None, class_to_index=None, task_config=None):
        return datasets.SpectrogramDataset(
            ["bird-first.wav", "frog-second.wav", "rain-third.wav"],
            classes=classes if classes is not None else ["bird", "frog", "rain"],
            labels=labels,
            class_to_index=class_to_index,
            task_config=task_config or TaskConfig(task_type="multilabel"),
        )

    return build


def test_multilabel_vectors_are_float32_mapping_order_and_deduplicated(make_dataset):
    data = make_dataset(labels=[["bird", "rain"], [], ["frog", "frog"]])
    expected = [[1, 0, 1], [0, 0, 0], [0, 1, 0]]
    assert data.class_dist_dict == {"bird": 0, "frog": 1, "rain": 2}
    for index, values in enumerate(expected):
        label = data._load_label(data.file_names[index], index)
        assert label["call"].dtype == torch.float32
        assert label["call"].tolist() == values
        assert label["file_name"] == data.file_names[index]


def test_explicit_mapping_controls_multilabel_columns(make_dataset):
    data = make_dataset(
        labels=[["bird", "rain"], [], ["frog"]],
        class_to_index={"bird": 2, "frog": 0, "rain": 1},
    )
    assert data._load_label("first.wav", 0)["call"].tolist() == [0, 1, 1]
    assert data._load_label("third.wav", 2)["call"].tolist() == [1, 0, 0]


def test_getitem_returns_multi_hot_with_mocked_spectrogram(make_dataset):
    data = make_dataset(labels=[["bird", "rain"], [], ["frog"]])
    data.t_spectrogram = lambda path: torch.ones((2, 4))
    data.t_compr_f = lambda features: features
    data.t_compr_a = lambda features: features
    data.t_norm = lambda features: features
    data.t_subseq = lambda features: features.unsqueeze(0)
    features, labels = data[0]
    assert features.shape == (1, 2, 4)
    assert torch.equal(labels["call"], torch.tensor([1.0, 0.0, 1.0]))


@pytest.mark.parametrize(
    ("annotations", "match"),
    [
        ([["bad"], [], []], "unknown"),
        ([["bird"]], "one annotation per file"),
        (["bird", [], []], "not a string"),
        ([[], "rain", []], "not a string"),
        ([[3], [], []], "unknown or invalid"),
        ([["bird"], None, []], "iterable"),
    ],
)
def test_multilabel_rejects_invalid_annotations(make_dataset, annotations, match):
    with pytest.raises(ValueError, match=match):
        make_dataset(labels=annotations)


def test_multilabel_rejects_missing_annotations_even_with_prefix_labels(make_dataset):
    with pytest.raises(ValueError, match="explicit per-file labels"):
        make_dataset(labels=None)


@pytest.mark.parametrize(
    ("classes", "mapping", "match"),
    [
        (["bird", "bird", "rain"], None, "duplicate"),
        (["bird", "", "rain"], None, "nonempty"),
        (None, {"bird": 0, "frog": 0, "rain": 2}, "class_to_index"),
        (None, {"bird": 0, "frog": 1}, "class_to_index"),
        (None, {"bird": 0, "frog": 1, "rain": 3}, "class_to_index"),
        (None, {"bird": False, "frog": 1, "rain": 2}, "class_to_index"),
        (None, {"bird": 0, "frog": 1, "rain": 2, "extra": 3}, "class_to_index"),
    ],
)
def test_multilabel_rejects_invalid_class_declarations(make_dataset, classes, mapping, match):
    with pytest.raises(ValueError, match=match):
        make_dataset(labels=[[], [], []], classes=classes, class_to_index=mapping)


def test_one_class_multilabel_is_supported(make_dataset):
    data = make_dataset(
        classes=["bird"],
        labels=[["bird"], [], ["bird", "bird"]],
    )
    assert data._load_label("a.wav", 0)["call"].tolist() == [1.0]
    assert data._load_label("b.wav", 1)["call"].tolist() == [0.0]


def test_single_label_explicit_annotations_and_filename_prefix_still_work(make_dataset):
    single = make_dataset(
        labels=["bird", "frog", "rain"],
        task_config=TaskConfig(task_type="multiclass"),
    )
    assert single._load_label("first.wav", 0)["call"] == 0
    assert isinstance(single._load_label("first.wav", 0)["call"], int)

    parsed = make_dataset(labels=None, task_config=TaskConfig(task_type="multiclass"))
    assert parsed._load_label("rain-sample.wav", 0)["call"] == 2
    with pytest.raises(ValueError, match="unknown single-label class"):
        parsed._load_label("unknown-sample.wav", 0)
