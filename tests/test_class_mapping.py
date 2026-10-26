"""Unit regressions for binary class polarity and class mapping (SPEC-08)."""

import logging

import pytest
import torch

from bioacoustics_dl_toolbox.audio import datasets
from bioacoustics_dl_toolbox.config import TaskConfig


@pytest.fixture
def make_dataset(monkeypatch):
    # Construction and label extraction must not require an audio file or reader.
    monkeypatch.setattr(datasets, "AsyncFileReader", lambda: object())

    def build(classes, *, file_names=None, **kwargs):
        if file_names is None:
            file_names = [f"{name}-sample-{index}.wav" for index, name in enumerate(classes)]
        return datasets.SpectrogramDataset(file_names, classes=classes, **kwargs)

    return build


@pytest.mark.parametrize("classes", [["sparrow", "crow"], ["crow", "sparrow"]])
def test_non_target_binary_classes_use_declared_order_and_log_fallback(
    make_dataset, caplog, classes
):
    with caplog.at_level(logging.INFO, logger=datasets.__name__):
        data = make_dataset(classes)
    assert data.task_type == "binary"
    assert data.class_dist_dict == {name: index for index, name in enumerate(classes)}
    assert [data.get_class_name_from_index(i) for i in range(2)] == classes
    assert [data._load_label(path, i)["call"] for i, path in enumerate(data.file_names)] == [
        0,
        1,
    ]
    assert "class-list order" in caplog.text


@pytest.mark.parametrize("classes", [["target", "noise"], ["noise", "target"]])
def test_legacy_target_polarity_is_independent_of_order(make_dataset, classes):
    data = make_dataset(classes)
    assert data.class_dist_dict == {"target": 1, "noise": 0}
    assert data.get_class_name_from_index(0) == "noise"
    assert data.get_class_name_from_index(1) == "target"
    assert [data._get_class_index(name + "-sample.wav", i) for i, name in enumerate(classes)] == [
        data.class_dist_dict[name] for name in classes
    ]


def test_explicit_positive_class_changes_binary_polarity(make_dataset):
    data = make_dataset(["sparrow", "crow"], positive_class="sparrow")
    assert data.class_dist_dict == {"sparrow": 1, "crow": 0}
    assert data.get_class_name_from_index(1) == "sparrow"


def test_explicit_mapping_wins_over_legacy_target_and_class_order(make_dataset):
    data = make_dataset(["target", "noise"], class_to_index={"target": 0, "noise": 1})
    assert data.class_dist_dict == {"target": 0, "noise": 1}
    assert data.get_class_name_from_index(0) == "target"
    assert data._load_label("target-one.wav", 0)["call"] == 0
    assert data._load_label("noise-one.wav", 1)["call"] == 1


def test_consistent_positive_class_and_explicit_mapping_are_allowed(make_dataset):
    data = make_dataset(
        ["sparrow", "crow"],
        positive_class="sparrow",
        class_to_index={"sparrow": 1, "crow": 0},
    )
    assert data.class_dist_dict == {"sparrow": 1, "crow": 0}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"positive_class": "sparow"}, "positive_class.*sparow.*declared"),
        (
            {"positive_class": "sparrow", "class_to_index": {"sparrow": 0, "crow": 1}},
            "positive_class.*conflicts.*class_to_index",
        ),
    ],
)
def test_invalid_positive_class_is_rejected(make_dataset, options, message):
    with pytest.raises(ValueError, match=message):
        make_dataset(["sparrow", "crow"], **options)


@pytest.mark.parametrize("classes", [["bird", "bird"], ["bird", ""], [" ", "bird"]])
def test_duplicate_or_empty_class_names_are_rejected(make_dataset, classes):
    with pytest.raises(ValueError, match="classes.*(duplicate|nonempty)"):
        make_dataset(classes)


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"sparrow": 0, "crow": 0}, "class_to_index"),
        ({"sparrow": 0}, "class_to_index"),
        ({"sparrow": 0, "crow": 1, "extra": 2}, "class_to_index"),
        ({"sparrow": 1, "crow": 2}, "class_to_index"),
        ({"sparrow": -1, "crow": 1}, "class_to_index"),
        ({"sparrow": True, "crow": 0}, "class_to_index"),
        ({"sparrow": 0.0, "crow": 1}, "class_to_index"),
    ],
)
def test_invalid_explicit_mapping_is_rejected(make_dataset, mapping, message):
    with pytest.raises(ValueError, match=message):
        make_dataset(["sparrow", "crow"], class_to_index=mapping)


def test_binary_task_requires_exactly_two_classes(make_dataset):
    with pytest.raises(ValueError, match="binary.*exactly 2"):
        make_dataset(["bird", "frog", "rain"], task_config=TaskConfig(task_type="binary"))


def test_unknown_extracted_class_names_declared_classes(make_dataset):
    with pytest.raises(ValueError, match=r"robin.*sparrow.*crow"):
        make_dataset(
            ["sparrow", "crow"],
            file_names=["unknown-sample.wav"],
            label_extractor=lambda path: "robin",
        )


def test_unknown_explicit_annotation_names_declared_classes(make_dataset):
    with pytest.raises(ValueError, match=r"robin.*sparrow.*crow"):
        make_dataset(["sparrow", "crow"], labels=["robin", "crow"])


def test_multilabel_two_classes_never_uses_target_convention(make_dataset):
    data = make_dataset(
        ["target", "noise"],
        labels=[["target"], ["noise"]],
        task_config=TaskConfig(task_type="multilabel"),
    )
    assert data.class_dist_dict == {"target": 0, "noise": 1}
    assert data.get_class_name_from_index(0) == "target"
    assert torch.equal(data._load_label("one.wav", 0)["call"], torch.tensor([1.0, 0.0]))
    assert torch.equal(data._load_label("two.wav", 1)["call"], torch.tensor([0.0, 1.0]))


def test_multilabel_two_classes_honors_explicit_mapping(make_dataset):
    data = make_dataset(
        ["target", "noise"],
        labels=[["target"], ["noise"]],
        class_to_index={"target": 1, "noise": 0},
        task_config=TaskConfig(task_type="multilabel"),
    )
    assert data._load_label("one.wav", 0)["call"].tolist() == [0.0, 1.0]
    assert data._load_label("two.wav", 1)["call"].tolist() == [1.0, 0.0]


@pytest.mark.parametrize("task", ["multilabel", "multiclass"])
def test_positive_class_is_binary_only(make_dataset, task):
    labels = [["target"], ["noise"]] if task == "multilabel" else None
    with pytest.raises(ValueError, match="positive_class.*binary"):
        make_dataset(
            ["target", "noise"],
            labels=labels,
            task_config=TaskConfig(task_type=task),
            positive_class="target",
        )


def test_explicit_two_class_multiclass_follows_class_order(make_dataset):
    data = make_dataset(["target", "noise"], task_config=TaskConfig(task_type="multiclass"))
    assert data.class_dist_dict == {"target": 0, "noise": 1}
