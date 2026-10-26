"""Unit tests for the v0.2 classification task contract (SPEC-03)."""

import json
import logging
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bioacoustics_dl_toolbox import TaskConfig, resolve_task_type
from bioacoustics_dl_toolbox.audio import datasets
from bioacoustics_dl_toolbox.config import ClassifierConfig
from bioacoustics_dl_toolbox.models.classifier import Classifier
from bioacoustics_dl_toolbox.tasks import validate_task_batch
from bioacoustics_dl_toolbox.training.trainer import Trainer


@pytest.mark.parametrize(
    ("setting", "classes", "expected"),
    [
        ("auto", 2, "binary"),
        ("auto", 3, "multiclass"),
        ("binary", 2, "binary"),
        ("multiclass", 2, "multiclass"),
        ("multiclass", 3, "multiclass"),
        ("multilabel", 1, "multilabel"),
        ("multilabel", 2, "multilabel"),
        ("multilabel", 5, "multilabel"),
    ],
)
def test_resolve_valid_task_types(setting, classes, expected):
    assert resolve_task_type(TaskConfig(task_type=setting), classes) == expected


def test_default_auto_uses_two_logits_for_binary():
    assert resolve_task_type(TaskConfig(), 2) == "binary"
    assert resolve_task_type(None, 2) == "binary"
    assert Classifier(ClassifierConfig(num_classes=2)).linear.out_features == 2


@pytest.mark.parametrize(
    ("setting", "classes", "message"),
    [
        ("auto", 1, "auto.*num_classes=1"),
        ("binary", 1, "binary.*exactly 2"),
        ("binary", 3, "binary.*exactly 2"),
        ("multiclass", 1, "multiclass.*at least 2"),
        ("multilabel", 0, "positive integer"),
        ("auto", -1, "positive integer"),
        ("unknown", 2, "Unknown task_type"),
    ],
)
def test_resolve_invalid_task_types(setting, classes, message):
    with pytest.raises(ValueError, match=message):
        resolve_task_type(TaskConfig(task_type=setting), classes)


def test_task_config_is_frozen_and_json_serializable():
    config = TaskConfig(task_type="multilabel")
    assert asdict(config) == {"task_type": "multilabel"}
    assert json.loads(json.dumps(asdict(config))) == {"task_type": "multilabel"}
    with pytest.raises(FrozenInstanceError):
        config.task_type = "auto"


def test_classifier_keeps_raw_logits_for_each_class_count():
    for num_classes in (1, 2, 3):
        classifier = Classifier(ClassifierConfig(input_channels=4, num_classes=num_classes))
        with torch.no_grad():
            classifier.linear.weight.zero_()
            classifier.linear.bias.copy_(torch.arange(num_classes, dtype=torch.float) - 1)
        raw_logits = classifier(torch.zeros(2, 4, 3))
        assert raw_logits.shape == (2, num_classes)
        assert torch.equal(
            raw_logits, (torch.arange(num_classes, dtype=torch.float) - 1).expand(2, -1)
        )
        assert (raw_logits < 0).any()  # no sigmoid/softmax in the classifier


@pytest.mark.parametrize("task_type", ["binary", "multiclass"])
def test_validate_single_label_shape_and_dtype(task_type):
    config = TaskConfig(task_type=task_type)
    logits = torch.zeros(3, 2)
    validate_task_batch(logits, torch.tensor([0, 1, 0]), config, 2)
    with pytest.raises(ValueError, match=r"targets must have shape \[B\]"):
        validate_task_batch(logits, torch.zeros(3, 2), config, 2)
    with pytest.raises(ValueError, match="integer dtype"):
        validate_task_batch(logits, torch.zeros(3), config, 2)
    with pytest.raises(ValueError, match="logits must have shape"):
        validate_task_batch(torch.zeros(3, 1), torch.zeros(3, dtype=torch.long), config, 2)
    with pytest.raises(ValueError, match="logits must have a floating dtype"):
        validate_task_batch(
            torch.zeros(3, 2, dtype=torch.int64), torch.zeros(3, dtype=torch.long), config, 2
        )


def test_validate_multilabel_shape_and_dtype():
    config = TaskConfig(task_type="multilabel")
    validate_task_batch(torch.zeros(3, 1), torch.zeros(3, 1), config, 1)
    validate_task_batch(torch.zeros(2, 3), torch.zeros(2, 3), config, 3)
    with pytest.raises(ValueError, match="matching logits"):
        validate_task_batch(torch.zeros(2, 3), torch.zeros(2), config, 3)
    with pytest.raises(ValueError, match="floating dtype"):
        validate_task_batch(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.long), config, 3)


def make_dataset(monkeypatch, task_config=TaskConfig(), classes=None):
    monkeypatch.setattr(datasets, "AsyncFileReader", lambda: object())
    return datasets.SpectrogramDataset(
        [], classes=classes or ["noise", "target"], task_config=task_config
    )


def test_dataset_preserves_auto_single_label_and_explicit_multiclass(monkeypatch):
    legacy = make_dataset(monkeypatch)
    assert legacy.task_type == "binary"
    assert legacy.num_classes == 2
    assert legacy._load_label("target-example.wav", 0)["call"] == 1

    multiclass = make_dataset(monkeypatch, TaskConfig(task_type="multiclass"))
    assert multiclass.task_type == "multiclass"
    assert multiclass._load_label("noise-example.wav", 0)["call"] == 0


def test_dataset_multilabel_requires_explicit_annotations(monkeypatch):
    with pytest.raises(ValueError, match="explicit per-file labels"):
        make_dataset(monkeypatch, TaskConfig(task_type="multilabel"))


def make_trainer(task_config=TaskConfig(), num_classes=2):
    model = Classifier(ClassifierConfig(input_channels=4, num_classes=num_classes))
    return Trainer(model=model, logger=logging.getLogger(__name__), task_config=task_config)


def make_loader(task_config=TaskConfig(), num_classes=2, mapping=None):
    if mapping is None:
        mapping = {f"class_{i}": i for i in range(num_classes)}
    return SimpleNamespace(
        dataset=SimpleNamespace(
            task_config=task_config, num_classes=num_classes, class_dist_dict=mapping
        )
    )


def test_trainer_detects_task_mismatch_before_training():
    trainer = make_trainer()
    train = make_loader()
    val = make_loader(TaskConfig(task_type="multiclass"))
    with pytest.raises(ValueError, match="val dataset task 'multiclass'.*trainer task 'binary'"):
        trainer._validate_task_setup(train, val, None)


def test_trainer_detects_class_count_mapping_and_head_mismatch():
    trainer = make_trainer()
    train = make_loader()
    with pytest.raises(ValueError, match="val dataset has 3 classes"):
        trainer._validate_task_setup(train, make_loader(num_classes=3), None)
    with pytest.raises(ValueError, match="class_to_index differs"):
        trainer._validate_task_setup(train, make_loader(mapping={"class_0": 1, "class_1": 0}), None)
    with pytest.raises(ValueError, match="Classifier output num_classes=3"):
        make_trainer(num_classes=3)._validate_task_setup(train, make_loader(), None)


def test_trainer_accepts_legacy_task_and_explicit_two_class_multiclass():
    train = make_loader()
    trainer = make_trainer()
    trainer._validate_task_setup(train, make_loader(), None)
    assert trainer.task_type == "binary"

    multiclass = TaskConfig(task_type="multiclass")
    trainer = make_trainer(multiclass)
    trainer._validate_task_setup(make_loader(multiclass), make_loader(multiclass), None)
    assert trainer.task_type == "multiclass"


def test_trainer_accepts_multilabel_task_configuration():
    multilabel = TaskConfig(task_type="multilabel")
    trainer = make_trainer(multilabel)
    trainer._validate_task_setup(make_loader(multilabel), make_loader(multilabel), None)
    assert trainer.task_type == "multilabel"


def test_legacy_binary_one_step_still_uses_integer_targets():
    trainer = make_trainer()
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    batch = (torch.zeros(2, 4, 3), {"call": torch.tensor([0, 1]), "file_name": ["a", "b"]})

    class TinyLoader:
        dataset = make_loader().dataset

        def __iter__(self):
            yield batch

    loss = trainer._train_epoch(
        epoch=0,
        train_loader=TinyLoader(),
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        metrics={},
        device=torch.device("cpu"),
        checkpoint=lambda: None,
    )
    assert loss > 0


def test_single_label_trainer_rejects_float_targets_before_silent_cast():
    trainer = make_trainer()
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

    class TinyLoader:
        dataset = make_loader().dataset

        def __iter__(self):
            yield (torch.zeros(2, 4, 3), {"call": torch.tensor([0.0, 1.0])})

    with pytest.raises(ValueError, match="integer dtype"):
        trainer._train_epoch(
            epoch=0,
            train_loader=TinyLoader(),
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            metrics={},
            device=torch.device("cpu"),
            checkpoint=lambda: None,
        )


def test_resume_checkpoint_refuses_task_mismatch():
    trainer = make_trainer()
    trainer.checkpoint_handler = SimpleNamespace(read_latest=lambda: {"taskType": "multiclass"})
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="Resume checkpoint task 'multiclass'.*'binary'"):
        trainer.fit(
            train_loader=make_loader(),
            val_loader=make_loader(),
            test_loader=None,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=SimpleNamespace(),
            n_epochs=0,
            val_interval=1,
            patience_early_stopping=2,
            device=torch.device("cpu"),
        )


def test_explicit_two_class_multiclass_uses_confusion_not_binary_auc(monkeypatch):
    import bioacoustics_dl_toolbox.training.trainer as trainer_module

    def fail_binary_auc():
        pytest.fail("binary AUC must not run for explicit multiclass")

    monkeypatch.setattr(trainer_module, "AUCMeter", fail_binary_auc)
    task = TaskConfig(task_type="multiclass")
    trainer = make_trainer(task)
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

    class TinyLoader:
        dataset = make_loader(task).dataset

        def __iter__(self):
            yield (
                torch.zeros(2, 4, 3),
                {"call": torch.tensor([0, 1]), "file_name": ["a", "b"]},
            )

    loss = trainer._train_epoch(
        epoch=0,
        train_loader=TinyLoader(),
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        metrics={},
        device=torch.device("cpu"),
        checkpoint=lambda: None,
    )
    assert loss > 0
