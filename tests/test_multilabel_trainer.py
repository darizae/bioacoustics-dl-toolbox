"""Isolated one-step trainer checks for SPEC-04 (no training runs or audio I/O)."""

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bioacoustics_dl_toolbox.config import TaskConfig
from bioacoustics_dl_toolbox.metrics.core import Accuracy
from bioacoustics_dl_toolbox.training.trainer import Trainer


class FakeLoader(list):
    def __init__(self, inputs, labels, *, task="multilabel", mapping=None):
        super().__init__([(inputs, {"call": labels, "file_name": ["a.wav"] * len(labels)})])
        if mapping is None:
            mapping = {f"class_{i}": i for i in range(3 if task == "multilabel" else 2)}
        self.dataset = SimpleNamespace(
            task_config=TaskConfig(task_type=task),
            num_classes=len(mapping),
            class_dist_dict=mapping,
        )


class RecordingBCE(nn.BCEWithLogitsLoss):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, logits, targets):
        self.seen.append((logits.shape, targets.shape, targets.dtype, targets.detach().clone()))
        return super().forward(logits, targets)


class RecordingCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, logits, targets):
        self.seen.append((tuple(logits.shape), tuple(targets.shape), targets.dtype))
        return super().forward(logits, targets)


class RecordingMultiLabelMetric:
    supports_multilabel = True

    def __init__(self):
        self.calls = []

    def reset(self, device):
        self.calls = []

    def update(self, targets, predictions):
        self.calls.append((targets.clone(), predictions.clone()))

    def get(self):
        return 0.0


def trainer(task="multilabel", outputs=3):
    model = nn.Linear(2, outputs)
    return Trainer(
        model=model,
        logger=logging.getLogger(__name__),
        task_config=TaskConfig(task_type=task),
    )


def test_multilabel_train_step_uses_bce_float_gradients_and_no_argmax(monkeypatch):
    instance = trainer()
    loss = RecordingBCE()
    metric = RecordingMultiLabelMetric()
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.int64)
    loader = FakeLoader(inputs, targets)
    optimizer = torch.optim.SGD(instance.model.parameters(), lr=0.1)
    before = instance.model.weight.detach().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("argmax must not run for multilabel")

    monkeypatch.setattr(torch, "argmax", forbidden)
    monkeypatch.setattr(instance, "_write_summaries", forbidden)
    monkeypatch.setattr(instance, "_write_classification_summaries", forbidden)
    score = instance._train_epoch(
        0, loader, loss, optimizer, {"ml": metric}, torch.device("cpu"), lambda: None
    )

    assert score > 0
    assert loss.seen[0][:3] == (torch.Size([2, 3]), torch.Size([2, 3]), torch.float32)
    assert torch.equal(loss.seen[0][3], targets.float())
    assert instance.model.weight.grad is not None
    assert not torch.equal(instance.model.weight.detach(), before)
    assert len(metric.calls) == 1
    y_true, y_pred = metric.calls[0]
    assert y_true.shape == y_pred.shape == (2, 3)
    assert y_true.dtype == y_pred.dtype == torch.bool
    assert torch.equal(y_true, targets.bool())


def test_multilabel_validation_all_zero_targets_loss_without_single_label_summaries(monkeypatch):
    instance = trainer()
    metric = RecordingMultiLabelMetric()
    loader = FakeLoader(torch.zeros(2, 2), torch.zeros(2, 3))
    loss = RecordingBCE()

    def forbidden(*args, **kwargs):
        raise AssertionError("single-label operation must not run")

    monkeypatch.setattr(torch, "argmax", forbidden)
    monkeypatch.setattr(instance, "_write_summaries", forbidden)
    monkeypatch.setattr(instance, "_write_classification_summaries", forbidden)
    score = instance._test_epoch(0, loader, loss, {"ml": metric}, torch.device("cpu"))
    assert score > 0
    assert len(metric.calls) == 1
    assert not metric.calls[0][0].any()
    assert metric.calls[0][1].shape == (2, 3)


def test_rejects_wrong_loss_and_logit_shapes_before_bce():
    instance = trainer(outputs=2)
    loader = FakeLoader(torch.zeros(2, 2), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="BCEWithLogitsLoss"):
        instance._test_epoch(0, loader, nn.CrossEntropyLoss(), {}, torch.device("cpu"))
    with pytest.raises(ValueError, match=r"logits must have shape \[B, 3\]"):
        instance._test_epoch(0, loader, nn.BCEWithLogitsLoss(), {}, torch.device("cpu"))


def test_multilabel_rejects_wrong_target_shape():
    instance = trainer()
    loader = FakeLoader(torch.zeros(2, 2), torch.zeros(2, 2))
    with pytest.raises(ValueError, match="matching logits"):
        instance._test_epoch(0, loader, nn.BCEWithLogitsLoss(), {}, torch.device("cpu"))


def test_multilabel_skips_legacy_metrics_and_warns(caplog):
    instance = trainer()
    loader = FakeLoader(torch.zeros(2, 2), torch.zeros(2, 3))
    with caplog.at_level(logging.WARNING):
        score = instance._test_epoch(
            0, loader, nn.BCEWithLogitsLoss(), {"accuracy": Accuracy()}, torch.device("cpu")
        )
    assert score > 0
    assert "Skipping single-label metrics" in caplog.text
    assert "ROC/confusion" in caplog.text


def test_multilabel_requires_consistent_task_and_class_index_mapping():
    instance = trainer()
    train = FakeLoader(torch.zeros(1, 2), torch.zeros(1, 3))
    val = FakeLoader(
        torch.zeros(1, 2),
        torch.zeros(1, 3),
        mapping={"class_0": 1, "class_1": 0, "class_2": 2},
    )
    with pytest.raises(ValueError, match="class_to_index differs"):
        instance._validate_task_setup(train, val, None)
    with pytest.raises(ValueError, match="dataset task 'multiclass' does not match"):
        instance._validate_task_setup(
            train,
            FakeLoader(
                torch.zeros(1, 2), torch.zeros(1, 3), task="multiclass",
                mapping={"class_0": 0, "class_1": 1, "class_2": 2},
            ),
            None,
        )


def test_single_label_training_and_validation_keep_integer_targets():
    instance = trainer("binary", outputs=2)
    loader = FakeLoader(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]), task="binary")
    loss = RecordingCrossEntropy()
    optimizer = torch.optim.SGD(instance.model.parameters(), lr=0.01)
    assert (
        instance._train_epoch(0, loader, loss, optimizer, {}, torch.device("cpu"), lambda: None)
        > 0
    )
    assert instance._test_epoch(0, loader, loss, {}, torch.device("cpu")) > 0
    assert loss.seen == [((2, 2), (2,), torch.int64), ((2, 2), (2,), torch.int64)]
