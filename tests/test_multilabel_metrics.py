"""SPEC-05 unit tests: independent decisions, streaming counts and trainer dispatch."""

import logging
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bioacoustics_dl_toolbox.config import TaskConfig
from bioacoustics_dl_toolbox.metrics import (
    MultiLabelMetrics,
    multilabel_probabilities,
    threshold_multilabel,
)
from bioacoustics_dl_toolbox.metrics.core import Accuracy
from bioacoustics_dl_toolbox.training.trainer import Trainer


def test_sigmoid_keeps_independent_probabilities_and_autograd():
    logits = torch.tensor([[0.0, math.log(3), -math.log(3)]], requires_grad=True)
    probabilities = multilabel_probabilities(logits)
    assert probabilities.shape == logits.shape
    assert probabilities.dtype == logits.dtype
    torch.testing.assert_close(probabilities, torch.tensor([[0.5, 0.75, 0.25]]))
    assert probabilities.sum().item() == pytest.approx(1.5)
    probabilities.sum().backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, probabilities.detach() * (1 - probabilities.detach()))


@pytest.mark.parametrize(
    "bad",
    [torch.ones(2, dtype=torch.int64), torch.ones(1, 1, 2), torch.empty(1, 0)],
)
def test_logits_require_floating_matrix_with_classes(bad):
    with pytest.raises(ValueError, match="logits"):
        multilabel_probabilities(bad)


def test_inclusive_thresholds_and_per_class_override():
    probabilities = torch.tensor([[0.5, 0.75, 0.25], [0.4, 0.8, 0.6]])
    assert threshold_multilabel(probabilities).tolist() == [
        [True, True, False],
        [False, True, True],
    ]
    assert threshold_multilabel(probabilities, [0.5, 0.8, 0.2]).tolist() == [
        [True, False, True],
        [False, True, True],
    ]
    assert threshold_multilabel(probabilities, torch.tensor([0.5, 0.8, 0.2])).dtype == torch.bool
    assert threshold_multilabel(probabilities, torch.tensor(0.5)).shape == (2, 3)
    # A valid threshold just below one must not round to 1 in float16.
    assert threshold_multilabel(torch.tensor([[1.0]], dtype=torch.float16), 0.999999).item()


@pytest.mark.parametrize(
    "thresholds",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        1.0,
        -0.01,
        1.01,
        [0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, float("nan"), 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 1.0, 0.5],
        [[0.5, 0.5, 0.5]],
        "bad",
    ],
)
def test_invalid_thresholds_are_rejected(thresholds):
    with pytest.raises(ValueError, match="threshold"):
        threshold_multilabel(torch.zeros(2, 3), thresholds)


@pytest.mark.parametrize(
    "bad",
    [
        torch.ones(1, 3, dtype=torch.int64),
        torch.ones(1, 3, 1),
        torch.tensor([[float("nan")]]),
        torch.tensor([[1.01]]),
    ],
)
def test_invalid_probability_shapes_or_values(bad):
    with pytest.raises(ValueError, match="probabilities"):
        threshold_multilabel(bad)


def test_manual_confusion_counts_and_micro_macro_scores():
    # Columns: A = TP 1, FP 1, FN 0, TN 1; B = TP 1, FP 0, FN 1, TN 1;
    # C = no positives (unsupported) and TN 3.
    targets = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float32)
    predictions = torch.tensor([[True, False, False], [True, True, False], [False, False, False]])
    metric = MultiLabelMetrics(3, score="macro_f1")
    metric.update(targets, predictions)
    result = metric.compute()
    per_class = result["per_class"]
    assert per_class["tp"] == [1, 1, 0]
    assert per_class["fp"] == [1, 0, 0]
    assert per_class["fn"] == [0, 1, 0]
    assert per_class["tn"] == [1, 1, 3]
    assert per_class["support"] == [1, 2, 0]
    assert per_class["precision"] == pytest.approx([0.5, 1.0, 0.0])
    assert per_class["recall"] == pytest.approx([1.0, 0.5, 0.0])
    assert per_class["f1"] == pytest.approx([2 / 3, 2 / 3, 0])
    assert result["micro_precision"] == pytest.approx(2 / 3)
    assert result["micro_recall"] == pytest.approx(2 / 3)
    assert result["micro_f1"] == pytest.approx(2 / 3)
    assert result["macro_precision"] == pytest.approx(0.5)
    assert result["macro_recall"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(4 / 9)
    assert metric.get() == pytest.approx(4 / 9)


def test_empty_reset_and_all_negative_predictions_have_defined_zero_scores():
    metric = MultiLabelMetrics(2)
    before = metric.compute()
    assert all(
        before[name] == 0.0
        for name in (
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
    )
    assert before["per_class"]["support"] == [0, 0]
    metric.update(torch.zeros(3, 2), torch.zeros(3, 2, dtype=torch.bool))
    assert metric.compute()["per_class"]["tn"] == [3, 3]
    assert metric.get() == 0.0
    metric.update(torch.zeros(1, 2), torch.tensor([[True, False]]))
    assert metric.compute()["per_class"]["fp"] == [1, 0]
    assert metric.get() == 0.0  # unsupported classes never become NaN
    metric.reset(torch.device("cpu"))
    assert metric.compute() == before


def test_split_batches_aggregate_counts_not_batch_f1():
    targets = torch.ones(4, 1)
    predictions = torch.tensor([[True], [False], [False], [False]])
    single = MultiLabelMetrics(1)
    single.update(targets, predictions)
    split = MultiLabelMetrics(1)
    split.update(targets[:1], predictions[:1])  # batch F1 = 1
    split.update(targets[1:], predictions[1:])  # batch F1 = 0
    assert split.compute() == single.compute()
    assert split.get() == pytest.approx(0.4)  # not mean(batch F1) = (1 + 0) / 2
    # After one more missed positive: 1 TP, 4 FN -> F1 = 2 / 6.
    split.update(torch.tensor([[1.0]]), torch.tensor([[False]]))
    assert split.get() == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "score",
    [
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ],
)
def test_named_scores_are_scalar_metricbase_compatible(score):
    metric = MultiLabelMetrics(2, score=score)
    metric.update(torch.tensor([[1.0, 0.0]]), torch.tensor([[True, False]]))
    assert metric.get() == pytest.approx(metric.compute()[score])


@pytest.mark.parametrize(
    "targets, predictions, message",
    [
        (torch.zeros(2, 2), torch.zeros(2, 1, dtype=torch.bool), "predictions"),
        (torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool), "targets"),
        (torch.tensor([[1, 0]]), torch.tensor([[True, False]]), "targets"),
        (torch.tensor([[float("nan"), 0]]), torch.tensor([[True, False]]), "multi-hot"),
        (torch.tensor([[0.4, 0]]), torch.tensor([[True, False]]), "multi-hot"),
        (torch.tensor([[1.0, 0]]), torch.tensor([[1, 0]]), "predictions"),
    ],
)
def test_invalid_metric_inputs_rejected(targets, predictions, message):
    with pytest.raises(ValueError, match=message):
        MultiLabelMetrics(2).update(targets, predictions)


class BatchLoader(list):
    def __init__(self, logits_labels):
        super().__init__(
            [
                (logits, {"call": labels, "file_name": ["recording.wav"] * len(labels)})
                for logits, labels in logits_labels
            ]
        )
        self.dataset = SimpleNamespace(
            num_classes=2,
            task_config=TaskConfig(task_type="multilabel"),
            class_dist_dict={"bird": 0, "frog": 1},
        )


class IdentityModel(nn.Module):
    def forward(self, x):
        return x


def test_trainer_dispatch_uses_named_streaming_metrics_and_skips_single_label(monkeypatch):
    trainer = Trainer(
        IdentityModel(),
        logging.getLogger("multilabel-dispatch"),
        task_config=TaskConfig(task_type="multilabel"),
        multilabel_thresholds=[0.5, 0.75],
    )
    loader = BatchLoader(
        [
            (torch.tensor([[0.0, math.log(3)]]), torch.tensor([[1.0, 0.0]])),
            (
                torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
                torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
            ),
        ]
    )
    metric = MultiLabelMetrics(2, score="micro_f1")
    # None of the single-label AUC/confusion constructors or plotting paths may execute.
    def forbidden(*args, **kwargs):
        raise AssertionError("single-label metric/plot path called")

    import bioacoustics_dl_toolbox.training.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "AUCMeter", forbidden)
    monkeypatch.setattr(trainer_module, "ConfusionMeter", forbidden)
    monkeypatch.setattr(trainer_module.importlib, "import_module", forbidden)
    monkeypatch.setattr(trainer, "_write_classification_summaries", forbidden)
    monkeypatch.setattr(trainer, "_write_summaries", forbidden)
    result = {}

    def capture(**kwargs):
        result.update(kwargs)

    monkeypatch.setattr(trainer, "_write_scalar_summaries_logs", capture)
    assert (
        trainer._test_epoch(
            0,
            loader,
            nn.BCEWithLogitsLoss(),
            {"micro_f1": metric, "old_accuracy": Accuracy()},
            torch.device("cpu"),
        )
        > 0
    )
    assert list(result["metrics"]) == ["micro_f1"]
    assert metric.compute()["per_class"]["tp"] == [1, 0]
    assert metric.compute()["per_class"]["fp"] == [1, 1]
    assert metric.compute()["per_class"]["fn"] == [1, 1]
    assert metric.get() == pytest.approx(1 / 3)


def test_task_thresholds_validate_against_dataset_class_order():
    trainer = Trainer(
        IdentityModel(),
        logging.getLogger("threshold-validate"),
        task_config=TaskConfig(task_type="multilabel"),
        multilabel_thresholds=[0.1, 0.5, 0.8],
    )
    loader = BatchLoader([])
    with pytest.raises(ValueError, match="length-2"):
        trainer._validate_task_setup(loader, loader, None)
    trainer.multilabel_thresholds = [0.5, 0.75]
    trainer._validate_task_setup(loader, loader, None)
    probabilities = multilabel_probabilities(torch.tensor([[0.0, math.log(3)]]))
    decisions = threshold_multilabel(probabilities, trainer.multilabel_thresholds)
    torch.testing.assert_close(probabilities, torch.tensor([[0.5, 0.75]]))
    assert decisions.tolist() == [[True, True]]


def test_fit_selects_named_multilabel_score_for_validation(monkeypatch):
    # Stub the epoch methods: this verifies val_metric dispatch without a training run.
    model = nn.Linear(2, 2)
    trainer = Trainer(
        model,
        logging.getLogger("named-multilabel-fit"),
        task_config=TaskConfig(task_type="multilabel"),
    )
    loader = BatchLoader([])
    metric = MultiLabelMetrics(2, score="macro_f1")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler_scores = []

    class FakeScheduler:
        def step(self, value):
            scheduler_scores.append(value)

    def fake_train(epoch, train_loader, loss_fn, optimizer, metrics, device, checkpoint):
        assert list(metrics) == ["macro_f1"]
        return 0.5

    def fake_val(epoch, test_loader, loss_fn, metrics, device, **kwargs):
        metrics["macro_f1"].reset(device)
        metrics["macro_f1"].update(torch.tensor([[1.0, 0.0]]), torch.tensor([[True, False]]))
        return 0.9  # selecting loss instead of the named score would use 0.9

    monkeypatch.setattr(trainer, "_train_epoch", fake_train)
    monkeypatch.setattr(trainer, "_test_epoch", fake_val)
    trained_model = trainer.fit(
        loader,
        loader,
        None,
        nn.BCEWithLogitsLoss(),
        optimizer,
        FakeScheduler(),
        n_epochs=1,
        val_interval=1,
        patience_early_stopping=2,
        device=torch.device("cpu"),
        metrics={"macro_f1": metric},
        val_metric="macro_f1",
        val_metric_mode="max",
    )
    assert trained_model is model
    assert scheduler_scores == [pytest.approx(0.5)]  # (F1=1 + F1=0) / 2 classes


def test_fit_rejects_single_label_val_metric_for_multilabel():
    model = nn.Linear(2, 2)
    trainer = Trainer(
        model,
        logging.getLogger("invalid-multilabel-fit"),
        task_config=TaskConfig(task_type="multilabel"),
    )
    loader = BatchLoader([])
    with pytest.raises(ValueError, match="val_metric 'accuracy' is unavailable"):
        trainer.fit(
            loader,
            loader,
            None,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.01),
            object(),
            n_epochs=1,
            val_interval=1,
            patience_early_stopping=2,
            device=torch.device("cpu"),
            metrics={"accuracy": Accuracy()},
            val_metric="accuracy",
            val_metric_mode="max",
        )
