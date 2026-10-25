"""
Training loop with validation, testing, checkpointing, and TensorBoard logging.

Ported from ANIMAL-SPOT ``trainer.py`` (Bergler & Schroeter, GPL-3.0).
Key changes:
- Accepts an optional ``SummaryWriter``; does not create one internally.
- Uses standard ``logging.Logger`` instead of Singleton Logger.
- Visualization helpers are called as standalone functions.
"""

from __future__ import annotations

import copy
import importlib
import logging
import math
import operator
import platform
import time
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from bioacoustics_dl_toolbox.config import ResolvedTaskType, TaskConfig, resolve_task_type
from bioacoustics_dl_toolbox.metrics.auc import AUCMeter
from bioacoustics_dl_toolbox.metrics.confusion import ConfusionMeter
from bioacoustics_dl_toolbox.metrics.core import Mean, MetricBase, Sum
from bioacoustics_dl_toolbox.metrics.multilabel import (
    multilabel_probabilities,
    threshold_multilabel,
)
from bioacoustics_dl_toolbox.tasks import validate_task_batch
from bioacoustics_dl_toolbox.training.checkpoints import (
    CheckpointHandler,
    validate_resume_checkpoint_task,
)
from bioacoustics_dl_toolbox.training.early_stopping import EarlyStoppingCriterion

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment, misc]


class Trainer:
    """Manages the train/val/test loop with checkpointing and logging.

    Parameters
    ----------
    model:
        The full ``nn.Sequential(encoder, classifier)`` model.
    logger:
        A ``logging.Logger`` instance.
    prefix:
        Name prefix for checkpoints and TensorBoard runs.
    checkpoint_dir:
        Directory for saving checkpoints. ``None`` to disable.
    summary_dir:
        Directory for TensorBoard summaries. ``None`` to disable.
    n_summaries:
        Number of sample images to write per epoch.
    start_scratch:
        If ``True``, ignore existing checkpoints.
    task_config:
        Task contract, defaulting to legacy single-label ``auto`` behavior.
        Explicit multilabel uses multi-hot targets and BCEWithLogitsLoss.
    multilabel_thresholds:
        Scalar or class-index-ordered thresholds, strictly between 0 and 1,
        for independent multi-label decisions (default 0.5).
    """

    def __init__(
        self,
        model: nn.Module,
        logger: logging.Logger,
        prefix: str = "",
        checkpoint_dir: str | None = None,
        summary_dir: str | None = None,
        n_summaries: int = 4,
        input_shape: tuple[int, ...] | None = None,
        start_scratch: bool = False,
        task_config: TaskConfig = TaskConfig(),
        multilabel_thresholds: float | Sequence[float] | torch.Tensor = 0.5,
    ) -> None:
        self.model = model
        self.task_config = task_config
        self.multilabel_thresholds = multilabel_thresholds
        self.task_type: ResolvedTaskType | None = None
        self._multilabel_warning_emitted = False
        self._multilabel_metrics_warning_emitted = False
        self.logger = logger
        self.prefix = prefix
        self.n_summaries = n_summaries

        self.logger.info("Init summary writer")
        if summary_dir is not None and SummaryWriter is not None:
            run_name = prefix + "_" if prefix != "" else ""
            run_name += "{time}-{host}".format(
                time=time.strftime("%y-%m-%d-%H-%M", time.localtime()),
                host=platform.uname()[1],
            )
            import os

            summary_dir = os.path.join(summary_dir, run_name)
            self.writer: Any = SummaryWriter(summary_dir)
            if input_shape is not None:
                dummy_input = torch.rand(input_shape)
                self.logger.info("Writing graph to summary")
                self.writer.add_graph(self.model, dummy_input)
        else:
            self.writer = None

        if checkpoint_dir is not None:
            self.checkpoint_handler: CheckpointHandler | None = CheckpointHandler(
                checkpoint_dir, prefix=prefix, logger=self.logger
            )
        else:
            self.checkpoint_handler = None

        self.start_scratch = start_scratch
        self.class_dist_dict: dict[str, int] | None = None

    def _resolve_loader_task(
        self, loader: torch.utils.data.DataLoader, phase: str
    ) -> ResolvedTaskType:
        """Check the loader's resolved task against this trainer's task."""
        num_classes = loader.dataset.num_classes  # type: ignore[union-attr]
        trainer_type = resolve_task_type(self.task_config, num_classes)
        dataset_config = getattr(loader.dataset, "task_config", TaskConfig())
        dataset_type = resolve_task_type(dataset_config, num_classes)
        if trainer_type != dataset_type:
            raise ValueError(
                f"{phase} dataset task '{dataset_type}' does not match trainer task "
                f"'{trainer_type}'; pass a consistent task_config to both"
            )
        if self.task_type is not None and dataset_type != self.task_type:
            raise ValueError(
                f"{phase} dataset task '{dataset_type}' does not match "
                f"the training task '{self.task_type}'"
            )
        return dataset_type

    def _validate_task_setup(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader | None,
    ) -> None:
        """Reject mismatched datasets/classifier before beginning training."""
        num_classes = train_loader.dataset.num_classes  # type: ignore[union-attr]
        self.task_type = self._resolve_loader_task(train_loader, "train")
        train_mapping = getattr(train_loader.dataset, "class_dist_dict", None)
        if self.task_type == "multilabel":
            threshold_multilabel(torch.empty((0, num_classes)), self.multilabel_thresholds)
            if (
                train_mapping is None
                or len(train_mapping) != num_classes
                or set(train_mapping.values()) != set(range(num_classes))
            ):
                raise ValueError(
                    "multilabel train dataset requires a contiguous class_to_index mapping"
                )
        for phase, loader in (("val", val_loader), ("test", test_loader)):
            if loader is None:
                continue
            count = loader.dataset.num_classes  # type: ignore[union-attr]
            if count != num_classes:
                raise ValueError(
                    f"{phase} dataset has {count} classes; train dataset has {num_classes}"
                )
            self._resolve_loader_task(loader, phase)
            mapping = getattr(loader.dataset, "class_dist_dict", None)
            if self.task_type == "multilabel" and mapping is None:
                raise ValueError(f"{phase} dataset requires class_to_index in multilabel mode")
            if train_mapping is not None and mapping is not None and mapping != train_mapping:
                raise ValueError(f"{phase} dataset class_to_index differs from train dataset")

        head = (
            self.model[-1]
            if isinstance(self.model, nn.Sequential) and len(self.model)
            else self.model
        )
        configured_classes = getattr(getattr(head, "config", None), "num_classes", None)
        if configured_classes is not None and configured_classes != num_classes:
            raise ValueError(
                f"Classifier output num_classes={configured_classes} does not match "
                f"dataset num_classes={num_classes}"
            )

    @staticmethod
    def _validate_loss_fn(task_type: ResolvedTaskType, loss_fn: nn.Module) -> None:
        """Multilabel logits require BCEWithLogitsLoss (never sigmoid + BCE)."""
        if task_type == "multilabel" and not isinstance(loss_fn, nn.BCEWithLogitsLoss):
            raise ValueError("multilabel training requires nn.BCEWithLogitsLoss on raw logits")

    def _metrics_for_task(
        self,
        metrics: dict[str, MetricBase] | list[MetricBase],
        task_type: ResolvedTaskType,
    ) -> dict[str, MetricBase] | list[MetricBase]:
        """Use only metrics that explicitly support multi-label targets.

        ``MultiLabelMetrics`` implements the existing reset/update/get API
        with corpus-wide counters, including named micro/macro scores. Other
        metrics must opt in with ``supports_multilabel = True``.
        """
        if task_type != "multilabel":
            return metrics
        if not self._multilabel_warning_emitted:
            self.logger.warning(
                "Multilabel ROC/confusion and per-sample class image summaries are disabled; "
                "only metrics declaring supports_multilabel=True are logged."
            )
            self._multilabel_warning_emitted = True
        if isinstance(metrics, dict):
            selected = {
                name: metric
                for name, metric in metrics.items()
                if getattr(metric, "supports_multilabel", False)
            }
        else:
            selected = [
                metric for metric in metrics if getattr(metric, "supports_multilabel", False)
            ]
        if len(selected) != len(metrics) and not self._multilabel_metrics_warning_emitted:
            self.logger.warning("Skipping single-label metrics for multilabel %s", task_type)
            self._multilabel_metrics_warning_emitted = True
        return selected

    @staticmethod
    def _task_targets(
        raw_label: torch.Tensor | None, device: torch.device, task_type: ResolvedTaskType
    ) -> torch.Tensor | None:
        if raw_label is None:
            return None
        dtype = torch.float32 if task_type == "multilabel" else torch.int64
        return raw_label.to(device, non_blocking=True, dtype=dtype)

    @staticmethod
    def _task_predictions(
        output: torch.Tensor,
        task_type: ResolvedTaskType,
        thresholds: float | Sequence[float] | torch.Tensor = 0.5,
    ) -> torch.Tensor:
        if task_type == "multilabel":
            # Keep independent probabilities separate from boolean decisions.
            return threshold_multilabel(multilabel_probabilities(output.detach()), thresholds)
        return torch.argmax(output.detach(), dim=1)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader | None,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        n_epochs: int,
        val_interval: int,
        patience_early_stopping: int,
        device: torch.device,
        metrics: dict[str, MetricBase] | list[MetricBase] = {},
        val_metric: str = "loss",
        val_metric_mode: str = "min",
        start_epoch: int = 0,
        checkpoint: Callable[[], None] | None = None,
        epoch_callback: Callable[[int, int, str], None] | None = None,
    ) -> nn.Module:
        """Run train → validate and an optional final test pass.

        Returns the model loaded with the best validation weights.
        """
        self._validate_task_setup(train_loader, val_loader, test_loader)
        self._validate_loss_fn(self.task_type, loss_fn)
        selected_metrics = self._metrics_for_task(metrics, self.task_type)
        if self.task_type == "multilabel" and val_metric != "loss":
            if isinstance(selected_metrics, dict):
                valid_metric = val_metric in selected_metrics
            else:
                valid_metric = isinstance(val_metric, int) and 0 <= val_metric < len(
                    selected_metrics
                )
            if not valid_metric:
                raise ValueError(
                    f"val_metric {val_metric!r} is unavailable for task {self.task_type}"
                )
        checkpoint = checkpoint or (lambda: None)
        epoch_callback = epoch_callback or (lambda _current, _total, _phase: None)
        self.logger.info("Init model on device '{}'".format(device))
        self.model = self.model.to(device)
        self.class_dist_dict = train_loader.dataset.class_dist_dict  # type: ignore[union-attr]

        best_model = copy.deepcopy(self.model.state_dict())
        best_metric = 0.0 if val_metric_mode == "max" else float("inf")

        patience_stopping = math.ceil(patience_early_stopping / val_interval)
        patience_stopping = int(max(1, patience_stopping))
        early_stopping = EarlyStoppingCriterion(
            mode=val_metric_mode,
            patience=patience_stopping,  # type: ignore[arg-type]
        )

        if not self.start_scratch and self.checkpoint_handler is not None:
            saved_checkpoint = self.checkpoint_handler.read_latest()
            if saved_checkpoint is not None:
                validate_resume_checkpoint_task(saved_checkpoint, self.task_type)
                try:
                    try:
                        self.model.load_state_dict(saved_checkpoint["modelState"])
                    except RuntimeError as e:
                        self.logger.error(
                            "Failed to restore checkpoint: Checkpoint has different parameters"
                        )
                        self.logger.error(str(e))
                        raise
                    optimizer.load_state_dict(saved_checkpoint["trainState"]["optState"])
                    start_epoch = saved_checkpoint["trainState"]["epoch"] + 1
                    best_metric = saved_checkpoint["trainState"]["best_metric"]
                    best_model = saved_checkpoint["trainState"]["best_model"]
                    early_stopping.load_state_dict(saved_checkpoint["trainState"]["earlyStopping"])
                    scheduler.load_state_dict(saved_checkpoint["trainState"]["scheduler"])
                    self.logger.info("Resuming with epoch {}".format(start_epoch))
                except KeyError:
                    self.logger.error("Failed to restore checkpoint")
                    raise

        since = time.time()
        self.logger.info("Class Distribution: " + str(self.class_dist_dict))
        self.logger.info("Start training model " + self.prefix)

        try:
            val_comp = operator.gt if val_metric_mode == "max" else operator.lt

            for epoch in range(start_epoch, n_epochs):
                checkpoint()
                self._train_epoch(
                    epoch, train_loader, loss_fn, optimizer, selected_metrics, device, checkpoint
                )
                epoch_callback(epoch + 1, n_epochs, "train")
                if epoch % val_interval == 0 or epoch == n_epochs - 1:
                    val_loss = self._test_epoch(
                        epoch,
                        val_loader,
                        loss_fn,
                        selected_metrics,
                        device,
                        phase="val",
                        checkpoint=checkpoint,
                    )
                    epoch_callback(epoch + 1, n_epochs, "val")
                    if val_metric == "loss":
                        val_result = val_loss
                    else:
                        val_result = selected_metrics[val_metric].get()  # type: ignore[index]

                    if val_comp(val_result, best_metric):
                        best_metric = val_result
                        best_model = copy.deepcopy(self.model.state_dict())

                    if self.checkpoint_handler is not None:
                        self.checkpoint_handler.write(
                            {
                                "modelState": self.model.state_dict(),
                                "trainState": {
                                    "epoch": epoch,
                                    "best_metric": best_metric,
                                    "best_model": best_model,
                                    "optState": optimizer.state_dict(),
                                    "earlyStopping": early_stopping.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                },
                                "classes": self.class_dist_dict,
                                "taskType": self.task_type,
                            }
                        )

                    scheduler.step(val_result)

                    if early_stopping.step(val_result):
                        self.logger.info(
                            "No improvement over the last {} epochs. Stopping.".format(
                                patience_early_stopping
                            )
                        )
                        break
        except Exception:
            import traceback

            self.logger.warning(traceback.format_exc())
            self.logger.warning("Aborting...")
            raise

        self.model.load_state_dict(best_model)
        final_metric = None
        if test_loader is not None:
            final_loss = self._test_epoch(
                0,
                test_loader,
                loss_fn,
                selected_metrics,
                device,
                phase="test",
                checkpoint=checkpoint,
            )
            if val_metric == "loss":
                final_metric = final_loss
            else:
                final_metric = selected_metrics[val_metric].get()  # type: ignore[index]

        time_elapsed = time.time() - since
        self.logger.info(
            "Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
        )
        self.logger.info("Best val metric: {:4f}".format(best_metric))
        if final_metric is not None:
            self.logger.info("Final test metric: {:4f}".format(final_metric))

        return self.model

    def _train_epoch(
        self,
        epoch: int,
        train_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, MetricBase] | list[MetricBase],
        device: torch.device,
        checkpoint: Callable[[], None],
    ) -> float:
        self.logger.debug("train|{}|start".format(epoch))
        self.model.train()

        epoch_start = time.time()
        start_data_loading = epoch_start
        data_loading_time = Sum(torch.device("cpu"))
        epoch_loss = Mean(device)

        num_classes = train_loader.dataset.num_classes  # type: ignore[union-attr]
        task_type = self._resolve_loader_task(train_loader, "train")
        self._validate_loss_fn(task_type, loss_fn)
        active_metrics = self._metrics_for_task(metrics, task_type)
        self._reset_metrics(active_metrics, device)
        if task_type == "binary":
            auc: AUCMeter | None = AUCMeter()
            confusion: ConfusionMeter | None = None
        elif task_type == "multiclass":
            auc = None
            confusion = ConfusionMeter(n_categories=num_classes)
        else:
            auc = None
            confusion = None

        for i, (features, label) in enumerate(train_loader):
            checkpoint()
            features = features.to(device)
            raw_label: torch.Tensor | None = label.get("call")
            call_label = self._task_targets(raw_label, device, task_type)

            data_loading_time.update(torch.Tensor([(time.time() - start_data_loading)]))

            optimizer.zero_grad()
            output = self.model(features)
            if raw_label is not None and task_type != "multilabel":
                # Keep the SPEC-03 guard: do not silently coerce invalid
                # single-label floating annotations into class indices.
                validate_task_batch(output, raw_label, self.task_config, num_classes)
            if call_label is not None:
                validate_task_batch(output, call_label, self.task_config, num_classes)
            loss = loss_fn(output, call_label)
            loss.backward()
            optimizer.step()

            epoch_loss.update(loss)

            prediction: torch.Tensor | None = None
            if call_label is not None:
                prediction = self._task_predictions(output, task_type, self.multilabel_thresholds)
                metric_targets = call_label.bool() if task_type == "multilabel" else call_label
                self._update_metrics(active_metrics, metric_targets, prediction)
                if auc is not None:
                    score = nn.functional.softmax(output, dim=1)[:, 1]
                    auc.add(score.detach(), call_label)
                if confusion is not None:
                    confusion.add(prediction, call_label)

            if i == 0 and task_type != "multilabel":
                self._write_summaries(
                    features=features,
                    labels=call_label,
                    prediction=prediction,
                    file_names=label["file_name"],
                    epoch=epoch,
                    phase="train",
                )
            start_data_loading = time.time()

        self._write_scalar_summaries_logs(
            loss=epoch_loss.get(),
            metrics=active_metrics,
            lr=optimizer.param_groups[0]["lr"],
            epoch_time=time.time() - epoch_start,
            data_loading_time=data_loading_time.get(),
            epoch=epoch,
            phase="train",
        )

        if call_label is not None and task_type != "multilabel":
            self._write_classification_summaries(auc, confusion, train_loader, epoch, "train")

        if self.writer is not None:
            self.writer.flush()
        return epoch_loss.get()

    def _test_epoch(
        self,
        epoch: int,
        test_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        metrics: dict[str, MetricBase] | list[MetricBase],
        device: torch.device,
        phase: str = "val",
        checkpoint: Callable[[], None] | None = None,
    ) -> float:
        checkpoint = checkpoint or (lambda: None)
        self.logger.debug("{}|{}|start".format(phase, epoch))
        self.model.eval()

        with torch.no_grad():
            epoch_start = time.time()
            start_data_loading = epoch_start
            data_loading_time = Sum(torch.device("cpu"))
            epoch_loss = Mean(device)

            num_classes = test_loader.dataset.num_classes  # type: ignore[union-attr]
            task_type = self._resolve_loader_task(test_loader, phase)
            self._validate_loss_fn(task_type, loss_fn)
            active_metrics = self._metrics_for_task(metrics, task_type)
            self._reset_metrics(active_metrics, device)
            if task_type == "binary":
                auc: AUCMeter | None = AUCMeter()
                confusion: ConfusionMeter | None = None
            elif task_type == "multiclass":
                auc = None
                confusion = ConfusionMeter(n_categories=num_classes)
            else:
                auc = None
                confusion = None

            for i, (features, label) in enumerate(test_loader):
                checkpoint()
                features = features.to(device)
                raw_label: torch.Tensor | None = label.get("call")
                call_label = self._task_targets(raw_label, device, task_type)

                data_loading_time.update(torch.Tensor([(time.time() - start_data_loading)]))

                output = self.model(features)
                if raw_label is not None and task_type != "multilabel":
                    validate_task_batch(output, raw_label, self.task_config, num_classes)
                if call_label is not None:
                    validate_task_batch(output, call_label, self.task_config, num_classes)
                loss = loss_fn(output, call_label)
                epoch_loss.update(loss)

                prediction: torch.Tensor | None = None
                if call_label is not None:
                    prediction = self._task_predictions(
                        output, task_type, self.multilabel_thresholds
                    )
                    metric_targets = call_label.bool() if task_type == "multilabel" else call_label
                    self._update_metrics(active_metrics, metric_targets, prediction)
                    if auc is not None:
                        score = nn.functional.softmax(output, dim=1)[:, 1]
                        auc.add(score, call_label)
                    if confusion is not None:
                        confusion.add(prediction, call_label)

                if i == 0 and task_type != "multilabel":
                    self._write_summaries(
                        features=features,
                        labels=call_label,
                        prediction=prediction,
                        file_names=label["file_name"],
                        epoch=epoch,
                        phase=phase,
                    )
                start_data_loading = time.time()

        self._write_scalar_summaries_logs(
            loss=epoch_loss.get(),
            metrics=active_metrics,
            epoch_time=time.time() - epoch_start,
            data_loading_time=data_loading_time.get(),
            epoch=epoch,
            phase=phase,
        )

        if call_label is not None and task_type != "multilabel":
            self._write_classification_summaries(auc, confusion, test_loader, epoch, phase)

        if self.writer is not None:
            self.writer.flush()
        return epoch_loss.get()

    def _reset_metrics(
        self,
        metrics: dict[str, MetricBase] | list[MetricBase],
        device: torch.device,
    ) -> None:
        if isinstance(metrics, list):
            for metric in metrics:
                metric.reset(device)
        else:
            for metric in metrics.values():
                metric.reset(device)

    def _update_metrics(
        self,
        metrics: dict[str, MetricBase] | list[MetricBase],
        labels: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        if isinstance(metrics, list):
            for metric in metrics:
                metric.update(labels, predictions)
        else:
            for metric in metrics.values():
                metric.update(labels, predictions)

    def _write_summaries(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        prediction: torch.Tensor | None = None,
        file_names: list[str] | None = None,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        if self.writer is None:
            return
        prepare_img = importlib.import_module(
            "bioacoustics_dl_toolbox.visualization.image"
        ).prepare_img
        with torch.no_grad():
            if file_names is not None:
                if isinstance(file_names, torch.Tensor):
                    file_names = file_names.cpu().numpy()
                elif isinstance(file_names, list):
                    file_names = np.asarray(file_names)

            if labels is not None and prediction is not None:
                features = features.cpu()
                labels = labels.cpu()
                prediction = prediction.cpu()
                matches = torch.eq(prediction, labels)
                for idx in range(len(matches)):
                    if matches[idx]:
                        name_tag = "true - {} as {}".format(
                            self._get_class_name_from_index(labels[idx].item()),
                            self._get_class_name_from_index(prediction[idx].item()),
                        )
                    else:
                        name_tag = "false - {} as {}".format(
                            self._get_class_name_from_index(labels[idx].item()),
                            self._get_class_name_from_index(prediction[idx].item()),
                        )
                    try:
                        self.writer.add_image(
                            tag=phase + "/" + name_tag,
                            img_tensor=prepare_img(
                                features[idx].unsqueeze(dim=0),
                                num_images=self.n_summaries,
                                file_names=[file_names[idx]] if file_names is not None else None,
                            ),
                            global_step=epoch,
                        )
                    except ValueError:
                        pass
            else:
                self.writer.add_image(
                    tag=phase + "/input",
                    img_tensor=prepare_img(
                        features, num_images=self.n_summaries, file_names=file_names
                    ),
                    global_step=epoch,
                )

    def _write_scalar_summaries_logs(
        self,
        loss: float,
        metrics: dict[str, MetricBase] | list[MetricBase] = {},
        lr: float | None = None,
        epoch_time: float | None = None,
        data_loading_time: float | None = None,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        with torch.no_grad():
            log_str = phase
            if epoch is not None:
                log_str += "|{}".format(epoch)

            if self.writer is not None:
                self.writer.add_scalar(phase + "/epoch_loss", loss, epoch)
            log_str += "|loss:{:0.3f}".format(loss)

            if isinstance(metrics, dict):
                for name, metric in metrics.items():
                    if self.writer is not None:
                        self.writer.add_scalar(phase + "/" + name, metric.get(), epoch)
                    log_str += "|{}:{:0.3f}".format(name, metric.get())
            else:
                for i, metric in enumerate(metrics):
                    if self.writer is not None:
                        self.writer.add_scalar(phase + "/metric_" + str(i), metric.get(), epoch)
                    log_str += "|m_{}:{:0.3f}".format(i, metric.get())

            if lr is not None:
                if self.writer is not None:
                    self.writer.add_scalar("lr", lr, epoch)
                log_str += "|lr:{:0.2e}".format(lr)

            if epoch_time is not None:
                if self.writer is not None:
                    self.writer.add_scalar(phase + "/time", epoch_time, epoch)
                log_str += "|t:{:0.1f}".format(epoch_time)

            if data_loading_time is not None and self.writer is not None:
                self.writer.add_scalar(phase + "/data_loading_time", data_loading_time, epoch)

            self.logger.info(log_str)

    def _write_classification_summaries(
        self,
        auc: AUCMeter | None,
        confusion: ConfusionMeter | None,
        loader: torch.utils.data.DataLoader,
        epoch: int,
        phase: str,
    ) -> None:
        if self.writer is None:
            return
        plotting = importlib.import_module("bioacoustics_dl_toolbox.visualization.plotting")
        with torch.no_grad():
            if auc is not None:
                auc_val, tpr, fpr = auc.value()
                phase_prefix = phase + "_" if phase else ""
                fig = plotting.roc_fig(tpr, fpr, auc_val)
                self.writer.add_figure(phase_prefix + "roc/roc", fig, epoch)

            if confusion is not None:
                confusion_matrix_raw = confusion.confusion.clone()
                confusion_matrix_norm = confusion.value()
                label_str = [
                    loader.dataset.get_class_name_from_index(i)  # type: ignore[union-attr]
                    for i in range(confusion_matrix_norm.shape[0])
                ]
                phase_prefix = phase + "_" if phase else ""

                fig_norm = plotting.confusion_matrix_fig(
                    confusion_matrix_norm, label_str=label_str, numbering=True
                )
                self.writer.add_figure(
                    phase_prefix + "confusion_matrix_norm/cm_numbered",
                    fig_norm,
                    epoch,
                )

                fig_norm_no_num = plotting.confusion_matrix_fig(
                    confusion_matrix_norm, label_str=label_str, numbering=False
                )
                self.writer.add_figure(
                    phase_prefix + "confusion_matrix_norm/cm",
                    fig_norm_no_num,
                    epoch,
                )

                fig_raw = plotting.confusion_matrix_fig(
                    confusion_matrix_raw, label_str=label_str, numbering=True
                )
                self.writer.add_figure(
                    phase_prefix + "confusion_matrix_raw/cm_numbered",
                    fig_raw,
                    epoch,
                )

    def _get_class_name_from_index(self, idx: int) -> str:
        if self.class_dist_dict is None:
            return str(idx)
        for name, index in self.class_dist_dict.items():
            if index == idx:
                return name
        raise ValueError("Unknown class type for index {}".format(idx))
