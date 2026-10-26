"""Headless, task-aware sliding-window inference."""

from bioacoustics_dl_toolbox.prediction.predictor import WindowPrediction, predict_windows
from bioacoustics_dl_toolbox.prediction.raven import (
    RAVEN_COLUMNS,
    predictions_to_raven_rows,
    write_raven_tsv,
)

__all__ = [
    "RAVEN_COLUMNS",
    "WindowPrediction",
    "predict_windows",
    "predictions_to_raven_rows",
    "write_raven_tsv",
]
