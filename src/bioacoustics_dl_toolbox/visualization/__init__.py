"""Visualization namespace with lazy legacy plotting dependencies."""

from __future__ import annotations

from importlib import import_module

# Preserve existing public imports without loading optional visualization
# extras when importing the NumPy-only spectrogram submodule.
__all__ = [
    "apply_cm",
    "confusion_matrix_fig",
    "flip",
    "plasma_cm",
    "plot_spectrogram",
    "prepare_img",
    "roc_fig",
    "spec2img",
    "viridis_cm",
]

_LEGACY_MODULES = {
    "apply_cm": "colormaps",
    "plasma_cm": "colormaps",
    "viridis_cm": "colormaps",
    "flip": "image",
    "prepare_img": "image",
    "spec2img": "image",
    "confusion_matrix_fig": "plotting",
    "plot_spectrogram": "plotting",
    "roc_fig": "plotting",
}


def __getattr__(name: str) -> object:
    if name not in _LEGACY_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_LEGACY_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
