"""
Dataset classes for audio classification.

Ported from ANIMAL-SPOT ``data/audiodataset.py`` (Bergler & Schroeter, GPL-3.0).
Key changes from the original:
- Label extraction is pluggable via ``LabelExtractor`` protocol.
- Configuration is done via typed dataclasses, not raw dicts.
- Path handling uses ``pathlib`` instead of platform-branching string splits.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import pathlib
import random
from collections import defaultdict
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping
from functools import partial
from math import floor, fsum, isclose, isfinite
from typing import Any, Callable, Dict, Iterable, List, Protocol

import soundfile as sf
import torch
import torch.multiprocessing as mp
import torch.utils.data

from bioacoustics_dl_toolbox.audio import io as audio_io
from bioacoustics_dl_toolbox.audio import transforms as T
from bioacoustics_dl_toolbox.audio.preprocessing import (
    build_feature_preprocessor,
    spectrogram_frames_for_samples,
)
from bioacoustics_dl_toolbox.config import (
    AugmentationConfig,
    NormalizationConfig,
    SpectrogramConfig,
    TaskConfig,
    resolve_task_type,
)
from bioacoustics_dl_toolbox.io.async_file import AsyncFileReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label extraction protocol
# ---------------------------------------------------------------------------


class LabelExtractor(Protocol):
    """Protocol for extracting a class label from a file path."""

    def __call__(self, file_path: str) -> str: ...


class FilenamePrefixLabelExtractor:
    """Extract the class name from the filename prefix before the first ``-``.

    This matches the ANIMAL-SPOT naming convention where files are named
    ``<class>-<info>_ID_YEAR_TAPE_START_END.wav``, e.g.
    ``target-whistles_001_2020_TAPE01_0000_1000.wav`` → ``"target"``.
    """

    def __call__(self, file_path: str) -> str:
        basename = pathlib.Path(file_path).name
        return basename.split("-", 1)[0]


# ---------------------------------------------------------------------------
# File discovery and filtering
# ---------------------------------------------------------------------------


def get_audio_files_from_dir(path: str) -> Iterable[str]:
    """Recursively find all ``.wav`` files under ``path``, excluding ``*.bkp/`` dirs."""
    audio_files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
    audio_paths = map(lambda p: pathlib.Path(p), audio_files)
    audio_paths = filter(lambda p: not p.match("*.bkp/*"), audio_paths)
    base = pathlib.Path(path)
    return map(lambda p: str(p.relative_to(base)), audio_paths)


class _FilterPickleHelper:
    def __init__(self, predicate: Callable[..., Any], *pred_args: Any) -> None:
        self.predicate = predicate
        self.args = pred_args

    def __call__(self, item: Any) -> Any:
        return self.predicate(item, *self.args)


class _ParallelFilter:
    def __init__(
        self,
        iterable: Iterable[Any],
        n_threads: int | None = None,
        chunk_size: int = 1,
    ) -> None:
        self.data = iterable
        self.n_threads = n_threads
        self.chunk_size = chunk_size

    def __call__(self, func: Callable[..., Any], *func_args: Any) -> Iterable[Any]:
        with mp.Pool(self.n_threads) as pool:
            func_pickle = _FilterPickleHelper(func, *func_args)
            for keep, candidate in pool.imap_unordered(func_pickle, self.data, self.chunk_size):
                if keep:
                    yield candidate


def _loudness_criteria(file_name: str, working_dir: str | None = None) -> tuple[bool, str | None]:
    if working_dir is not None:
        file_path = os.path.join(working_dir, file_name)
    else:
        file_path = file_name
    audio_data, _ = sf.read(file_path, always_2d=True, dtype="float32")
    max_amplitude = audio_data.max()
    if max_amplitude < 1e-3:
        return True, file_name
    return False, None


def get_broken_audio_files(files: Iterable[str], working_dir: str | None = None) -> Iterable[Any]:
    """Find audio files that are below the minimum loudness threshold (1e-3)."""
    parallel_filter = _ParallelFilter(files, chunk_size=100)
    return parallel_filter(_loudness_criteria, working_dir)


# ---------------------------------------------------------------------------
# CSV Split
# ---------------------------------------------------------------------------


class CsvSplit:
    """Split file identifiers and optionally reuse partitions saved as one-column CSVs.

    The file identifiers (including any relative directory components) are saved
    verbatim. A positive-fraction partition must contain at least one entry when
    persisting to CSV; zero-fraction partitions are allowed to be empty.
    """

    def __init__(
        self,
        split_fracs: Dict[str, float],
        working_dir: str | None = None,
        seed: int | None = None,
        split_per_dir: bool = False,
    ) -> None:
        if not isinstance(split_fracs, Mapping) or not split_fracs:
            raise ValueError("`split_fracs` must be a nonempty mapping.")
        # Materialize the mapping so callers cannot change the splits after creation.
        fractions: dict[str, float] = {}
        for name, fraction in split_fracs.items():
            if (
                not isinstance(name, str)
                or not name
                or name in fractions
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
            ):
                raise ValueError("Split names must be nonempty, unique file-safe strings.")
            try:
                value = float(fraction)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid fraction for split '{name}': {fraction!r}") from exc
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid fraction for split '{name}': {fraction!r}")
            fractions[name] = value
        if not isclose(fsum(fractions.values()), 1.0, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Split probabilities have to sum up to 1.")

        self.split_fracs = fractions
        self.working_dir = working_dir
        self.seed = seed
        self.split_per_dir = split_per_dir
        self.splits: dict[str, list[Any]] = {name: [] for name in fractions}
        self._rng = random.Random(seed)
        self._loaded = False  # Empty (zero-fraction) splits are cached too.

    @staticmethod
    def _validate_files(files: Iterable[Any] | None) -> list[Any]:
        if files is None:
            raise ValueError("A nonempty list of `files` is required to generate splits.")
        items = list(files)
        if not items:
            raise ValueError("A nonempty list of `files` is required to generate splits.")
        try:
            if len(set(items)) != len(items):
                raise ValueError("Duplicate input file identifiers are not allowed.")
        except TypeError as exc:
            raise ValueError("Input file identifiers must be hashable and unique.") from exc
        return items

    def _csv_files_for_split(self, split: str) -> list[pathlib.Path]:
        root = pathlib.Path(self.working_dir)  # type: ignore[arg-type]
        if not self.split_per_dir:
            return [root / f"{split}.csv"]
        manifest = root / split
        csv_paths = []
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                path = pathlib.Path(entry)
                # Accept both the usual manifest-relative paths and legacy
                # entries already prefixed with a relative working directory.
                if not path.is_absolute() and not path.is_relative_to(root):
                    path = root / path
                csv_paths.append(path)
        return csv_paths

    def _read_csv_splits(self) -> dict[str, list[str]]:
        root = pathlib.Path(self.working_dir)  # type: ignore[arg-type]
        splits: dict[str, list[str]] = {name: [] for name in self.split_fracs}
        seen: set[str] = set()
        for split in self.split_fracs:
            for csv_path in self._csv_files_for_split(split):
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    for row_number, row in enumerate(csv.reader(handle), start=1):
                        if len(row) != 1 or not row[0].strip():
                            raise ValueError(
                                f"Invalid CSV row {row_number} in '{csv_path}': "
                                "expected exactly one nonempty file identifier."
                            )
                        identifier = row[0]
                        # Earlier per-directory CSVs sometimes contain just a
                        # basename. Resolve those against the directory holding
                        # the CSV; full relative paths are kept verbatim.
                        if self.split_per_dir and pathlib.Path(identifier).parent == pathlib.Path(
                            "."
                        ):
                            parent = csv_path.parent.resolve().relative_to(root.resolve())
                            identifier = str(parent / identifier)
                        if identifier in seen:
                            raise ValueError(
                                f"Duplicate file identifier in saved CSVs: {identifier!r}"
                            )
                        seen.add(identifier)
                        splits[split].append(identifier)
        for split, fraction in self.split_fracs.items():
            if fraction > 0 and not splits[split]:
                raise ValueError(f"Split '{split}' has 0 entries despite a positive fraction.")
        return splits

    def load(self, split: str, files: List[Any] | None = None) -> list[Any]:
        if split not in self.split_fracs:
            raise ValueError(f"Provided split '{split}' is not in `self.split_fracs`.")
        if self._loaded:
            return self.splits[split]
        if self.working_dir is not None and self.can_load_from_csv():
            self.splits = self._read_csv_splits()
            self._loaded = True
            return self.splits[split]

        items = self._validate_files(files)
        if self.working_dir is None:
            self.splits = self._split_with_seed(items)
            self._loaded = True
            return self.splits[split]

        root = pathlib.Path(self.working_dir)
        if self.split_per_dir:
            groups = self.get_file_dir_map(items)
        else:
            groups = {".": items}
        pending: dict[str, list[Any]] = {name: [] for name in self.split_fracs}
        csv_data: dict[pathlib.Path, list[Any]] = {}
        manifest_data: dict[str, list[str]] = {name: [] for name in self.split_fracs}
        for directory, group_files in groups.items():
            group_splits = self._split_with_seed(group_files)
            for name, fraction in self.split_fracs.items():
                assigned = group_splits[name]
                if fraction > 0 and not assigned:
                    raise ValueError(
                        f"Split '{name}' has 0 entries despite a positive fraction "
                        f"for {len(group_files)} files in '{directory}'."
                    )
                csv_path = root / directory / f"{name}.csv"
                csv_data[csv_path] = assigned
                pending[name].extend(assigned)
                if self.split_per_dir:
                    manifest_data[name].append(str(csv_path.relative_to(root)))

        root.mkdir(parents=True, exist_ok=True)
        for csv_path, assigned in csv_data.items():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for identifier in assigned:
                    writer.writerow([identifier])
        if self.split_per_dir:
            for name, paths in manifest_data.items():
                (root / name).write_text("\n".join(paths) + "\n", encoding="utf-8")
        self.splits = pending
        self._loaded = True
        return self.splits[split]

    def can_load_from_csv(self) -> bool:
        if self.working_dir is None:
            return False
        root = pathlib.Path(self.working_dir)
        if self.split_per_dir:
            for split in self.split_fracs:
                manifest = root / split
                if not manifest.is_file():
                    return False
                for csv_path in self._csv_files_for_split(split):
                    if not csv_path.is_file():
                        raise ValueError(f"Split file found, but CSV file '{csv_path}' is missing.")
        else:
            for split in self.split_fracs:
                if not (root / f"{split}.csv").is_file():
                    return False
        return True

    def get_file_dir_map(self, files: List[Any]) -> dict[str, list[Any]]:
        """Group file identifiers by their relative parent directory."""
        file_dir_map: dict[str, list[Any]] = defaultdict(list)
        root = pathlib.Path(self.working_dir) if self.working_dir is not None else None
        for identifier in files:
            parent = pathlib.Path(identifier).parent
            if root is not None:
                if parent.is_absolute():
                    try:
                        parent = parent.relative_to(root.resolve())
                    except ValueError as exc:
                        raise ValueError(
                            "Per-directory file identifiers must be inside `working_dir`."
                        ) from exc
                elif parent.is_relative_to(root):
                    parent = parent.relative_to(root)
                if not (root / parent).resolve().is_relative_to(root.resolve()):
                    raise ValueError("Per-directory file identifiers must be inside `working_dir`.")
            file_dir_map[str(parent)].append(identifier)
        return dict(file_dir_map)

    def _split_with_seed(self, files: List[Any] | None) -> dict[str, list[Any]]:
        """Legacy entry point for a split using this instance's local RNG."""
        return self.split_fn(self._validate_files(files))

    def split_fn(self, files: List[Any]) -> dict[str, list[Any]]:
        items = self._validate_files(files)
        shuffled = self._rng.sample(items, len(items))
        # k-1 cumulative cuts; the final partition receives the remainder.
        fractions = list(self.split_fracs.values())
        cuts = []
        for count in range(1, len(fractions)):
            boundary = len(items) * fsum(fractions[:count])
            # Avoid losing an item to binary floating-point error when a
            # mathematically integral boundary is represented as 57.9999999...
            rounded = round(boundary)
            cut = (
                rounded
                if isclose(boundary, rounded, rel_tol=0.0, abs_tol=1e-10)
                else floor(boundary)
            )
            cuts.append(min(len(items), cut))
        boundaries = [0, *cuts, len(items)]
        return {
            name: shuffled[boundaries[i] : boundaries[i + 1]]
            for i, name in enumerate(self.split_fracs)
        }


def get_tape_key(file: str, valid_years: set[int] | None = None) -> str | None:
    """Extract year and tape from ANIMAL-SPOT filename convention."""
    while "__" in file:
        file = file.replace("__", "_")
    try:
        attributes = file.split(sep="_")
        year = attributes[-4]
        tape = attributes[-3]
        if valid_years is not None and int(year) not in valid_years:
            return None
        return year + "_" + tape.upper()
    except Exception:
        import traceback

        print("Warning: skipping file {}\n{}".format(file, traceback.format_exc()))
    return None


class DatabaseCsvSplit(CsvSplit):
    """Split files by tape, ensuring all clips from one tape stay in the same partition."""

    valid_years: set[int] = set(range(1950, 2200))

    def split_fn(self, files: Iterable[Any]) -> dict[str, list[Any]]:
        files = self._validate_files(files)
        n_files = len(files)
        tapes: dict[str, int] = defaultdict(int)
        for file in files:
            try:
                key = get_tape_key(file, self.valid_years)
                if key is not None:
                    tapes[key] += 1
                else:
                    n_files -= 1
            except IndexError:
                n_files -= 1

        tape_names = list(tapes)

        class _Mapping:
            def __init__(self) -> None:
                self.count = 0
                self.names: list[str] = []

            def add(self, name: str, count: int) -> None:
                self.count += count
                self.names.append(name)

        mappings = {s: _Mapping() for s in self.split_fracs.keys()}
        for tape_name in tape_names:
            missing_files = {
                s: n_files * f - mappings[s].count for s, f in self.split_fracs.items()
            }
            r = self._rng.uniform(0.0, sum(missing_files.values()))
            for _split, _n_files in missing_files.items():
                r -= _n_files
                if r < 0:
                    mappings[_split].add(tape_name, tapes[tape_name])
                    break
            assert r < 0, "Should not get here"

        splits: dict[str, list[Any]] = defaultdict(list)
        for file in files:
            tape = get_tape_key(file, self.valid_years)
            if tape is not None:
                for s, m in mappings.items():
                    if tape in m.names:
                        splits[s].append(file)
        return splits


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------


class AudioDataset(torch.utils.data.Dataset):
    """Dataset that loads raw audio waveforms."""

    def __init__(
        self,
        file_names: Iterable[str],
        working_dir: str | None = None,
        sample_rate: int = 44100,
        mono: bool = True,
        transform: Callable[..., Any] | None = None,
        dataset_name: str | None = None,
    ) -> None:
        self.file_names = list(file_names) if not isinstance(file_names, list) else file_names
        self.working_dir = working_dir
        self.sample_rate = sample_rate
        self.mono = mono
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Any:
        file = self.file_names[idx]
        if self.working_dir is not None:
            file = os.path.join(self.working_dir, file)
        sample = audio_io.load_audio_file(file, self.sample_rate, self.mono)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class SpectrogramDataset(AudioDataset):
    """Dataset that loads audio, computes spectrograms, applies augmentation.

    Parameters
    ----------
    file_names:
        Iterable of file paths (relative to ``working_dir``).
    working_dir:
        Base directory prepended to each filename.
    spec_config:
        Spectrogram configuration.
    norm_config:
        Normalization configuration.
    aug_config:
        Augmentation configuration.
    classes:
        List of class names. The class-to-index mapping is built from this.
    sequence_length:
        Target sequence length in spectrogram time-steps. Without augmentation,
        excess frames are center-cropped and missing frames are center-zero-padded
        after normalization. Match the corresponding inference window via
        ``spectrogram_frames_for_samples(window_samples, spec_config)``.
    label_extractor:
        Callable that extracts a class name string from a file path.
        Defaults to ``FilenamePrefixLabelExtractor``.
    labels:
        For binary/multiclass, optional class names aligned with ``file_names``;
        omitted labels are extracted from filename prefixes. For an explicit
        multilabel task, **required** per-item iterables of class names, e.g.
        ``[["bird", "rain"], [], ["frog", "frog"]]``. Empty annotations are valid
        and repetitions within an item count only once. A bare string is not a
        valid per-item multilabel annotation; filename parsing is single-label only.
    class_to_index:
        Explicit mapping from every class name to a unique, contiguous integer
        index. Takes precedence over automatic mapping; if ``positive_class``
        is also provided, the two settings must agree.
    positive_class:
        For binary classification only. Maps this declared class to index 1
        and the other class to 0. If omitted, binary classes named ``target``
        retain the ANIMAL-SPOT convention (``target`` = 1). Otherwise binary
        mapping follows class-list order (first = 0, second = 1).
        Multilabel and multiclass tasks always use class-list order unless an
        explicit ``class_to_index`` is provided.
    cache_dir:
        Optional directory for caching computed spectrograms.
    dataset_name:
        Name for logging.
    task_config:
        Task semantics. Defaults to legacy single-label ``auto`` behavior.
        Explicit ``multilabel`` produces float32 multi-hot vectors in mapping order.
    """

    def __init__(
        self,
        file_names: Iterable[str],
        working_dir: str | None = None,
        spec_config: SpectrogramConfig = SpectrogramConfig(),
        norm_config: NormalizationConfig = NormalizationConfig(),
        aug_config: AugmentationConfig = AugmentationConfig(),
        classes: list[str] | None = None,
        sequence_length: int = 128,
        label_extractor: LabelExtractor | None = None,
        labels: Iterable[str] | Iterable[Iterable[str]] | None = None,
        class_to_index: dict[str, int] | None = None,
        positive_class: str | None = None,
        cache_dir: str | None = None,
        dataset_name: str | None = None,
        task_config: TaskConfig = TaskConfig(),
    ) -> None:
        super().__init__(
            file_names,
            working_dir=working_dir,
            sample_rate=spec_config.sample_rate,
            dataset_name=dataset_name,
        )
        if classes is None:
            raise ValueError("classes must be provided")

        if dataset_name is not None:
            logger.info("Init dataset {}...".format(dataset_name))

        self.spec_config = spec_config
        self.norm_config = norm_config
        self.aug_config = aug_config
        if any(not isinstance(name, str) or not name.strip() for name in classes):
            raise ValueError("classes must contain nonempty class names")
        if len(set(classes)) != len(classes):
            raise ValueError("classes must not contain duplicate class declarations")
        self.classes = list(classes)
        self.num_classes = len(classes)
        self.task_config = task_config
        self.task_type = resolve_task_type(task_config, self.num_classes)
        self.label_extractor = label_extractor or FilenamePrefixLabelExtractor()
        self.labels = None if labels is None else list(labels)
        if self.task_type == "multilabel" and self.labels is None:
            raise ValueError(
                "multilabel requires explicit per-file labels; "
                "filename extraction is single-label only"
            )
        if self.labels is not None and len(self.labels) != len(self.file_names):
            raise ValueError("labels must contain one annotation per file")

        self.preprocessing = build_feature_preprocessor(spec_config, norm_config)
        if (
            not isinstance(sequence_length, int)
            or isinstance(sequence_length, bool)
            or sequence_length <= 0
        ):
            raise ValueError("sequence_length must be a positive number of feature frames")

        logger.debug("Number of files: {}".format(len(self.file_names)))

        # Validate explicit mapping first: caller-specified indices always win.
        if class_to_index is not None:
            if (
                not isinstance(class_to_index, Mapping)
                or set(class_to_index) != set(self.classes)
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in class_to_index.values()
                )
                or set(class_to_index.values()) != set(range(self.num_classes))
            ):
                raise ValueError(
                    "class_to_index must map every declared class once to distinct indices 0..C-1"
                )
        if positive_class is not None:
            if self.task_type != "binary":
                raise ValueError("positive_class is supported only for binary classification")
            if positive_class not in self.classes:
                raise ValueError(
                    f"positive_class {positive_class!r} is not among declared classes "
                    f"{self.classes!r}"
                )
            if class_to_index is not None and class_to_index[positive_class] != 1:
                raise ValueError(
                    f"positive_class {positive_class!r} conflicts with class_to_index: "
                    "the positive class must have index 1"
                )

        if class_to_index is not None:
            self.class_dist_dict = dict(class_to_index)
        elif self.task_type == "binary" and positive_class is not None:
            self.class_dist_dict = {
                name: 1 if name == positive_class else 0 for name in self.classes
            }
        elif self.task_type == "binary" and "target" in self.classes:
            # Preserve ANIMAL-SPOT's binary target=1 convention, independent of order.
            self.class_dist_dict = {name: 1 if name == "target" else 0 for name in self.classes}
        else:
            # Default mapping for other single-label tasks and *all* multilabel tasks.
            self.class_dist_dict = {name: idx for idx, name in enumerate(self.classes)}
            if self.task_type == "binary":
                logger.info(
                    "Binary classes have no 'target' or explicit polarity; using "
                    "class-list order for class_to_index: %s",
                    self.class_dist_dict,
                )

        # Validate all supplied annotations at construction time, before reading
        # audio. Store each multilabel item as a deduplicated tuple of names.
        if self.task_type == "multilabel":
            normalized: list[tuple[str, ...]] = []
            for item in self.labels:  # type: ignore[union-attr]
                if isinstance(item, (str, bytes, Mapping)) or not isinstance(item, IterableABC):
                    raise ValueError(
                        "each multilabel item must be an iterable of class names, not a string"
                    )
                names = tuple(item)
                if any(
                    not isinstance(name, str) or name not in self.class_dist_dict for name in names
                ):
                    raise ValueError(
                        f"unknown or invalid multilabel class in annotation: {names!r}"
                    )
                normalized.append(tuple(dict.fromkeys(names)))
            self.labels = normalized
        elif self.labels is not None:
            for name in self.labels:
                self._class_index_for_name(name)

        calls: dict[int, int] = defaultdict(int)
        for index, file_name in enumerate(self.file_names):
            if self.task_type == "multilabel":
                for name in self.labels[index]:  # type: ignore[index]
                    calls[self.class_dist_dict[name]] += 1
            else:
                calls[self._get_class_index(file_name, index)] += 1
        for class_index, n in calls.items():
            logger.debug(
                "Number of samples in {} for {}: {}".format(
                    dataset_name, self.get_class_name_from_index(class_index), n
                )
            )

        augmentation_enabled = aug_config.enabled

        # Cache only the raw power spectrogram (same v4 format as before).
        # Frequency compression and normalization are never cached.
        spec_transforms: list[Any] = [
            partial(audio_io.load_audio_file, sample_rate=spec_config.sample_rate),
            *self.preprocessing.waveform_to_power.transforms,
        ]

        self.file_reader = AsyncFileReader()

        if cache_dir is None:
            self.t_spectrogram = T.Compose(spec_transforms)
        else:
            self.t_spectrogram = T.CachedSpectrogram(
                cache_dir=cache_dir,
                spec_transform=T.Compose(spec_transforms),
                n_fft=spec_config.n_fft,
                hop_length=spec_config.hop_length,
                file_reader=self.file_reader,
            )

        if augmentation_enabled:
            logger.debug("Init augmentation transforms for time and pitch shift")
            self.t_amplitude = T.RandomAmplitude(
                aug_config.amplitude_increase_db, aug_config.amplitude_decrease_db
            )
            self.t_timestretch = T.RandomTimeStretch(
                aug_config.time_stretch_from, aug_config.time_stretch_to
            )
            self.t_pitchshift = T.RandomPitchShift(
                aug_config.pitch_shift_from, aug_config.pitch_shift_to
            )
        else:
            logger.debug("Running without augmentation")

        self.t_compr_f = self.preprocessing.frequency_compression

        if augmentation_enabled:
            if aug_config.noise_files:
                logger.debug("Init augmentation transform for random noise addition")
                self.t_addnoise: T.RandomAddNoise | None = T.RandomAddNoise(
                    aug_config.noise_files,
                    self.t_spectrogram,
                    T.Compose(self.t_timestretch, self.t_pitchshift, self.t_compr_f),
                    min_length=sequence_length,
                    return_original=True,
                    min_snr=aug_config.noise_min_snr,
                    max_snr=aug_config.noise_max_snr,
                )
            else:
                self.t_addnoise = None
                logger.debug("No noise augmentation")
        else:
            self.t_addnoise = None

        self.t_subseq = T.PaddedSubsequenceSampler(
            sequence_length, dim=1, random=augmentation_enabled
        )
        self.augmentation_enabled = augmentation_enabled

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        file_name = self.file_names[idx]
        if self.working_dir is not None:
            file = os.path.join(self.working_dir, file_name)
        else:
            file = file_name

        sample = self.t_spectrogram(file)

        if self.augmentation_enabled:
            sample = self.t_amplitude(sample)
            sample = self.t_pitchshift(sample)
            sample = self.t_timestretch(sample)

        # ANIMAL-SPOT augmentation occurs on power before representation.
        # The same shared post-compression conversion is used by prediction.
        sample = self.t_compr_f(sample)

        ground_truth: torch.Tensor | None = None
        if self.augmentation_enabled and self.t_addnoise is not None:
            result = self.t_addnoise(sample)
            sample, ground_truth = result  # type: ignore[misc]
            ground_truth = self.preprocessing.to_features(ground_truth)

        sample = self.preprocessing.to_features(sample)

        if ground_truth is not None:
            stacked = torch.cat((sample, ground_truth), dim=0)
            stacked = self.t_subseq(stacked)
            sample = stacked[0].unsqueeze(0)
            ground_truth = stacked[1].unsqueeze(0)
        else:
            sample = self.t_subseq(sample)

        label = self._load_label(file, idx)
        if ground_truth is not None:
            label["ground_truth"] = ground_truth
        return sample, label

    def _load_label(self, file_name: str, index: int) -> dict[str, Any]:
        if self.task_type == "multilabel":
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            for name in self.labels[index]:  # type: ignore[index]
                multi_hot[self.class_dist_dict[name]] = 1.0
            return {"file_name": file_name, "call": multi_hot}
        return {
            "file_name": file_name,
            "call": self._get_class_index(file_name, index),
        }

    def _class_index_for_name(self, name: Any) -> int:
        if not isinstance(name, str) or name not in self.class_dist_dict:
            raise ValueError(
                f"unknown single-label class: {name!r}; declared classes: {self.classes!r}"
            )
        return self.class_dist_dict[name]

    def _get_class_index(self, file_name: str, index: int) -> int:
        if self.task_type == "multilabel":
            raise ValueError("class indices are only available for single-label tasks")
        class_name = (
            self.labels[index] if self.labels is not None else self.label_extractor(file_name)
        )
        return self._class_index_for_name(class_name)

    def get_class_name_from_index(self, idx: int) -> str:
        for name, index in self.class_dist_dict.items():
            if index == idx:
                return name
        raise ValueError("Unknown class type for index {}".format(idx))


class StridedAudioDataset(torch.utils.data.Dataset):
    """Process an audio tape via a sliding window for inference.

    Parameters
    ----------
    file_name:
        Path to the audio file.
    sequence_len:
        Window length in audio samples. The corresponding fixed-length
        feature frames use ``center=False`` FFT geometry; incomplete tail
        windows receive centered zero-padding *after* normalization.
    hop:
        Hop length in samples between consecutive windows.
    spec_config:
        Spectrogram configuration.
    norm_config:
        Normalization configuration.
    """

    def __init__(
        self,
        file_name: str,
        sequence_len: int,
        hop: int,
        spec_config: SpectrogramConfig = SpectrogramConfig(),
        norm_config: NormalizationConfig = NormalizationConfig(),
    ) -> None:
        if not isinstance(sequence_len, int) or isinstance(sequence_len, bool) or sequence_len <= 0:
            raise ValueError("sequence_len must be a positive number of audio samples")
        if not isinstance(hop, int) or isinstance(hop, bool) or hop <= 0:
            raise ValueError("hop must be a positive number of audio samples")
        if hop > sequence_len:
            raise ValueError("hop cannot exceed sequence_len (gapped scan)")
        self.preprocessing = build_feature_preprocessor(spec_config, norm_config)
        self.sequence_len = sequence_len
        self.hop = hop
        try:
            self.audio = audio_io.load_audio_file(
                file_name, sample_rate=spec_config.sample_rate, mono=True
            )
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Unable to read audio file {file_name!r}: {exc}") from exc
        if self.audio.ndim != 2 or self.audio.shape[0] != 1 or self.audio.shape[1] == 0:
            raise ValueError(f"Audio file {file_name!r} must contain nonempty mono audio")
        if not bool(torch.isfinite(self.audio).all()):
            raise ValueError(f"Audio file {file_name!r} contains non-finite samples")
        self.n_frames = self.audio.shape[1]
        self.transform_pipeline = self.preprocessing
        self.t_subseq = T.PaddedSubsequenceSampler(
            spectrogram_frames_for_samples(sequence_len, spec_config), dim=1, random=False
        )

    def __len__(self) -> int:
        # Keep the first tail start whose window reaches the final sample.
        remaining = max(self.n_frames - self.sequence_len, 0)
        return 1 + (remaining + self.hop - 1) // self.hop

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not isinstance(idx, int) or idx < 0 or idx >= len(self):
            raise IndexError(idx)
        start = idx * self.hop
        end = min(start + self.sequence_len, self.n_frames)
        y = self.audio[:, start:end]
        return self.t_subseq(self.transform_pipeline(y))
