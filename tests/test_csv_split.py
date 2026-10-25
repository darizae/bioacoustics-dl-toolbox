"""Unit tests for deterministic CSV partitioning and tape-aware splitting."""

import csv
import random

import pytest

from bioacoustics_dl_toolbox.audio.datasets import CsvSplit, DatabaseCsvSplit


FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def test_fractional_boundaries_allocate_every_item_once():
    files = [f"call_{i}.wav" for i in range(100)]
    splits = CsvSplit(FRACTIONS, seed=42).split_fn(files)

    assert list(splits) == ["train", "val", "test"]
    assert [len(items) for items in splits.values()] == [70, 15, 15]
    assert set().union(*(set(items) for items in splits.values())) == set(files)
    assert sum(map(len, splits.values())) == len(files)


def test_small_dataset_boundaries_do_not_drop_items(tmp_path):
    files = ["one.wav", "two.wav"]
    split = CsvSplit(FRACTIONS, seed=1)
    assignments = split.load("train", files)

    assert len(assignments) + len(split.load("val")) + len(split.load("test")) == 2
    assert set().union(*(set(items) for items in split.splits.values())) == set(files)

    disk_split = CsvSplit(FRACTIONS, working_dir=str(tmp_path), seed=1)
    with pytest.raises(ValueError, match="positive fraction"):
        disk_split.load("train", files)
    assert not (tmp_path / "train.csv").exists()


def test_seed_zero_is_reproducible():
    files = [f"file_{i}.wav" for i in range(31)]
    left = CsvSplit(FRACTIONS, seed=0).load("train", files)
    right = CsvSplit(FRACTIONS, seed=0).load("train", files)

    assert left == right
    assert CsvSplit(FRACTIONS, seed=0).split_fn(files) == CsvSplit(
        FRACTIONS, seed=0
    ).split_fn(files)


def test_global_random_state_unchanged():
    files = [f"clip_{i}.wav" for i in range(30)]
    random.seed(111)
    before = random.getstate()

    CsvSplit(FRACTIONS, seed=0).load("train", files)
    DatabaseCsvSplit({"train": 1.0}, seed=0).load(
        "train", ["target-X_002_2020_TAPE01_0000_1000.wav"]
    )

    assert random.getstate() == before


@pytest.mark.parametrize(
    "fractions",
    [
        {},
        {"train": float("nan"), "test": 1.0},
        {"train": float("inf"), "test": 0.0},
        {"train": -0.1, "test": 1.1},
        {"train": 0.4, "test": 0.4},
        {"train": 1.01, "test": -0.01},
        {"train": "not-a-number"},
    ],
)
def test_invalid_fraction_inputs_raise(fractions):
    with pytest.raises(ValueError):
        CsvSplit(fractions)


def test_duplicate_split_names_raise():
    from collections.abc import Mapping

    class DuplicateNames(Mapping):
        def __getitem__(self, key):
            return 0.5

        def __iter__(self):
            return iter(("train", "train"))

        def __len__(self):
            return 2

        def items(self):
            return [("train", 0.5), ("train", 0.5)]

    with pytest.raises(ValueError, match="unique"):
        CsvSplit(DuplicateNames())


def test_duplicate_input_file_identifiers_raise():
    with pytest.raises(ValueError, match="Duplicate"):
        CsvSplit(FRACTIONS).load("train", ["same.wav", "same.wav"])
    with pytest.raises(ValueError, match="Duplicate"):
        CsvSplit(FRACTIONS).split_fn(["same.wav", "same.wav"])


def test_relative_workdir_csv_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = [f"site_{i % 3}/session_{i % 5}/recording_{i:03}.wav" for i in range(20)]
    working_dir = "partitions"
    writer = CsvSplit({"train": 0.6, "val": 0.4}, working_dir=working_dir, seed=0)
    first = {name: writer.load(name, files)[:] for name in writer.split_fracs}

    for name, expected in first.items():
        with (tmp_path / working_dir / f"{name}.csv").open(newline="") as handle:
            assert [row[0] for row in csv.reader(handle)] == expected

    reader = CsvSplit({"train": 0.6, "val": 0.4}, working_dir=working_dir, seed=999)
    assert {name: reader.load(name) for name in reader.split_fracs} == first
    assert reader.load("train", ["ignored.wav"]) == first["train"]
    assert not (tmp_path / working_dir / working_dir).exists()


def test_relative_workdir_per_directory_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = [f"region/site_{i // 6}/clip_{i}.wav" for i in range(12)]
    fracs = {"train": 0.5, "val": 0.5}
    writer = CsvSplit(fracs, working_dir="splits", split_per_dir=True, seed=0)
    expected = {name: writer.load(name, files)[:] for name in fracs}

    for name in fracs:
        manifest = (tmp_path / "splits" / name).read_text().splitlines()
        assert manifest == [f"region/site_{i}/{name}.csv" for i in range(2)]
    assert CsvSplit(fracs, working_dir="splits", split_per_dir=True).load(
        "train"
    ) == expected["train"]
    reloaded = CsvSplit(fracs, working_dir="splits", split_per_dir=True)
    assert {name: reloaded.load(name) for name in fracs} == expected


def test_legacy_per_dir_csv_with_basename_reloads(tmp_path):
    folder = tmp_path / "site" / "session"
    folder.mkdir(parents=True)
    (tmp_path / "train").write_text("site/session/train.csv\n")
    (folder / "train.csv").write_text("bird.wav\n")

    assert CsvSplit({"train": 1.0}, working_dir=str(tmp_path), split_per_dir=True).load(
        "train"
    ) == ["site/session/bird.wav"]


def test_manifest_prefixed_with_relative_workdir_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_dir = tmp_path / "splits" / "site"
    csv_dir.mkdir(parents=True)
    (tmp_path / "splits" / "train").write_text("splits/site/train.csv\n")
    (csv_dir / "train.csv").write_text("site/bird.wav\n")

    assert CsvSplit({"train": 1.0}, working_dir="splits", split_per_dir=True).load(
        "train"
    ) == ["site/bird.wav"]


def test_invalid_csv_rows_raise(tmp_path):
    (tmp_path / "train.csv").write_text("\n")
    with pytest.raises(ValueError, match="Invalid CSV row"):
        CsvSplit({"train": 1.0}, working_dir=str(tmp_path)).load("train")

    (tmp_path / "train.csv").write_text("file.wav,extra-column\n")
    with pytest.raises(ValueError, match="Invalid CSV row"):
        CsvSplit({"train": 1.0}, working_dir=str(tmp_path)).load("train")


def test_duplicate_saved_csv_identifiers_raise(tmp_path):
    (tmp_path / "train.csv").write_text("a.wav\n")
    (tmp_path / "val.csv").write_text("a.wav\n")
    with pytest.raises(ValueError, match="Duplicate file identifier"):
        CsvSplit({"train": 0.5, "val": 0.5}, working_dir=str(tmp_path)).load("train")


def test_positive_empty_per_directory_partition_fails_before_writing(tmp_path):
    splitter = CsvSplit(
        {"train": 0.5, "val": 0.5}, working_dir=str(tmp_path), split_per_dir=True
    )
    with pytest.raises(ValueError, match="positive fraction"):
        splitter.load("train", ["folder/a.wav"])
    assert not (tmp_path / "folder" / "train.csv").exists()
    assert not (tmp_path / "train").exists()


def test_zero_fraction_split_can_be_empty_and_cached(tmp_path):
    fracs = {"train": 1.0, "val": 0.0}
    split = CsvSplit(fracs, working_dir=str(tmp_path), seed=0)
    assert split.load("val", ["a.wav", "b.wav"]) == []
    first = split.load("train")[:]
    assert split.load("val", ["different.wav"]) == []
    assert split.load("train") == first

    reloaded = CsvSplit(fracs, working_dir=str(tmp_path))
    assert reloaded.load("val") == []
    assert reloaded.load("train") == first


def test_tape_split_unchanged():
    files = [
        f"target-call_{i:03}_2020_TAPE{tape:02}_0000_1000.wav"
        for tape in range(1, 7)
        for i in range(3)
    ]
    fracs = {"train": 0.5, "val": 0.5}
    db = DatabaseCsvSplit(fracs, seed=0)
    assert db.load("train", files)  # At least one tape is assigned to training.
    tape_splits = {}
    for partition, names in db.splits.items():
        for filename in names:
            tape = filename.split("_")[-3]
            assert tape not in tape_splits or tape_splits[tape] == partition
            tape_splits[tape] = partition
    assert set().union(*(set(v) for v in db.splits.values())) == set(files)
    assert db.splits == DatabaseCsvSplit(fracs, seed=0).split_fn(files)
