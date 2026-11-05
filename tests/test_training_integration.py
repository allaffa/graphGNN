"""Opt-in training smoke test using the published training dataset."""

import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("GRAINGNN_RUN_TRAINING_TEST") != "1",
    reason="set GRAINGNN_RUN_TRAINING_TEST=1 to enable the dataset training test",
)
def test_training_completes_with_cached_dataset(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    cache_dir = Path(os.environ.get("GRAINGNN_DATA_DIR", Path.home() / ".cache/graingnn"))
    dataset = _training_dataset(cache_dir)

    work_dir = tmp_path / "training-run"
    work_dir.mkdir()
    (work_dir / "dataset_train.pkl").symlink_to(dataset.resolve())
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    subprocess.run(
        [
            sys.executable,
            str(repository / "train.py"),
            "--dataset", "train",
            "--device", "cpu",
            "--model_dir", str(model_dir) + os.sep,
            "--model_type", "regressor",
            "--model_id", "0",
            "--train_ratio", "0.5",
            "--max_samples", "2",
            "--epochs", "1",
        ],
        cwd=work_dir,
        check=True,
        timeout=600,
    )

    assert (model_dir / "regressor0.pt").is_file()
    assert (work_dir / "loss.png").is_file()


def test_training_dataset_downloads_once_then_reuses_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace

    download_calls = []

    def fake_download(*, id, output, quiet):
        download_calls.append(id)
        with gzip.open(output, "wb") as compressed:
            compressed.write(b"downloaded training data")
        return output

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=fake_download))
    dataset = _training_dataset(tmp_path)
    assert dataset.read_bytes() == b"downloaded training data"
    assert _training_dataset(tmp_path) == dataset
    assert download_calls == [DATASET_ID]


def test_training_dataset_resolves_relative_cache_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache_dir = Path("relative-cache")
    cache_dir.mkdir()
    dataset = cache_dir / "dataset_train.pkl"
    dataset.write_bytes(b"cached training data")

    assert _training_dataset(cache_dir) == dataset.resolve()


def test_concurrent_training_dataset_cache_population(tmp_path, monkeypatch):
    from types import SimpleNamespace

    download_barrier = Barrier(2)
    download_calls = []

    def fake_download(*, id, output, quiet):
        download_calls.append(id)
        with gzip.open(output, "wb") as compressed:
            compressed.write(b"concurrent training data")
        download_barrier.wait(timeout=10)
        return output

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=fake_download))
    with ThreadPoolExecutor(max_workers=2) as executor:
        datasets = list(executor.map(lambda _: _training_dataset(tmp_path), range(2)))

    assert datasets[0] == datasets[1]
    assert datasets[0].read_bytes() == b"concurrent training data"
    assert download_calls == [DATASET_ID, DATASET_ID]


DATASET_ID = "19uNh7XwHWMhF4PGLFcD-VM5YvYXBpHV_"


@pytest.mark.parametrize(
    ("sample_count", "max_samples", "train_ratio", "message"),
    [
        (10, 1, 0.8, "--max_samples must be at least 2"),
        (1, None, 0.8, "at least two dataset samples"),
        (2, None, 0.4, "at least one sample to training"),
        (2, None, 1.0, "at least one sample for validation"),
    ],
)
def test_training_split_rejects_empty_or_one_sample_training(
    sample_count, max_samples, train_ratio, message
):
    from train import _training_split_size

    with pytest.raises(ValueError, match=message):
        _training_split_size(sample_count, max_samples, train_ratio)


def test_training_split_keeps_validation_sample():
    from train import _training_split_size

    assert _training_split_size(2, None, 0.5) == (2, 1)


def _training_dataset(cache_dir):
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = cache_dir / "dataset_train.pkl"
    if dataset.is_file():
        return dataset

    archive = cache_dir / "dataset_train.pkl.gz"
    if not archive.is_file():
        gdown = pytest.importorskip("gdown", reason="install requirements-test.txt to download data")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=archive.name + ".",
            suffix=".part",
            dir=cache_dir,
        )
        os.close(descriptor)
        partial_archive = Path(temporary_name)
        try:
            downloaded = gdown.download(
                id=DATASET_ID,
                output=str(partial_archive),
                quiet=False,
            )
            if downloaded is None or not partial_archive.is_file():
                raise RuntimeError("Google Drive did not provide the GrainGNN training dataset")
            partial_archive.replace(archive)
        finally:
            partial_archive.unlink(missing_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=dataset.name + ".",
        suffix=".part",
        dir=cache_dir,
    )
    os.close(descriptor)
    partial_dataset = Path(temporary_name)
    try:
        with gzip.open(archive, "rb") as compressed, partial_dataset.open("wb") as extracted:
            shutil.copyfileobj(compressed, extracted)
        partial_dataset.replace(dataset)
    finally:
        partial_dataset.unlink(missing_ok=True)
    return dataset
