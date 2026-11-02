"""Opt-in training smoke test using the published training dataset."""

import gzip
import os
import shutil
import subprocess
import sys
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
    (work_dir / "dataset_train.pkl").symlink_to(dataset)
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


DATASET_ID = "19uNh7XwHWMhF4PGLFcD-VM5YvYXBpHV_"


def _training_dataset(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = cache_dir / "dataset_train.pkl"
    if dataset.is_file():
        return dataset

    archive = cache_dir / "dataset_train.pkl.gz"
    if not archive.is_file():
        gdown = pytest.importorskip("gdown", reason="install requirements-test.txt to download data")
        partial_archive = archive.with_suffix(archive.suffix + ".part")
        partial_archive.unlink(missing_ok=True)
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

    partial_dataset = dataset.with_suffix(dataset.suffix + ".part")
    partial_dataset.unlink(missing_ok=True)
    try:
        with gzip.open(archive, "rb") as compressed, partial_dataset.open("wb") as extracted:
            shutil.copyfileobj(compressed, extracted)
        partial_dataset.replace(dataset)
    finally:
        partial_dataset.unlink(missing_ok=True)
    return dataset
