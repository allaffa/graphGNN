import gzip
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from scripts.trajectory_io import decompress_gzip_file


def test_decompress_gzip_file_creates_uncompressed_file(tmp_path):
    archive = tmp_path / "traj0.pkl.gz"
    expected = b"serialized trajectory contents"
    with gzip.open(archive, "wb") as compressed:
        compressed.write(expected)

    output = decompress_gzip_file(archive)

    assert output == tmp_path / "traj0.pkl"
    assert output.read_bytes() == expected
    assert decompress_gzip_file(archive) == output


def test_concurrent_decompression_uses_independent_temporary_files(tmp_path, monkeypatch):
    import scripts.trajectory_io as trajectory_io

    archive = tmp_path / "traj0.pkl.gz"
    expected = b"serialized trajectory contents"
    with gzip.open(archive, "wb") as compressed:
        compressed.write(expected)

    copy_barrier = Barrier(2)

    def synchronized_copy(source, destination):
        contents = source.read()
        copy_barrier.wait(timeout=10)
        destination.write(contents)

    monkeypatch.setattr(trajectory_io.shutil, "copyfileobj", synchronized_copy)
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(lambda _: decompress_gzip_file(archive), range(2)))

    assert outputs == [tmp_path / "traj0.pkl"] * 2
    assert outputs[0].read_bytes() == expected