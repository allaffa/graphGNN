import gzip

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