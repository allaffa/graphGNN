"""Helpers for reading trajectory files."""

import gzip
import shutil
from pathlib import Path


def decompress_gzip_file(archive_path):
    archive_path = Path(archive_path)
    if archive_path.suffix != ".gz":
        raise ValueError(f"Expected a .gz archive: {archive_path}")

    output_path = archive_path.with_suffix("")
    if output_path.is_file():
        return output_path
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    partial_path = output_path.with_name(output_path.name + ".part")
    try:
        with gzip.open(archive_path, "rb") as compressed:
            with partial_path.open("wb") as extracted:
                shutil.copyfileobj(compressed, extracted)
        partial_path.replace(output_path)
    finally:
        partial_path.unlink(missing_ok=True)

    return output_path
