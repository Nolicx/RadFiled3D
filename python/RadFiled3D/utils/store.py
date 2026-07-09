import tarfile
from pathlib import Path

import zstandard as zstd

from RadFiled3D.RadFiled3D import (
    CartesianRadiationField,
    FieldStore,
    PolarRadiationField,
    RadiationField,
    RadiationFieldMetadata,
)

from .load import load_rf3_metadata_plain


# STORE FILE
def store_rf3_file(
    crf: RadiationField | CartesianRadiationField | PolarRadiationField,
    file_path: str | Path,
    metadata: RadiationFieldMetadata | None = None,
) -> None:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if file_path.name.endswith(".rf3"):
        store_rf3_file_plain(crf, file_path, metadata)
    elif file_path.name.endswith(".tar.zst"):
        store_rf3_file_compressed(crf, file_path, metadata)
    else:
        msg = f"File {file_path} is neither a .rf3 nor a .tar.zst file."
        raise ValueError(msg)


def store_rf3_file_compressed(
    crf: RadiationField | CartesianRadiationField | PolarRadiationField,
    file_path: str | Path,
    metadata: RadiationFieldMetadata | None = None,
) -> None:
    file_path = store_rf3_file_plain(crf, file_path, metadata)
    archive_path = Path(file_path).with_suffix("").with_suffix(".tar.zst")

    with open(archive_path, "wb") as compressed_file:
        cctx = zstd.ZstdCompressor(level=3)
        with (
            cctx.stream_writer(compressed_file) as compressor,
            tarfile.open(fileobj=compressor, mode="w|") as tar,
        ):
            tar.add(str(file_path), arcname=Path(file_path).name)
        file_path.unlink()  # remove the uncompressed .rf3 file


def store_rf3_file_plain(
    crf: RadiationField | CartesianRadiationField | PolarRadiationField,
    file_path: str | Path,
    metadata: RadiationFieldMetadata | None = None,
) -> Path:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if not file_path.name.endswith(".rf3"):
        file_path = file_path.with_suffix("").with_suffix(".rf3")
    if metadata is None:
        metadata = load_rf3_metadata_plain(file_path)
    FieldStore.store(crf, metadata, str(file_path))
    return file_path
