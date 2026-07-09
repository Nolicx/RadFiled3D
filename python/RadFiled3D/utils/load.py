import tarfile
from pathlib import Path

import zstandard as zstd

from RadFiled3D.RadFiled3D import (
    CartesianRadiationField,
    FieldStore,
    PolarRadiationField,
    RadiationField,
)


# LOAD FILE
def load_rf3_file(
    file_path: str | Path,
) -> RadiationField | CartesianRadiationField | PolarRadiationField:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if file_path.name.endswith(".rf3"):
        return load_rf3_file_plain(file_path)
    elif file_path.name.endswith(".tar.zst"):
        return load_rf3_file_from_buffer(file_path)
    else:
        msg = f"File {file_path} is neither a .rf3 nor a .tar.zst file."
        raise ValueError(msg)


def load_rf3_file_plain(
    file_path: str | Path,
) -> RadiationField | CartesianRadiationField | PolarRadiationField:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if file_path.suffix != ".rf3":
        file_path = file_path.with_suffix(".rf3")
    return FieldStore.load(str(file_path))


def load_rf3_file_from_buffer(file_path: str | Path):
    buffer_content = load_rf3_file_buffer(file_path)
    return FieldStore.load_from_buffer(buffer_content)  # type: ignore


def load_rf3_file_buffer(file_path: str | Path) -> bytes:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if not file_path.name.endswith(".tar.zst"):
        file_path = file_path.with_suffix(".tar.zst")

    content = None
    with open(file_path, "rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with (
            dctx.stream_reader(compressed_file) as reader,
            tarfile.open(fileobj=reader, mode="r|") as tar,
        ):
            # Assuming there's only one .rf3 file in the tar archive
            for member in tar:
                if member.isfile() and member.name.endswith(".rf3"):
                    rf3_file = tar.extractfile(member)
                    if rf3_file:
                        content = rf3_file.read()
                        break
    if content is None:
        msg = f"Could not find rf3_actor.rf3 in the archive {file_path}"
        raise FileNotFoundError(msg)
    return content


# LOAD METADATA
def load_rf3_metadata(file_path: str | Path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if file_path.name.endswith(".rf3"):
        return load_rf3_metadata_plain(file_path)
    elif file_path.name.endswith(".tar.zst"):
        return load_rf3_metadata_from_buffer(file_path)
    else:
        msg = f"File {file_path} is neither a .rf3 nor a .tar.zst file."
        raise ValueError(msg)


def load_rf3_metadata_plain(file_path: str | Path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if file_path.suffix != ".rf3":
        file_path = file_path.with_suffix(".rf3")
    return FieldStore.load_metadata(str(file_path))


def load_rf3_metadata_from_buffer(file_path: str | Path):
    buffer_content = load_rf3_file_buffer(file_path)
    return FieldStore.load_metadata_from_buffer(buffer_content)  # type: ignore
