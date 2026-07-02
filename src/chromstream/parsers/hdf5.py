from __future__ import annotations

from pathlib import Path

import h5py

from chromstream.hdf5_common import (
    RESERVED_CHROMATOGRAM_ATTRS,
    RESERVED_EXPERIMENT_ATTRS,
    _from_hdf5_attr,
    _parse_timestamp,
    _read_channel_group,
    _read_chromatogram_group,
    _require_attr,
    _require_schema,
)
from chromstream.objects import (
    SCHEMA_CHANNEL,
    SCHEMA_CHROMATOGRAM,
    SCHEMA_EXPERIMENT,
    ChannelChromatograms,
    Chromatogram,
    Experiment,
)

__all__ = [
    "parse_channel_hdf5",
    "parse_chromatogram_hdf5",
    "parse_experiment_hdf5",
]


def parse_experiment_hdf5(path: str | Path) -> Experiment:
    """Parse a ChromStream HDF5 experiment file into an Experiment object.

    Args:
        path: Path to the HDF5 file written by ChromStream.

    Returns:
        Experiment: Reconstructed experiment containing persisted channels and chromatograms.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file schema or layout does not match the expected format.
        OSError: If the file cannot be opened as HDF5.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with h5py.File(input_path, "r") as hdf:
        schema = _require_schema(hdf, SCHEMA_EXPERIMENT)

        label = _require_attr(hdf.attrs, "label", "file root")
        creation_date = (
            _parse_timestamp(
                hdf.attrs["creation_date"], "file root attribute 'creation_date'"
            )
            if "creation_date" in hdf.attrs
            else None
        )
        author = (
            _from_hdf5_attr(hdf.attrs["author"]) if "author" in hdf.attrs else None
        )
        metadata = {
            key: _from_hdf5_attr(value)
            for key, value in hdf.attrs.items()
            if key not in RESERVED_EXPERIMENT_ATTRS
        }

        if "Channels" not in hdf:
            raise ValueError("Missing required group 'Channels' in file root.")
        channels_group = hdf["Channels"]
        if not isinstance(channels_group, h5py.Group):
            raise ValueError("'Channels' must be an HDF5 group.")

        experiment = Experiment(
            name=str(label),
            schema=str(schema),
            author=str(author) if author is not None else None,
            creation_date=creation_date,
            metadata=metadata,
        )

        for channel_name in channels_group:
            channel_group = channels_group[channel_name]
            if not isinstance(channel_group, h5py.Group):
                raise ValueError(
                    f"Channel entry {channel_name!r} must be an HDF5 group."
                )
            channel = _read_channel_group(
                channel_group, path=input_path, expected_name=channel_name
            )
            experiment.add_channel(channel_name, channel)

    return experiment


def parse_channel_hdf5(path: str | Path) -> ChannelChromatograms:
    """Parse a standalone ChromStream channel HDF5 file into a ChannelChromatograms object.

    Args:
        path: Path to the channel HDF5 file written by ChromStream.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file schema or layout does not match the expected format.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with h5py.File(input_path, "r") as hdf:
        _require_schema(hdf, SCHEMA_CHANNEL)
        channel = _read_channel_group(hdf, path=input_path)

    return channel


def _parse_chromatogram_hdf5_with_index(
    path: str | Path,
) -> tuple[int | None, Chromatogram]:
    """Parse a standalone chromatogram file, returning (injection_index, Chromatogram).

    injection_index is the stored channel-level key when present, else None.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with h5py.File(input_path, "r") as hdf:
        _require_schema(hdf, SCHEMA_CHROMATOGRAM)
        channel = _require_attr(hdf.attrs, "channel", "file root")
        injection_index = (
            int(_from_hdf5_attr(hdf.attrs["injection_index"]))
            if "injection_index" in hdf.attrs
            else None
        )
        # everything not owned by the layout is restored as chromatogram metadata
        metadata = {
            key: _from_hdf5_attr(value)
            for key, value in hdf.attrs.items()
            if key not in RESERVED_CHROMATOGRAM_ATTRS
        }
        chromatogram = _read_chromatogram_group(
            hdf,
            channel=str(channel),
            path=input_path,
            context="file root",
            metadata=metadata,
        )

    return injection_index, chromatogram


def parse_chromatogram_hdf5(path: str | Path) -> Chromatogram:
    """Parse a standalone ChromStream chromatogram HDF5 file into a Chromatogram object.

    Args:
        path: Path to the chromatogram HDF5 file written by ChromStream.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file schema or layout does not match the expected format.
    """
    return _parse_chromatogram_hdf5_with_index(path)[1]
