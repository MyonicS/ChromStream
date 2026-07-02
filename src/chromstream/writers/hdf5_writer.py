from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd

from chromstream.hdf5_common import (
    RESERVED_CHROMATOGRAM_ATTRS,
    RESERVED_EXPERIMENT_ATTRS,
    _to_hdf5_attr,
    _write_channel_group,
    _write_chromatogram_group,
)
from chromstream.objects import (
    SCHEMA_CHANNEL,
    SCHEMA_CHROMATOGRAM,
    ChannelChromatograms,
    Chromatogram,
    Experiment,
)

__all__ = [
    "write_channel_hdf5",
    "write_chromatogram_hdf5",
    "write_experiment_hdf5",
]


def _open_for_write(path: str | Path, overwrite: bool) -> tuple[Path, str]:
    """Resolve the output path and open mode, refusing to clobber unless allowed."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"File {output_path!r} already exists. Set overwrite=True to overwrite it."
        )
    return output_path, "w" if overwrite else "x"


def write_experiment_hdf5(
    experiment: Experiment,
    path: str | Path,
    *,
    overwrite: bool = False,
    compression: str | None = None,
) -> Path:
    """
    Write a single Experiment object to an HDF5 file.
    Assumptions: Col 0 is retention time, Col 1 is signal. All other columns are ignored.

    Args:
        experiment: The Experiment object to write.
        path: The path to the HDF5 file to write.
        overwrite: If True, overwrite the file if it exists.
        compression: The compression algorithm to use for datasets. Available options include "gzip", "lzf", or None for no compression. Compression can reduce file size but may increase read/write time.
    """
    output_path, mode = _open_for_write(path, overwrite)

    overlapping_keys = RESERVED_EXPERIMENT_ATTRS.intersection(experiment.metadata)
    if overlapping_keys:
        reserved = ", ".join(sorted(overlapping_keys))
        raise ValueError(
            f"Experiment metadata contains reserved attribute names: {reserved}."
        )

    with h5py.File(output_path, mode) as hdf:
        hdf.attrs["schema"] = experiment.schema
        hdf.attrs["label"] = experiment.label
        if experiment.creation_date is not None:
            hdf.attrs["creation_date"] = pd.Timestamp(
                experiment.creation_date
            ).isoformat()
        if experiment.author is not None:
            hdf.attrs["author"] = experiment.author
        # adding other experiment metadata, checking for conflicts with reserved attribute names
        for key, value in experiment.metadata.items():
            hdf.attrs[key] = _to_hdf5_attr(value)

        channels_group = hdf.create_group("Channels")
        for channel_name, channel in experiment.channels.items():
            channel_group = channels_group.create_group(channel_name)
            _write_channel_group(channel_group, channel, compression=compression)

    return output_path


def write_channel_hdf5(
    channel: ChannelChromatograms,
    path: str | Path,
    *,
    overwrite: bool = False,
    compression: str | None = None,
) -> Path:
    """Write a single ChannelChromatograms object to a standalone HDF5 file.

    Args:
        channel: The ChannelChromatograms object to write.
        path: The path to the HDF5 file to write.
        overwrite: If True, overwrite the file if it exists.
        compression: Dataset compression ("gzip", "lzf", or None).
    """
    output_path, mode = _open_for_write(path, overwrite)

    with h5py.File(output_path, mode) as hdf:
        hdf.attrs["schema"] = SCHEMA_CHANNEL
        _write_channel_group(hdf, channel, compression=compression)

    return output_path


def write_chromatogram_hdf5(
    chromatogram: Chromatogram,
    path: str | Path,
    *,
    injection_index: int | None = None,
    overwrite: bool = False,
    compression: str | None = None,
) -> Path:
    """Write a single Chromatogram object to a standalone HDF5 file.

    The full metadata dict is persisted so parsing restores it key-for-key.

    Args:
        chromatogram: The Chromatogram object to write.
        path: The path to the HDF5 file to write.
        injection_index: Optional channel-level injection index to persist for
            lossless reassembly into an Experiment.
        overwrite: If True, overwrite the file if it exists.
        compression: Dataset compression ("gzip", "lzf", or None).
    """
    output_path, mode = _open_for_write(path, overwrite)

    overlapping_keys = RESERVED_CHROMATOGRAM_ATTRS.intersection(chromatogram.metadata)
    if overlapping_keys:
        reserved = ", ".join(sorted(overlapping_keys))
        raise ValueError(
            f"Chromatogram metadata contains reserved attribute names: {reserved}."
        )

    with h5py.File(output_path, mode) as hdf:
        hdf.attrs["schema"] = SCHEMA_CHROMATOGRAM
        hdf.attrs["channel"] = chromatogram.channel
        if injection_index is not None:
            hdf.attrs["injection_index"] = int(injection_index)
        # full metadata round-trip: persist every metadata key (units included)
        for key, value in chromatogram.metadata.items():
            hdf.attrs[key] = _to_hdf5_attr(value)
        _write_chromatogram_group(
            hdf, chromatogram, compression=compression, context="file root"
        )

    return output_path
