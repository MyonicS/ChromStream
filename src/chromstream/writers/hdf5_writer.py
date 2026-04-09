from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from chromstream.objects import Experiment

__all__ = ["write_experiment_hdf5"]

_RESERVED_ATTRS = frozenset({"schema", "label", "creation_date", "author"})


def _normalize_attr_value(value: object) -> str | int | float | bool | bytes:
    """Convert supported Python values to HDF5-compatible scalar attributes."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise TypeError(
        "Experiment metadata values must be scalar HDF5-compatible values. "
        f"Unsupported value {value!r} of type {type(value).__name__}."
    )


def write_experiment_hdf5(
    experiment: Experiment,
    path: str | Path,
    *,
    overwrite: bool = False,
    compression: str | None = None,
) -> Path:
    """Write a single Experiment object to an HDF5 file.

    Args:
        experiment: The Experiment object to write.
        path: The path to the HDF5 file to write.
        overwrite: If True, overwrite the file if it exists.
        compression: The compression algorithm to use for datasets. Available options include "gzip", "lzf", or None for no compression. Compression can reduce file size but may increase read/write time.
    """
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"File {output_path!r} already exists. Set overwrite=True to overwrite it."
        )

    overlapping_keys = _RESERVED_ATTRS.intersection(experiment.metadata)
    if overlapping_keys:
        reserved = ", ".join(sorted(overlapping_keys))
        raise ValueError(
            f"Experiment metadata contains reserved attribute names: {reserved}."
        )

    mode = "w" if overwrite else "x"

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
            hdf.attrs[key] = _normalize_attr_value(value)

        channels_group = hdf.create_group("Channels")
        for channel_name, channel in experiment.channels.items():
            channel_group = channels_group.create_group(channel_name)
            channel_group.attrs["name"] = channel_name

            injections_group = channel_group.create_group("injections")
            for injection_key in sorted(channel.chromatograms):
                chromatogram = channel.chromatograms[injection_key]
                if chromatogram.data.shape[1] < 2:
                    raise ValueError(
                        f"Chromatogram for channel {channel_name!r} injection "
                        f"{injection_key!r} must have at least two columns."
                    )
                # assuming the first and second columns of the chromatogram data are retention time and signal
                ret_time_column = chromatogram.data.columns[0]
                signal_column = chromatogram.data.columns[1]

                injection_group = injections_group.create_group(
                    f"inj-{injection_key:04d}"
                )
                if chromatogram.injection_time is not None and not pd.isna(
                    chromatogram.injection_time
                ):
                    injection_group.attrs["injection_time"] = pd.Timestamp(
                        chromatogram.injection_time
                    ).isoformat()
                else:
                    raise ValueError(
                        f"Chromatogram for channel {channel_name!r} injection "
                        f"{injection_key!r} is missing a valid injection_time."
                    )

                retention_time_dataset = injection_group.create_dataset(
                    "retention_time",
                    data=chromatogram.data[ret_time_column].to_numpy(),
                    compression=compression,
                )
                retention_time_dataset.attrs["unit"] = chromatogram.time_unit
                retention_time_dataset.attrs["column_name"] = ret_time_column

                signal_dataset = injection_group.create_dataset(
                    "signal",
                    data=chromatogram.data[signal_column].to_numpy(),
                    compression=compression,
                )
                signal_dataset.attrs["unit"] = chromatogram.signal_unit
                signal_dataset.attrs["column_name"] = signal_column

    return output_path
