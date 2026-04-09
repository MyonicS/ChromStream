from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from chromstream.objects import ChannelChromatograms, Chromatogram, Experiment

__all__ = ["parse_experiment_hdf5"]

_EXPECTED_SCHEMA = "chromstream-experiment/v0.1.0"
_RESERVED_ATTRS = frozenset({"schema", "label", "creation_date", "author"})
_INJECTION_NAME_PATTERN = re.compile(r"inj-(?P<index>\d+)$")
_UNSUPPORTED_FILE_MESSAGE = "Only parsing of ChromStream HDF5 files is supported."


def _normalize_attr_value(value: object) -> object:
    """Convert HDF5 attribute values to plain Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_attr(attrs: h5py.AttributeManager, key: str, context: str) -> object:
    """Return a required HDF5 attribute or raise a descriptive error."""
    if key not in attrs:
        raise ValueError(f"Missing required attribute {key!r} in {context}.")
    return _normalize_attr_value(attrs[key])


def _parse_timestamp(value: object, context: str) -> pd.Timestamp:
    """Parse a timestamp attribute and raise a consistent error on failure."""
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Invalid timestamp {value!r} in {context}.") from exc

    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp {value!r} in {context}.")
    return timestamp


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
        if "schema" not in hdf.attrs:
            raise ValueError(
                f"{_UNSUPPORTED_FILE_MESSAGE} Missing required attribute 'schema' in file root."
            )
        schema = _normalize_attr_value(hdf.attrs["schema"])
        if "chromstream" not in str(schema).lower():
            raise ValueError(
                f"{_UNSUPPORTED_FILE_MESSAGE} Found schema {schema!r}."
            )
        if schema != _EXPECTED_SCHEMA:
            raise ValueError(
                f"Unknown schema {schema!r}. Expected {_EXPECTED_SCHEMA!r}."
            )

        label = _require_attr(hdf.attrs, "label", "file root")
        creation_date = (
            _parse_timestamp(hdf.attrs["creation_date"], "file root attribute 'creation_date'")
            if "creation_date" in hdf.attrs
            else None
        )
        author = (
            _normalize_attr_value(hdf.attrs["author"]) if "author" in hdf.attrs else None
        )
        metadata = {
            key: _normalize_attr_value(value)
            for key, value in hdf.attrs.items()
            if key not in _RESERVED_ATTRS
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
                raise ValueError(f"Channel entry {channel_name!r} must be an HDF5 group.")

            stored_channel_name = _require_attr(
                channel_group.attrs, "name", f"channel group {channel_name!r}"
            )
            if stored_channel_name != channel_name:
                raise ValueError(
                    f"Channel group name mismatch for {channel_name!r}: "
                    f"stored name is {stored_channel_name!r}."
                )

            if "injections" not in channel_group:
                raise ValueError(
                    f"Missing required group 'injections' in channel {channel_name!r}."
                )

            injections_group = channel_group["injections"]
            if not isinstance(injections_group, h5py.Group):
                raise ValueError(
                    f"'injections' in channel {channel_name!r} must be an HDF5 group."
                )

            channel = ChannelChromatograms(channel=channel_name)
            for injection_group_name in injections_group:
                match = _INJECTION_NAME_PATTERN.fullmatch(injection_group_name)
                if match is None:
                    raise ValueError(
                        f"Invalid injection group name {injection_group_name!r} "
                        f"in channel {channel_name!r}."
                    )

                injection_group = injections_group[injection_group_name]
                if not isinstance(injection_group, h5py.Group):
                    raise ValueError(
                        f"Injection entry {injection_group_name!r} in channel "
                        f"{channel_name!r} must be an HDF5 group."
                    )

                injection_time = _parse_timestamp(
                    _require_attr(
                        injection_group.attrs,
                        "injection_time",
                        f"injection group {injection_group_name!r}",
                    ),
                    f"injection group {injection_group_name!r}",
                )

                if "retention_time" not in injection_group or "signal" not in injection_group:
                    raise ValueError(
                        f"Injection group {injection_group_name!r} in channel "
                        f"{channel_name!r} must contain 'retention_time' and 'signal' datasets."
                    )

                retention_time_dataset = injection_group["retention_time"]
                signal_dataset = injection_group["signal"]
                if not isinstance(retention_time_dataset, h5py.Dataset):
                    raise ValueError(
                        f"'retention_time' in injection {injection_group_name!r} "
                        f"must be an HDF5 dataset."
                    )
                if not isinstance(signal_dataset, h5py.Dataset):
                    raise ValueError(
                        f"'signal' in injection {injection_group_name!r} must be an HDF5 dataset."
                    )

                time_column_name = _require_attr(
                    retention_time_dataset.attrs,
                    "column_name",
                    f"dataset 'retention_time' in {injection_group_name!r}",
                )
                signal_column_name = _require_attr(
                    signal_dataset.attrs,
                    "column_name",
                    f"dataset 'signal' in {injection_group_name!r}",
                )
                time_unit = _require_attr(
                    retention_time_dataset.attrs,
                    "unit",
                    f"dataset 'retention_time' in {injection_group_name!r}",
                )
                signal_unit = _require_attr(
                    signal_dataset.attrs,
                    "unit",
                    f"dataset 'signal' in {injection_group_name!r}",
                )

                time_values = retention_time_dataset[()]
                signal_values = signal_dataset[()]
                if len(time_values) != len(signal_values):
                    raise ValueError(
                        f"Dataset length mismatch in injection {injection_group_name!r} "
                        f"for channel {channel_name!r}."
                    )

                chromatogram = Chromatogram(
                    data=pd.DataFrame(
                        {
                            str(time_column_name): time_values,
                            str(signal_column_name): signal_values,
                        }
                    ),
                    injection_time=injection_time,
                    metadata={"time_unit": str(time_unit), "Signal Unit": str(signal_unit)},
                    channel=channel_name,
                    path=input_path,
                )
                channel.add_chromatogram(int(match.group("index")), chromatogram)

            experiment.add_channel(channel_name, channel)

    return experiment
