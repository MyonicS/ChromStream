from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from chromstream.objects import ChannelChromatograms, Chromatogram

_INJECTION_NAME_PATTERN = re.compile(r"inj-(?P<index>\d+)$")
_UNSUPPORTED_FILE_MESSAGE = "Only parsing of ChromStream HDF5 files is supported."

# Root attribute names owned by each file type; excluded from the free-form
# metadata dict on read and rejected as metadata keys on write.
RESERVED_EXPERIMENT_ATTRS = frozenset({"schema", "label", "creation_date", "author"})
RESERVED_CHANNEL_ATTRS = frozenset({"schema", "label", "name"})
RESERVED_CHROMATOGRAM_ATTRS = frozenset(
    {"schema", "label", "channel", "injection_index", "injection_time"}
)


def _to_hdf5_attr(value: object) -> str | int | float | bool | bytes:
    """Convert a Python value to an HDF5-compatible scalar attribute."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise TypeError(
        "Metadata values must be scalar HDF5-compatible values. "
        f"Unsupported value {value!r} of type {type(value).__name__}."
    )


def _from_hdf5_attr(value: object) -> object:
    """Convert an HDF5 attribute value to a plain Python value."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_attr(attrs: h5py.AttributeManager, key: str, context: str) -> object:
    """Return a required HDF5 attribute or raise a descriptive error."""
    if key not in attrs:
        raise ValueError(f"Missing required attribute {key!r} in {context}.")
    return _from_hdf5_attr(attrs[key])


def _parse_timestamp(value: object, context: str) -> pd.Timestamp:
    """Parse a timestamp attribute and raise a consistent error on failure."""
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Invalid timestamp {value!r} in {context}.") from exc

    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp {value!r} in {context}.")
    return timestamp


def _require_schema(hdf: h5py.File, expected: str, context: str = "file root") -> str:
    """Validate the root 'schema' attribute against the expected schema string."""
    if "schema" not in hdf.attrs:
        raise ValueError(
            f"{_UNSUPPORTED_FILE_MESSAGE} Missing required attribute 'schema' in {context}."
        )
    schema = _from_hdf5_attr(hdf.attrs["schema"])
    if "chromstream" not in str(schema).lower():
        raise ValueError(f"{_UNSUPPORTED_FILE_MESSAGE} Found schema {schema!r}.")
    # TODO: Handle different schema versions. Currently only an exact match is accepted.
    if schema != expected:
        raise ValueError(f"Unknown schema {schema!r}. Expected {expected!r}.")
    return str(schema)


def _write_chromatogram_group(
    group: h5py.Group,
    chromatogram: Chromatogram,
    *,
    compression: str | None,
    context: str,
) -> None:
    """Write injection_time + retention_time/signal datasets into ``group``.

    Assumes column 0 is retention time and column 1 is signal; other columns
    are ignored.
    """
    if chromatogram.data.shape[1] < 2:
        raise ValueError(f"Chromatogram for {context} must have at least two columns.")
    ret_time_column = chromatogram.data.columns[0]
    signal_column = chromatogram.data.columns[1]

    if chromatogram.injection_time is None or pd.isna(chromatogram.injection_time):
        raise ValueError(
            f"Chromatogram for {context} is missing a valid injection_time."
        )
    group.attrs["injection_time"] = pd.Timestamp(
        chromatogram.injection_time
    ).isoformat()

    retention_time_dataset = group.create_dataset(
        "retention_time",
        data=chromatogram.data[ret_time_column].to_numpy(),
        compression=compression,
    )
    retention_time_dataset.attrs["unit"] = chromatogram.time_unit
    retention_time_dataset.attrs["column_name"] = ret_time_column

    signal_dataset = group.create_dataset(
        "signal",
        data=chromatogram.data[signal_column].to_numpy(),
        compression=compression,
    )
    signal_dataset.attrs["unit"] = chromatogram.signal_unit
    signal_dataset.attrs["column_name"] = signal_column


def _read_chromatogram_group(
    group: h5py.Group,
    *,
    channel: str,
    path: Path | None,
    context: str,
    metadata: dict | None = None,
) -> Chromatogram:
    """Read a single chromatogram from ``group``.

    When ``metadata`` is None (experiment/channel layout) the metadata dict is
    rebuilt from the stored units only; standalone chromatogram files pass the
    full metadata dict explicitly.
    """
    injection_time = _parse_timestamp(
        _require_attr(group.attrs, "injection_time", context), context
    )

    if "retention_time" not in group or "signal" not in group:
        raise ValueError(
            f"{context} must contain 'retention_time' and 'signal' datasets."
        )

    retention_time_dataset = group["retention_time"]
    signal_dataset = group["signal"]
    if not isinstance(retention_time_dataset, h5py.Dataset):
        raise ValueError(f"'retention_time' in {context} must be an HDF5 dataset.")
    if not isinstance(signal_dataset, h5py.Dataset):
        raise ValueError(f"'signal' in {context} must be an HDF5 dataset.")

    time_column_name = _require_attr(
        retention_time_dataset.attrs,
        "column_name",
        f"dataset 'retention_time' in {context}",
    )
    signal_column_name = _require_attr(
        signal_dataset.attrs, "column_name", f"dataset 'signal' in {context}"
    )
    time_unit = _require_attr(
        retention_time_dataset.attrs, "unit", f"dataset 'retention_time' in {context}"
    )
    signal_unit = _require_attr(
        signal_dataset.attrs, "unit", f"dataset 'signal' in {context}"
    )

    time_values = retention_time_dataset[()]
    signal_values = signal_dataset[()]
    if len(time_values) != len(signal_values):
        raise ValueError(f"Dataset length mismatch in {context}.")

    if metadata is None:
        metadata = {"time_unit": str(time_unit), "Signal Unit": str(signal_unit)}

    return Chromatogram(
        data=pd.DataFrame(
            {
                str(time_column_name): time_values,
                str(signal_column_name): signal_values,
            }
        ),
        injection_time=injection_time,
        metadata=metadata,
        channel=channel,
        path=path,
    )


def _write_channel_group(
    group: h5py.Group,
    channel: ChannelChromatograms,
    *,
    compression: str | None,
) -> None:
    """Write a channel's name attr and its injections subgroup into ``group``."""
    group.attrs["name"] = channel.channel
    injections_group = group.create_group("injections")
    for injection_key in sorted(channel.chromatograms):
        chromatogram = channel.chromatograms[injection_key]
        injection_group = injections_group.create_group(f"inj-{injection_key:04d}")
        _write_chromatogram_group(
            injection_group,
            chromatogram,
            compression=compression,
            context=f"channel {channel.channel!r} injection {injection_key!r}",
        )


def _read_channel_group(
    group: h5py.Group,
    *,
    path: Path | None,
    expected_name: str | None = None,
) -> ChannelChromatograms:
    """Read a ChannelChromatograms from ``group`` (name attr + injections group)."""
    context = f"channel group {group.name!r}"
    stored_name = _require_attr(group.attrs, "name", context)
    channel_name = str(stored_name)
    if expected_name is not None and channel_name != expected_name:
        raise ValueError(
            f"Channel group name mismatch for {expected_name!r}: "
            f"stored name is {stored_name!r}."
        )

    if "injections" not in group:
        raise ValueError(
            f"Missing required group 'injections' in channel {channel_name!r}."
        )
    injections_group = group["injections"]
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
        chromatogram = _read_chromatogram_group(
            injection_group,
            channel=channel_name,
            path=path,
            context=f"injection group {injection_group_name!r}",
        )
        channel.add_chromatogram(int(match.group("index")), chromatogram)

    return channel
