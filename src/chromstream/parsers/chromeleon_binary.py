"""
Parser for binary Chromeleon 7 data: `.cmbx` archives and raw signal files.

Decoding is done by [cmbx-reader](https://github.com/MyonicS/cmbx-reader),
which also documents the file format. This module converts its results to
`Chromatogram` objects with the column names and metadata keys of the
Chromeleon `.txt` parser.

A `.cmbx` file (Chromeleon "Export to archive") is a zip archive with the
sequence and one raw data file per signal. Raw signal files (`.raw` starting
with `SignHdr`) can also be found in the `RawData` folder of a data vault.
"""

from __future__ import annotations

from pathlib import Path

import cmbx_reader
import numpy as np
import pandas as pd

from chromstream.objects import Chromatogram

# cmbx-reader metadata keys -> keys used by the Chromeleon .txt parser
_METADATA_KEYS = {
    "url": "URL",
    "data_vault": "Data Vault",
    "sequence": "Sequence",
    "type": "Type",
    "status": "Status",
    "position": "Position",
    "instrument_method": "Instrument Method",
    "guid": "Injection GUID",
    "injection_guid": "Injection GUID",
    "raw_data_file": "Raw Data File",
    "raw_data_file_id": "Raw Data File ID",
    "signal_quantity": "Signal Quantity",
    "detector": "Detector",
    "driver": "Driver Name",
    "generating_data_system": "Generating Data System",
}


def _to_chromatogram(
    signal: cmbx_reader.Signal,
    injection: cmbx_reader.Injection | None,
    path: Path | str,
    channel_name: str | None = None,
    signal_unit: str | None = None,
    timezone: str | None = None,
) -> Chromatogram:
    time = signal.time
    values = signal.values  # noqa: PD011 (numpy array, not pandas)
    unit = signal_unit or signal.unit
    value_column = f"Value ({unit})" if unit else "Value"
    frame = pd.DataFrame({"Time (min)": time, value_column: values})

    metadata: dict = {}
    for source in (injection.metadata if injection else {}, signal.metadata):
        for key, value in source.items():
            metadata[_METADATA_KEYS.get(key, key)] = value
    metadata["Channel"] = signal.channel
    if injection is not None:
        metadata["Name"] = injection.name
    if unit:
        metadata["Signal Unit"] = unit
    if signal.acquisition_start is not None:
        start = pd.Timestamp(signal.acquisition_start)
        metadata["Acquisition Start (UTC)"] = start.tz_convert(None)

    metadata["Data Points"] = len(frame)
    if len(frame):
        metadata["Time Min. (min)"] = float(time[0])
        metadata["Time Max. (min)"] = float(time[-1])
        metadata["Signal Min."] = float(values.min())
        metadata["Signal Max."] = float(values.max())
    if len(frame) > 1:
        steps = np.diff(time) * 60
        # rounded to 1 ns to drop float noise from the minute conversion
        metadata["Min. Step (s)"] = round(float(steps.min()), 9)
        metadata["Max. Step (s)"] = round(float(steps.max()), 9)
        metadata["Average Step (s)"] = round(float(steps.mean()), 9)
    metadata["time_unit"] = "min"

    if injection is not None and injection.inject_time is not None:
        # local wall time of the instrument, like the .txt parser
        injection_time = pd.Timestamp(injection.inject_time)
        if injection_time.tzinfo is not None:
            injection_time = injection_time.tz_localize(None)
    elif signal.acquisition_start is not None:
        start = pd.Timestamp(signal.acquisition_start)
        if timezone:
            start = start.tz_convert(timezone)
        injection_time = start.tz_localize(None)
    else:
        raise ValueError(f"No injection time found in {path}.")
    metadata["Inject Time"] = injection_time

    return Chromatogram(
        data=frame,
        injection_time=injection_time,
        metadata=metadata,
        channel=channel_name or signal.channel or "unknown",
        path=path,
    )


def parse_chromeleon_raw(
    file_path,
    file_name: str | None = None,
    channel_name: str | None = None,
    signal_unit: str | None = None,
    timezone: str | None = None,
) -> Chromatogram:
    """Parse a single Chromeleon 7 raw signal file into a Chromatogram.

    These are the `.raw` files starting with `SignHdr`, found inside `.cmbx`
    archives or in the `RawData` folder of a Chromeleon data vault. Use
    `parse_chromeleon_cmbx` for archives, which also reads the signal unit and
    injection metadata from the sequence.

    A raw file on its own only stores the acquisition start in UTC; it does not
    contain the signal unit or the time zone of the instrument.

    Args:
        file_path (str | Path | bytes | file-like): Raw signal file.
        file_name (str, optional): Name used as path when `file_path` is bytes
            or a file-like object.
        channel_name (str, optional): Override the channel name stored in the file.
        signal_unit (str, optional): Signal unit (e.g. "pA"), not stored in raw files.
        timezone (str, optional): Time zone (e.g. "Europe/Amsterdam") to convert
            the UTC acquisition start to local wall time. Defaults to UTC.

    Returns:
        Chromatogram: Columns `Time (min)` and `Value (<unit>)` (`Value` when the
            unit is unknown). The injection time is a naive timestamp.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a supported Chromeleon signal file.
    """
    if isinstance(file_path, (bytes, bytearray)) or hasattr(file_path, "read"):
        path = file_name or "unknown.raw"
    else:
        path = str(Path(file_path))
    signal = cmbx_reader.read_raw(file_path)
    return _to_chromatogram(signal, None, path, channel_name, signal_unit, timezone)


def parse_chromeleon_cmbx(
    file_path: str | Path, timezone: str | None = None
) -> list[Chromatogram]:
    """Parse all signals in a Chromeleon 7 `.cmbx` archive.

    Injection metadata (inject time, type, status, instrument method) and the
    signal unit are read from the sequence file in the archive. MS data and
    audit trails in the archive are skipped.

    Args:
        file_path (str | Path): Path to the `.cmbx` file.
        timezone (str, optional): Only used when the archive has no readable
            inject time; see `parse_chromeleon_raw`.

    Returns:
        list[Chromatogram]: One chromatogram per signal, in archive order. The
            injection time is the naive local wall time of the instrument, as
            in the Chromeleon text export.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a Chromeleon archive.
    """
    path = Path(file_path)
    return [
        _to_chromatogram(signal, injection, path, timezone=timezone)
        for injection in cmbx_reader.read_cmbx(path)
        for signal in injection.signals.values()
    ]
