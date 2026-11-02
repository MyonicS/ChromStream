"""
Conversion of chromparser `Injection` objects into ChromStream `Chromatogram`s.

All raw-file reading is done by chromparser
(https://github.com/MyonicS/chromparser). This module only maps its data model
onto `Chromatogram`, keeping ChromStream's column and metadata conventions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import chromparser
import pandas as pd

from chromstream.objects import Chromatogram

# Formats whose chromatograms historically used "Time"/"Signal" columns.
_AGILENT_FORMATS = ("agilent-ch", "agilent-d", "agilent-dx")


def _scalar(value: object) -> object:
    """Make a chromparser metadata value a scalar (lists/dicts become JSON)."""
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "; ".join(value)
    if isinstance(value, list | dict):
        return json.dumps(value)
    return value


def to_timestamp(
    injection: chromparser.Injection, signal: chromparser.Signal
) -> pd.Timestamp:
    """
    Injection time of a chromparser injection as a naive `pd.Timestamp`.

    The local wall time of the instrument is kept and any UTC offset dropped,
    matching ChromStream's earlier parsers. Formats without an inject time
    (Chromeleon raw files, Shimadzu) fall back to the signal's acquisition
    start, converted to the local time zone of this computer.

    Raises:
        ValueError: If the file holds neither time.
    """
    if injection.inject_time is not None:
        return pd.Timestamp(injection.inject_time.replace(tzinfo=None))
    if signal.acquisition_start is not None:
        return pd.Timestamp(signal.acquisition_start.astimezone().replace(tzinfo=None))
    raise ValueError(f"No injection time found for injection '{injection.name}'.")


def to_chromatograms(
    injection: chromparser.Injection,
    fmt: str,
    path: str | Path | None,
    legacy_metadata: Callable[[chromparser.Signal], dict] | None = None,
    channel_name: str | None = None,
) -> list[Chromatogram]:
    """
    Convert one chromparser injection into one `Chromatogram` per signal.

    Args:
        injection: Injection returned by chromparser.
        fmt: chromparser format id (`chromparser.sniff`), selects column names.
        path: Value for `Chromatogram.path`.
        legacy_metadata: Optional callable giving the metadata keys of the
            earlier ChromStream parser for a signal; placed before the
            chromparser (snake_case) keys.
        channel_name: Optional channel name override (single-signal files).

    Returns:
        list[Chromatogram]: Retention time in minutes in the first column,
        signal in the second.
    """
    chroms = []
    for signal in injection.signals.values():
        unit = signal.unit or "unknown"
        if fmt in _AGILENT_FORMATS:
            columns = ["Time", "Signal"]
        else:
            columns = ["Time (min)", f"Value ({unit})"]
        data = pd.DataFrame({columns[0]: signal.time, columns[1]: signal.values})

        injection_time = to_timestamp(injection, signal)
        metadata = dict(legacy_metadata(signal)) if legacy_metadata else {}
        for key, value in {**injection.metadata, **signal.metadata}.items():
            metadata.setdefault(key, _scalar(value))
        if injection.name:
            metadata.setdefault("injection_name", injection.name)
        if signal.sampling_interval is not None:
            metadata.setdefault("sampling_interval_s", signal.sampling_interval)
        metadata.setdefault("Signal Unit", unit)
        metadata.setdefault("time_unit", "min")
        metadata.setdefault("source_format", fmt)

        chroms.append(
            Chromatogram(
                data=data,
                injection_time=injection_time,
                metadata=metadata,
                channel=channel_name or signal.channel or "unknown",
                path=path,
            )
        )
    return chroms
