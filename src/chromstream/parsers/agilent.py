"""
Agilent `.ch` files, `.d` folders and OpenLab `.dx` archives.

Reading is done by chromparser (https://github.com/MyonicS/chromparser).
"""

from __future__ import annotations

import logging as log
from pathlib import Path

import chromparser
from chromparser.vendors.agilent import (
    decode_ch,
    read_agilent_ch,
    read_agilent_d,
    read_agilent_dx,
)

from chromstream.objects import Chromatogram

from ._chromparser import to_chromatograms

__all__ = ["parse_agilent_ch", "parse_agilent_dot_d", "parse_agilent_dx"]


def _legacy_metadata(injection: chromparser.Injection):
    """Metadata keys of the earlier ChromStream Agilent parser, per signal."""
    metadata = injection.metadata

    def legacy(signal: chromparser.Signal) -> dict:
        return {
            "version": signal.metadata.get("agilent_file_version", ""),
            "Sample Name": metadata.get("sample_name", ""),
            "Sample Description": metadata.get("agilent_sample_description", ""),
            "Method Name": metadata.get("instrument_method", ""),
            "Operator": metadata.get("operator", ""),
            "date": metadata.get("agilent_date", ""),
            "Instrument": metadata.get("instrument", ""),
            "Inlet": metadata.get("agilent_inlet", ""),
            "Signal Unit": signal.unit or "",
        }

    return legacy


def _convert(
    injection: chromparser.Injection,
    fmt: str,
    path: str | Path,
    channel_name: str | None = None,
) -> list[Chromatogram]:
    if injection.inject_time is None:
        raise ValueError(f"Invalid injection time parsed from {path}")
    return to_chromatograms(
        injection,
        fmt,
        path,
        legacy_metadata=_legacy_metadata(injection),
        channel_name=channel_name,
    )


def parse_agilent_ch(file_path, file_name=None, channel_name=None) -> Chromatogram:
    """
    Parses Agilent .ch files to a Chromatogram object.
    Supports versions 8, 81, 179, 181.

    Args:
        file_path (str | Path | file-like): Path to the .ch file or file-like object from zipfile.
        file_name (str, optional): Name of the file when file_path is a file-like object.
        channel_name (str, optional): Override channel name (otherwise extracted from filename).

    Returns:
        Chromatogram: Parsed chromatogram object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the version is unsupported or the injection time is invalid.
    """
    if hasattr(file_path, "read"):
        name = file_name if file_name else "unknown.ch"
        signal, inject_time, metadata = decode_ch(file_path.read(), name)
        injection = chromparser.Injection(
            name=metadata.get("sample_name", ""),
            inject_time=inject_time,
            signals={signal.channel: signal},
            metadata=metadata,
        )
        path = name
    else:
        (injection,) = read_agilent_ch(file_path)
        path = str(file_path)

    (chrom,) = _convert(injection, "agilent-ch", path, channel_name=channel_name)
    return chrom


def parse_agilent_dot_d(path_dir: Path) -> list[Chromatogram]:
    """
    Given a path to an Agilent .d directory, parses all chromatogram files
    and returns a list of Chromatogram objects.

    Args:
        path_dir (str | Path): Path to the .d directory.

    Returns:
        list[Chromatogram]: Parsed Chromatogram objects, one per .ch file in
        file name order.
    """
    path_dir = Path(path_dir)
    if not path_dir.is_dir() or not path_dir.name.lower().endswith(".d"):
        raise ValueError(f"Provided path is not a valid .d directory: {path_dir}")

    try:
        (injection,) = read_agilent_d(path_dir)
    except chromparser.UnsupportedFormat:
        log.warning(f"No .ch files found in directory: {path_dir}")
        return []

    files = {p.stem: p for p in path_dir.iterdir() if p.suffix.lower() == ".ch"}
    return [
        _convert(
            chromparser.Injection(
                name=injection.name,
                inject_time=injection.inject_time,
                signals={channel: signal},
                metadata=injection.metadata,
            ),
            "agilent-d",
            str(files.get(channel, path_dir / f"{channel}.ch")),
        )[0]
        for channel, signal in injection.signals.items()
    ]


def parse_agilent_dx(file_path) -> list[Chromatogram]:
    """
    Parses Agilent .dx files to a list of Chromatogram objects.

    Channel names are taken from the .acmd method file of the archive.

    Args:
        file_path (str | Path): Path to the .dx file.

    Returns:
        list[Chromatogram]: List of parsed Chromatogram objects.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file() or not path.suffix.lower() == ".dx":
        raise ValueError(f"Provided path is not a valid .dx file: {file_path}")

    (injection,) = read_agilent_dx(path)
    return _convert(injection, "agilent-dx", str(path))
