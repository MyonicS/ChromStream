from __future__ import annotations

import logging as log
import re
from pathlib import Path
from typing import Optional

import chromparser
import pandas as pd
from chromparser.vendors.chromeleon import read_chromeleon_txt
from chromparser.vendors.chromeleon.txt import parse_inject_time as _parse_inject_time

from chromstream.objects import ChannelChromatograms, Chromatogram

from ._chromparser import to_chromatograms, to_timestamp


def _read_header(path: Path) -> dict[str, str]:
    """
    Read the raw ``key: value`` header of a Chromeleon text export.

    Kept for backwards compatibility: these are the metadata keys (as written
    by Chromeleon) that ChromStream exposed before switching to chromparser.
    """
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252")

    metadata = {}
    metadata_pattern = re.compile(r"^(?P<key>[^\t]+?)\s*[:\t]\s*(?P<value>.+)$")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Chromatogram Data:"):
            break
        match = metadata_pattern.match(line)
        if match:
            metadata[match.group("key").strip()] = match.group("value").strip()
    return metadata


def _read_chromeleon_txt(
    file_path: str | Path, dayfirst: bool = True
) -> tuple[chromparser.Injection, dict[str, str]]:
    """Read a Chromeleon text export with chromparser, plus its raw header."""
    path = Path(file_path)
    (injection,) = read_chromeleon_txt(path, dayfirst=dayfirst)
    return injection, _read_header(path)


def parse_chromeleon_txt(
    file_path: str | Path, dayfirst: bool = True
) -> tuple[dict, pd.DataFrame]:
    """
    Parses a txt file exported using Chromeleon software into a dict of metadata and a pd.DataFrame for chromatogram data.

    Reading is done by chromparser. The metadata contains the header keys as
    written by Chromeleon (e.g. ``"Signal Unit"``, ``"Channel"``), the parsed
    ``"Inject Time"`` as a naive `pd.Timestamp`, ``"time_unit"`` and the
    normalised snake_case keys of chromparser.

    Args:
        file_path (str | Path): Path to the chromatogram file.
        dayfirst (bool): Day/month order for ambiguous dates. Defaults to True.

    Returns:
        tuple[dict, pd.DataFrame]: Metadata and chromatogram data with columns
        ``"Time (min)"`` and ``"Value (<signal unit>)"``.
    """
    chrom = parse_chromatogram_txt(file_path, dayfirst=dayfirst)
    return chrom.metadata, chrom.data


def parse_inject_time(
    inject_time: str, metadata: dict, dayfirst: bool = True
) -> pd.Timestamp:
    """
    Parses the injection time of a Chromeleon txt file into a pd.Timestamp object.

    The format follows the locale of the exporting computer. Day/month order
    is read from the data when one component is >12, and dates with an AM/PM
    time are month first (US locale); otherwise `dayfirst` decides. When only a time is
    given, the date is taken from ``metadata["Injection Date"]``. Parsing is
    done by chromparser; any UTC offset is dropped, keeping the local wall time.

    Args:
        inject_time (str): The Inject Time string to parse.
        metadata (dict): Metadata dictionary, used for ``"Injection Date"``.
        dayfirst (bool): Day/month order used only when the date is genuinely
            ambiguous. Defaults to True.

    Returns:
        pd.Timestamp: Parsed, timezone-naive timestamp.

    Raises:
        ValueError: If the time cannot be parsed.
    """
    parsed = _parse_inject_time(inject_time, metadata.get("Injection Date"), dayfirst)
    return pd.Timestamp(parsed.replace(tzinfo=None))


def parse_chromatogram_txt(path: str | Path, dayfirst: bool = True) -> Chromatogram:
    """
    Parses a txt file exported using Chromeleon software into a Chromatogram object.

    Args:
        path (str | Path): Path to the chromatogram file.
        dayfirst (bool): Day/month order for ambiguous dates. Defaults to True.

    Returns:
        Chromatogram: Parsed Chromatogram object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a Chromeleon chromatogram export or has
            no parsable injection time.
    """
    path = Path(path)
    injection, header = _read_chromeleon_txt(path, dayfirst=dayfirst)

    def legacy(_signal: chromparser.Signal) -> dict:
        metadata = dict(header)
        metadata["Inject Time"] = injection_time
        return metadata

    signal = next(iter(injection.signals.values()))
    injection_time = to_timestamp(injection, signal)
    (chrom,) = to_chromatograms(
        injection,
        "chromeleon-txt",
        path,
        legacy_metadata=legacy,
        channel_name=header.get("Channel", "unknown"),
    )
    return chrom


def parse_to_channel(
    files: list[str | Path] | str | Path, channel_name: Optional[str] = None
) -> ChannelChromatograms:
    """
    Parses multiple chromatogram txt files into a ChannelChromatograms object.
    Takes either a directory path or a list of file paths.
    The chromatograms are loaded, sorted by the injection time, assigned a number, and added to the ChannelChromatograms object.

    Args:
        files (list[str | Path] | str | Path): List of file paths or a directory path containing chromatogram files.
        Channel (Optional[str]): Optional channel name to override the one in the metadata.

    Returns:
        ChannelChromatograms: Parsed ChannelChromatograms object containing all chromatograms.
    """
    if isinstance(files, (str, Path)):
        files = sorted(Path(files).iterdir())

    chromatograms = []
    channel = None
    for file in files:
        try:
            chrom = parse_chromatogram_txt(file)
            chromatograms.append(chrom)
            if channel is None:
                channel = chrom.channel
            elif channel != chrom.channel:
                log.critical(
                    f"Channel mismatch: {channel} vs {chrom.channel} in file {file}"
                )
        except Exception as e:
            log.warning(f"Failed to parse {file}: {e}")

    # Sort chromatograms by injection time
    chromatograms.sort(key=lambda x: x.injection_time)

    if not chromatograms:
        raise ValueError("No valid chromatograms were parsed.")
    # use the manual channel name, if not provided, use the one from the first chromatogram
    channel = channel_name if channel_name else chromatograms[0].channel

    # initialize ChannelChromatograms object
    channel_chroms = ChannelChromatograms(channel=channel)

    # adding chromatograms with injection number
    for i, chrom in enumerate(chromatograms, start=0):
        channel_chroms.add_chromatogram(i, chrom)

    return channel_chroms
