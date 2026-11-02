from __future__ import annotations

from pathlib import Path

import chromparser

from chromstream.objects import Chromatogram

from ._chromparser import to_chromatograms
from .agilent import parse_agilent_ch, parse_agilent_dot_d, parse_agilent_dx
from .chromeleon import parse_chromatogram_txt


def parse_chromatograms(path: str | Path, **options) -> list[Chromatogram]:
    """
    Parse every chromatogram (one per injection and signal) of a file.

    The format is detected from the file content. Reading of chromatogram
    files is done by chromparser (https://github.com/MyonicS/chromparser),
    which maintains the list of supported formats. MS data is not converted.

    Args:
        path (str | Path): File, or Agilent `.d` folder.
        **options: Reader options passed to `chromparser.read`, e.g.
            ``dayfirst`` (Chromeleon text exports) or ``signal_unit``
            (Chromeleon raw files).

    Returns:
        list[Chromatogram]: Retention time in minutes in the first column,
        signal in the second.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is not a supported chromatogram file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    fmt = chromparser.sniff(path)
    if fmt == "chromeleon-txt":
        return [parse_chromatogram_txt(path, dayfirst=options.get("dayfirst", True))]
    if fmt == "agilent-ch":
        return [parse_agilent_ch(path)]
    if fmt == "agilent-d":
        return parse_agilent_dot_d(path)
    if fmt == "agilent-dx":
        return parse_agilent_dx(path)

    chroms = []
    for injection in chromparser.read(path, ms=False, **options):
        chroms.extend(to_chromatograms(injection, fmt, path))
    if not chroms:
        raise ValueError(f"No chromatograms found in '{path}'.")
    return chroms


def parse_chromatogram(path: str | Path) -> Chromatogram:
    """
    Parse a single chromatogram file and infer parser type from file content.

    Supports every format of `parse_chromatograms` that holds exactly one
    chromatogram, e.g. Agilent `.ch` and Chromeleon `.txt` exports.

    Args:
        path (str | Path): Path to the chromatogram file.

    Returns:
        Chromatogram: The parsed chromatogram.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a supported chromatogram file or holds
            more than one chromatogram (use `parse_chromatograms`).
    """
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        chroms = parse_chromatograms(path)
    except chromparser.UnsupportedFormat as e:
        raise ValueError(f"Unsupported chromatogram file '{path}': {e}") from e
    except ValueError as e:
        raise ValueError(f"Failed to parse chromatogram file '{path}': {e}") from e

    if len(chroms) != 1:
        raise ValueError(
            f"'{path}' holds {len(chroms)} chromatograms; use parse_chromatograms()."
        )
    return chroms[0]
