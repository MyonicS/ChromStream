from __future__ import annotations

from pathlib import Path

from chromstream.objects import Chromatogram

from .agilent import parse_agilent_ch
from .chromeleon import parse_chromatogram_txt


def _is_chromeleon_txt(path: Path) -> bool:
    """
    Check whether a .txt file looks like a Chromeleon chromatogram export.
    """
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    head = "\n".join(lines[:120])
    markers = [
        "Injection Information:",
        "Chromatogram Data Information:",
        "Chromatogram Data:",
    ]
    has_data_marker = any(marker in head for marker in markers[1:])
    has_injection_section = markers[0] in head
    has_signal_unit = any(line.startswith("Signal Unit") for line in lines[:200])
    has_inject_time = any(
        line.startswith("Inject Time") or line.startswith("Injection Time")
        for line in lines[:200]
    )

    return (
        has_data_marker
        and has_injection_section
        and has_signal_unit
        and has_inject_time
    )


def parse_chromatogram(path: str | Path) -> Chromatogram:
    """
    Parse a single chromatogram file and infer parser type from file content/type.

    Supported:
    - Agilent: .ch
    - Chromeleon: .txt (validated by metadata signature)
    """
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".ch":
        try:
            return parse_agilent_ch(path)
        except Exception as e:
            raise ValueError(f"Failed to parse Agilent .ch file '{path}': {e}") from e

    if suffix == ".txt":
        if not _is_chromeleon_txt(path):
            raise ValueError(
                f"File '{path}' is .txt but does not match Chromeleon chromatogram metadata."
            )
        try:
            return parse_chromatogram_txt(path)
        except Exception as e:
            raise ValueError(
                f"Failed to parse Chromeleon .txt file '{path}': {e}"
            ) from e

    raise ValueError(
        f"Unsupported chromatogram file type '{suffix}' for '{path}'. Expected .ch or .txt."
    )
