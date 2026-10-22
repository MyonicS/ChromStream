"""
Parser for binary Chromeleon 7 data: `.cmbx` archives and raw signal files.

A `.cmbx` file (Chromeleon "Export to archive") is a zip archive containing an
`header.xml` manifest, the serialized sequence (`*.cmd`) and one raw data file
per signal, audit trail and MS data item. Raw signal files can also be found
directly in the `RawData` folder of a Chromeleon data vault.

The format was reverse engineered from Chromeleon 7.3 files and verified point
for point against Chromeleon's own text export. Notes on the layout:

- Raw signal file: a sequence of chunks, each an 8-byte tag, the total chunk
  length (uint64, little-endian, including tag and length) and a payload.
    - `SignHdr\\0`: header with the acquisition start (Windows FILETIME, UTC),
      the channel name, the GUID of the injection and the Chromeleon version.
    - `PtsLL2Df`: a block of data points (about one minute each). Two float64
      scale factors (integer time ticks per minute and integer signal counts
      per signal unit) followed by signed varints `t0, v0, dt, dv` and
      `(ddt, ddv)` pairs: a second-order delta encoding of the integer time
      and signal. A "negative zero" varint terminates the block.
- Signed varints keep the sign in bit 6 of the first byte. The first byte is
  `[continue][sign][6 value bits]`, every following byte `[continue][7 value
  bits]`, least significant group first.
- `*.cmd`: protobuf-net serialized sequence. Top-level field 18 holds item
  records (type name, own GUID, parent GUID), field 19 the item data (linked
  by GUID). Injection and signal metadata are read from it.
- Audit trails (`CpAu`) and embedded XML blobs (`DIAC`/`DdAC`) are
  LZMA2-compressed XML and are not needed for the chromatogram.
- MS data (`MSRawItem`) is a Thermo `.raw` file and is not parsed.
"""

from __future__ import annotations

import logging as log
import struct
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from chromstream.objects import Chromatogram

_SIGNAL_MAGIC = b"SignHdr\x00"
_POINTS_TAG = b"PtsLL2Df"
_AUDIT_MAGIC = b"CpAu"
_THERMO_MAGIC = b"\x01\xa1" + "Finnigan".encode("utf-16-le")
_CHUNK_HEADER = struct.Struct("<8sQ")
# 100 ns intervals between 1601-01-01 (FILETIME epoch) and 1970-01-01
_FILETIME_UNIX_OFFSET = 116_444_736_000_000_000
_SIGNAL_ITEM_TYPE = "Dionex.Chromeleon.Data.Signal"
_INJECTION_ITEM_TYPE = "Dionex.Chromeleon.Data.Injection"
_SEQUENCE_ITEM_TYPE = "Dionex.Chromeleon.Data.Sequence"


# ---------------------------------------------------------------------------
# Raw signal files
# ---------------------------------------------------------------------------


def _read_svarint(buf: bytes, pos: int) -> tuple[int, int]:
    """Read one signed varint (sign in bit 6 of the first byte) at `pos`."""
    byte = buf[pos]
    pos += 1
    negative = byte & 0x40
    value = byte & 0x3F
    shift = 6
    while byte & 0x80:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
    return (-value if negative else value), pos


def _decode_svarints(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode a buffer of signed varints.

    Returns:
        Tuple of the decoded values (int64) and a boolean mask marking
        "negative zero" values, which Chromeleon uses as block terminators.
    """
    data = np.frombuffer(buf, dtype=np.uint8)
    if data.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
    is_last = data < 0x80
    if not is_last[-1]:
        raise ValueError("Truncated varint at the end of a data block.")
    ends = np.flatnonzero(is_last)
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts + 1
    if lengths.max() > 9:
        raise ValueError("Varint longer than 63 bits in data block.")
    # Position of every byte within its varint: the first byte carries 6
    # value bits, every following byte 7.
    pos_in_varint = np.arange(data.size) - np.repeat(starts, lengths)
    shifts = np.where(pos_in_varint == 0, 0, 7 * pos_in_varint - 1)
    bits = np.where(pos_in_varint == 0, data & 0x3F, data & 0x7F).astype(np.int64)
    magnitude = np.add.reduceat(bits << shifts, starts)
    negative = (data[starts] & 0x40) != 0
    return np.where(negative, -magnitude, magnitude), negative & (magnitude == 0)


def _iter_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Split a raw signal file into its (tag, payload) chunks."""
    chunks = []
    offset = 0
    while offset < len(data):
        if offset + _CHUNK_HEADER.size > len(data):
            raise ValueError(f"Truncated chunk header at offset {offset}.")
        tag, length = _CHUNK_HEADER.unpack_from(data, offset)
        if length < _CHUNK_HEADER.size or offset + length > len(data):
            raise ValueError(
                f"Invalid length {length} for chunk {tag!r} at offset {offset}."
            )
        chunks.append((tag, data[offset + _CHUNK_HEADER.size : offset + length]))
        offset += length
    return chunks


def _parse_signal_header(payload: bytes) -> dict:
    """Parse the `SignHdr` chunk.

    The field order was derived from Chromeleon 7.3 files. Fields whose meaning
    is unknown are skipped; failures only cost metadata, never signal data.
    """
    try:
        pos = 0
        values = []
        for _ in range(5):  # format version, ?, major, minor, revision
            value, pos = _read_svarint(payload, pos)
            values.append(value)
        filetime = struct.unpack_from("<q", payload, pos)[0]
        start = pd.Timestamp((filetime - _FILETIME_UNIX_OFFSET) * 100, unit="ns")
        pos += 8
        _, pos = _read_svarint(payload, pos)
        name_length, pos = _read_svarint(payload, pos)
        name = payload[pos : pos + name_length].decode("utf-8")
        pos += name_length
        injection_guid = uuid.UUID(bytes_le=payload[pos : pos + 16])
        pos += 16
        build, pos = _read_svarint(payload, pos)
    except (IndexError, struct.error, UnicodeDecodeError, ValueError) as e:
        log.warning(f"Could not parse Chromeleon signal header: {e}")
        return {}

    major, minor, revision = values[2:5]
    return {
        "Channel": name,
        "Acquisition Start (UTC)": start,
        "Injection GUID": str(injection_guid),
        "Generating Data System": f"Chromeleon {major}.{minor}.{revision}.{build}",
    }


def _decode_points_chunk(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode one `PtsLL2Df` chunk into time (min) and signal arrays."""
    if len(payload) < 16:
        raise ValueError("Truncated PtsLL2Df data block.")
    ticks_per_minute, counts_per_unit = struct.unpack_from("<dd", payload, 0)
    if ticks_per_minute <= 0 or counts_per_unit <= 0:
        raise ValueError(
            f"Invalid scale factors ({ticks_per_minute}, {counts_per_unit}) "
            "in PtsLL2Df data block."
        )
    values, terminators = _decode_svarints(payload[16:])
    end = np.flatnonzero(terminators)
    if end.size == 0:
        raise ValueError("PtsLL2Df data block has no end marker.")
    if end[0] != values.size - 1:
        raise ValueError("Unexpected data after the end of a PtsLL2Df data block.")
    values = values[: end[0]]

    if values.size == 0:
        return np.empty(0), np.empty(0)
    if values.size % 2:
        raise ValueError("PtsLL2Df data block has an incomplete data point.")
    # rows: (t0, v0), (dt, dv), then (ddt, ddv) for every further point
    pairs = values.reshape(-1, 2)
    steps = np.cumsum(pairs[1:], axis=0)
    points = np.vstack((pairs[:1], pairs[0] + np.cumsum(steps, axis=0)))
    return points[:, 0] / ticks_per_minute, points[:, 1] / counts_per_unit


def _decode_raw_signal(data: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode a Chromeleon raw signal file.

    Returns:
        Tuple of retention time (min), signal and header metadata.

    Raises:
        ValueError: If the data is not a Chromeleon signal or is malformed.
    """
    if not data.startswith(_SIGNAL_MAGIC):
        if data.startswith(_AUDIT_MAGIC):
            raise ValueError(
                "File is a Chromeleon audit trail (CpAu), not a signal. "
                "Chromatogram data is stored in the raw files starting with 'SignHdr'."
            )
        if data.startswith(_THERMO_MAGIC):
            raise ValueError(
                "File is a Thermo MS raw file, which is not supported by this parser."
            )
        raise ValueError("Not a Chromeleon signal file (missing 'SignHdr' header).")

    header: dict = {}
    times, signals = [], []
    for tag, payload in _iter_chunks(data):
        if tag == _SIGNAL_MAGIC:
            header = _parse_signal_header(payload)
        elif tag == _POINTS_TAG:
            time, signal = _decode_points_chunk(payload)
            times.append(time)
            signals.append(signal)
        else:
            raise ValueError(f"Unsupported chunk {tag!r} in Chromeleon signal file.")

    if not times:
        return np.empty(0), np.empty(0), header
    return np.concatenate(times), np.concatenate(signals), header


def _read_source(file_path, file_name=None) -> tuple[bytes, str]:
    """Read bytes from a path, a bytes object or a file-like object."""
    if isinstance(file_path, (bytes, bytearray)):
        return bytes(file_path), file_name or "unknown.raw"
    if hasattr(file_path, "read"):
        return file_path.read(), file_name or "unknown.raw"
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_bytes(), str(path)


def _build_chromatogram(
    data: bytes,
    path: Path | str,
    metadata: dict,
    channel_name: str | None,
    signal_unit: str | None,
    timezone: str | None,
) -> Chromatogram:
    time, signal, header = _decode_raw_signal(data)
    metadata = {**header, **metadata}

    if signal_unit:
        metadata["Signal Unit"] = signal_unit
    unit = metadata.get("Signal Unit")
    value_column = f"Value ({unit})" if unit else "Value"
    frame = pd.DataFrame({"Time (min)": time, value_column: signal})

    expected = metadata.pop("_expected_points", None)
    if expected is not None and expected != len(frame):
        log.warning(
            f"{path}: decoded {len(frame)} data points, but the sequence lists {expected}."
        )
    metadata["Data Points"] = len(frame)
    if len(frame):
        metadata["Time Min. (min)"] = float(time[0])
        metadata["Time Max. (min)"] = float(time[-1])
        metadata["Signal Min."] = float(signal.min())
        metadata["Signal Max."] = float(signal.max())
    if len(frame) > 1:
        steps = np.diff(time) * 60
        # rounded to 1 ns to drop float noise from the minute conversion
        metadata["Min. Step (s)"] = round(float(steps.min()), 9)
        metadata["Max. Step (s)"] = round(float(steps.max()), 9)
        metadata["Average Step (s)"] = round(float(steps.mean()), 9)
    metadata["time_unit"] = "min"

    injection_time = metadata.get("Inject Time")
    if injection_time is None:
        start = metadata.get("Acquisition Start (UTC)")
        if start is None:
            raise ValueError(f"No injection time found in {path}.")
        if timezone:
            start = start.tz_localize("UTC").tz_convert(timezone).tz_localize(None)
        injection_time = start
        metadata["Inject Time"] = injection_time

    channel = channel_name or metadata.get("Channel") or "unknown"
    return Chromatogram(
        data=frame,
        injection_time=pd.Timestamp(injection_time),
        metadata=metadata,
        channel=channel,
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
    data, path = _read_source(file_path, file_name)
    return _build_chromatogram(data, path, {}, channel_name, signal_unit, timezone)


# ---------------------------------------------------------------------------
# Sequence (.cmd) metadata
# ---------------------------------------------------------------------------


def _read_uvarint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, pos


def _pb_decode(buf: bytes) -> dict[int, list]:
    """Decode a protobuf message into {field number: [raw values]}."""
    fields: dict[int, list] = {}
    pos = 0
    while pos < len(buf):
        key, pos = _read_uvarint(buf, pos)
        wire_type = key & 7
        if wire_type == 0:
            value, pos = _read_uvarint(buf, pos)
        elif wire_type == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire_type == 2:
            length, pos = _read_uvarint(buf, pos)
            value, pos = buf[pos : pos + length], pos + length
        elif wire_type == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}.")
        if pos > len(buf):
            raise ValueError("Truncated protobuf message.")
        fields.setdefault(key >> 3, []).append(value)
    return fields


def _pb_get(message: dict, *path: int):
    """Return the first value at a path of nested field numbers, or None."""
    for depth, field_number in enumerate(path):
        values = message.get(field_number)
        if not values:
            return None
        if depth == len(path) - 1:
            return values[0]
        message = _pb_decode(values[0])
    return None


def _pb_str(message: dict, *path: int) -> str | None:
    value = _pb_get(message, *path)
    return value.decode("utf-8") if isinstance(value, bytes) else None


def _pb_guid(message: dict, field_number: int) -> uuid.UUID | None:
    """Decode a protobuf-net `bcl.Guid` (two fixed64 halves)."""
    value = _pb_get(message, field_number)
    if not isinstance(value, bytes):
        return None
    halves = _pb_decode(value)
    low = halves.get(1, [b"\x00" * 8])[0]
    high = halves.get(2, [b"\x00" * 8])[0]
    return uuid.UUID(bytes_le=low + high)


def _signal_metadata(record: dict) -> dict:
    signal = _pb_decode(_pb_get(record, 5, 3) or b"")
    expected_points = _pb_get(signal, 4)
    metadata = {
        "Signal Unit": _pb_str(signal, 6, 3),
        "Signal Quantity": _pb_str(signal, 6, 4),
        "Detector": _pb_str(signal, 11),
        "Driver Name": _pb_str(signal, 12),
        "_expected_points": expected_points
        if isinstance(expected_points, int)
        else None,
    }
    time_unit = _pb_str(signal, 5, 3)
    if time_unit and time_unit != "min":
        log.warning(f"Unexpected time axis unit {time_unit!r} in Chromeleon sequence.")
    return metadata


def _injection_metadata(record: dict) -> dict:
    metadata = {
        "Name": _pb_str(record, 19, 2),
        "Type": _pb_str(record, 19, 3, 3),
        "Status": _pb_str(record, 19, 4, 3),
        "Position": _pb_str(record, 19, 5),
        "Instrument Method": _pb_str(record, 40, 2, 5),
    }
    inject_time = _pb_str(record, 19, 7, 1)
    if inject_time:
        # ISO 8601 with offset; keep the local wall time like the .txt parser
        parsed = pd.Timestamp(inject_time)
        metadata["Inject Time"] = (
            parsed.tz_localize(None) if parsed.tzinfo is not None else parsed
        )
    return metadata


def _parse_sequence_file(data: bytes) -> dict[str, dict]:
    """Extract per-signal metadata from a `.cmd` sequence file.

    Returns:
        Mapping of raw data file id (lower case) to metadata of the signal and
        its injection.
    """
    top = _pb_decode(data)
    items: dict[uuid.UUID, tuple[str | None, uuid.UUID | None]] = {}
    for raw in top.get(18, []):
        record = _pb_decode(raw)
        guid = _pb_guid(record, 5)
        if guid is not None:
            items[guid] = (_pb_str(record, 3, 3), _pb_guid(record, 6))
    payloads = {}
    for raw in top.get(19, []):
        record = _pb_decode(raw)
        guid = _pb_guid(record, 25)
        if guid is not None:
            payloads[guid] = record

    signals = {}
    for guid, (item_type, parent) in items.items():
        if item_type != "Signal" or guid not in payloads:
            continue
        raw_id = _pb_str(payloads[guid], 5, 6)
        if not raw_id:
            continue
        metadata = _signal_metadata(payloads[guid])
        if parent in payloads:
            metadata.update(_injection_metadata(payloads[parent]))
        signals[raw_id.lower()] = {k: v for k, v in metadata.items() if v is not None}
    return signals


# ---------------------------------------------------------------------------
# .cmbx archives
# ---------------------------------------------------------------------------


def _data_vault(url: str) -> str | None:
    # chrom://<computer>/<data vault>/<folders...>
    parts = url.split("/")
    return parts[3] if url.startswith("chrom://") and len(parts) > 3 else None


def _iter_signal_elements(root: ET.Element):
    """Yield (signal, injection, sequence) elements from `header.xml`."""

    def walk(element, injection, sequence):
        for child in element:
            item_type = child.get("ItemType")
            if item_type == _SIGNAL_ITEM_TYPE:
                yield child, injection, sequence
            yield from walk(
                child,
                child if item_type == _INJECTION_ITEM_TYPE else injection,
                child if item_type == _SEQUENCE_ITEM_TYPE else sequence,
            )

    yield from walk(root, None, None)


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
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise ValueError(f"'{path}' is not a Chromeleon .cmbx archive: {e}") from e

    with archive:
        members = {name.lower(): name for name in archive.namelist()}
        if "header.xml" not in members:
            raise ValueError(
                f"'{path}' is not a Chromeleon .cmbx archive (no header.xml)."
            )
        root = ET.fromstring(archive.read(members["header.xml"]))

        sequence_cache: dict[str, dict[str, dict]] = {}

        def sequence_metadata(sequence) -> dict[str, dict]:
            name = sequence.get("Filename") if sequence is not None else None
            if not name or name.lower() not in members:
                return {}
            if name not in sequence_cache:
                try:
                    sequence_cache[name] = _parse_sequence_file(
                        archive.read(members[name.lower()])
                    )
                except Exception as e:
                    log.warning(f"Could not read sequence metadata from {name}: {e}")
                    sequence_cache[name] = {}
            return sequence_cache[name]

        chromatograms = []
        for signal, injection, sequence in _iter_signal_elements(root):
            raw_name = signal.get("RawDataFilename")
            if not raw_name or raw_name.lower() not in members:
                log.warning(
                    f"Raw data of signal {signal.get('Name')!r} missing in {path}."
                )
                continue
            url = signal.get("Url", "")
            metadata = {
                "URL": url,
                "Channel": signal.get("Name"),
                "Data Vault": _data_vault(url),
                "Sequence": sequence.get("Name") if sequence is not None else None,
                "Name": injection.get("Name") if injection is not None else None,
                "Raw Data File": raw_name,
            }
            raw_id = signal.get("RawDataFileId", "").lower()
            metadata.update(sequence_metadata(sequence).get(raw_id, {}))
            metadata = {k: v for k, v in metadata.items() if v is not None}

            chromatograms.append(
                _build_chromatogram(
                    archive.read(members[raw_name.lower()]),
                    path,
                    metadata,
                    channel_name=None,
                    signal_unit=None,
                    timezone=timezone,
                )
            )

    if not chromatograms:
        log.warning(f"No signals found in Chromeleon archive: {path}")
    return chromatograms
