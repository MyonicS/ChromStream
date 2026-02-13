"""
Agilent binary parser adapted from Matlab code found at:
https://github.com/chemplexity/chromatography/blob/master/Methods/Import/ImportAgilent.m
"""

import struct
import pandas as pd
import numpy as np
from pathlib import Path
from chromstream.objects import Chromatogram
import os
import logging as log
import zipfile
import xml.etree.ElementTree as ET


def _read_pascal_string(f, encoding="latin-1"):
    """Read a pascal string (length byte + chars)"""
    length_byte = f.read(1)
    if not length_byte:
        return ""
    length = struct.unpack("<B", length_byte)[0]
    if length == 0:
        return ""

    if encoding == "utf-16-le":
        string_data = f.read(length * 2)
    else:
        string_data = f.read(length)

    try:
        return string_data.decode(encoding).strip()
    except UnicodeDecodeError:
        return string_data.decode("latin-1", errors="replace").strip()


def _delta_compression(f, offset):
    """
    Decodes Delta Compressed signal (Version 8)
    """
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(offset)

    signals = []
    val = 0

    while f.tell() < file_size:
        header_bytes = f.read(2)
        if len(header_bytes) < 2:
            break
        header = struct.unpack(">h", header_bytes)[0]

        count = header & 0x0FFF
        if (header >> 12) == 0:
            break

        for _ in range(count):
            delta_bytes = f.read(2)
            if len(delta_bytes) < 2:
                break
            delta = struct.unpack(">h", delta_bytes)[0]

            if delta != -32768:
                val += delta
            else:
                val_bytes = f.read(4)
                if len(val_bytes) < 4:
                    break
                val = struct.unpack(">i", val_bytes)[0]

            signals.append(val)

    return np.array(signals, dtype=float)


def _double_delta_compression(f, offset):
    """
    Decodes Double Delta Compressed signal (Version 81, 181)
    """
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(offset)

    signals = []
    sig = 0
    delta = 0

    while f.tell() < file_size:
        val_bytes = f.read(2)
        if len(val_bytes) < 2:
            break
        val = struct.unpack(">h", val_bytes)[0]

        if val != 32767:
            delta += val
            sig += delta
        else:
            high_bytes = f.read(2)
            low_bytes = f.read(4)
            if len(high_bytes) < 2 or len(low_bytes) < 4:
                break
            high = struct.unpack(">h", high_bytes)[0]
            low = struct.unpack(">I", low_bytes)[0]
            sig = (high * 4294967296) + low
            delta = 0

        signals.append(sig)

    return np.array(signals, dtype=float)


def _double_array(f, offset):
    """
    Decodes Double Array signal (Version 179)
    """
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(offset)
    count = (file_size - offset) // 8
    if count <= 0:
        return np.array([])

    f.seek(offset)
    # Read bytes and convert to numpy array
    # Works with both real files and file-like objects from zipfile
    bytes_to_read = count * 8
    data_bytes = f.read(bytes_to_read)
    data = np.frombuffer(data_bytes, dtype="<d")
    return data.astype(float)


def _parse_date(date_str):
    if not date_str:
        return pd.NaT
    formats = [
        "%d %b %y %I:%M %p",
        "%m/%d/%y %I:%M:%S %p",
        "%d-%b-%y, %H:%M:%S",
        "%d %b %y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    try:
        from dateutil import parser

        return pd.to_datetime(parser.parse(date_str))
    except (ImportError, ValueError):
        pass
    return pd.to_datetime(date_str, errors="coerce")


def _prepare_file_input(file_path, file_name=None, channel_name=None):
    """
    Prepare file input for parsing, handling both file paths and file-like objects.
    File-like objects are required for direct reading of .dx files, which are zip archives.

    Args:
        file_path: Path to file or file-like object
        file_name: Optional filename for file-like objects
        channel_name: Optional channel name override

    Returns:
        Tuple of (file_object, channel, path_str, should_close)
    """
    if hasattr(file_path, "read"):
        # It's a file-like object
        f = file_path
        should_close = False
        channel = (
            channel_name
            if channel_name
            else (Path(file_name).stem if file_name else "unknown")
        )
        path_str = file_name if file_name else "unknown.ch"
    else:
        # It's a path
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        f = open(path, "rb")
        should_close = True
        channel = channel_name if channel_name else path.stem
        path_str = str(path)

    return f, channel, path_str, should_close


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
    """
    f, channel, path_str, should_close = _prepare_file_input(
        file_path, file_name, channel_name
    )

    metadata = {}

    try:
        # Read version string
        version = _read_pascal_string(f, "latin-1")
        metadata["version"] = version

        tic = np.array([])
        xmin = 0.0
        xmax = 0.0

        if version in ["8", "80"]:
            offsets = {
                "Sample Name": 24,
                "Sample Description": 86,
                "Method Name": 228,
                "Operator": 148,
                "date": 178,
                "Instrument": 218,
                "Inlet": 208,
                "Signal Unit": 580,
            }
            encoding = "latin-1"

            f.seek(264)
            sig_offset_raw = struct.unpack(">i", f.read(4))[0]
            sig_offset = (sig_offset_raw - 1) * 512

            for key, off in offsets.items():
                f.seek(off)
                metadata[key] = _read_pascal_string(f, encoding)

            tic = _delta_compression(f, sig_offset)

            f.seek(282)
            xmin = struct.unpack(">i", f.read(4))[0] / 60000.0
            xmax = struct.unpack(">i", f.read(4))[0] / 60000.0

            f.seek(542)
            header = struct.unpack(">i", f.read(4))[0]
            if header in [1, 2, 3]:
                tic = tic * 1.33321110047553
            else:
                f.seek(636)
                intercept = struct.unpack(">d", f.read(8))[0]
                slope = struct.unpack(">d", f.read(8))[0]
                tic = tic * slope + intercept

        elif version == "81":
            offsets = {
                "Sample Name": 24,
                "Sample Description": 86,
                "Method Name": 228,
                "Operator": 148,
                "date": 178,
                "Instrument": 218,
                "Inlet": 208,
                "Signal Unit": 580,
            }
            encoding = "latin-1"

            f.seek(264)
            sig_offset_raw = struct.unpack(">i", f.read(4))[0]
            sig_offset = (sig_offset_raw - 1) * 512

            for key, off in offsets.items():
                f.seek(off)
                metadata[key] = _read_pascal_string(f, encoding)

            tic = _double_delta_compression(f, sig_offset)

            f.seek(282)
            xmin = struct.unpack(">f", f.read(4))[0] / 60000.0
            xmax = struct.unpack(">f", f.read(4))[0] / 60000.0

            f.seek(636)
            intercept = struct.unpack(">d", f.read(8))[0]
            slope = struct.unpack(">d", f.read(8))[0]
            tic = tic * slope + intercept

        elif version in ["179", "181"]:
            offsets = {
                "Sample Name": 858,
                "Sample Description": 1369,
                "Method Name": 2574,
                "Operator": 1880,
                "date": 2391,
                "Instrument": 2533,
                "Inlet": 2492,
                "Signal Unit": 4172,
            }
            encoding = "utf-16-le"

            f.seek(264)
            sig_offset_raw = struct.unpack(">i", f.read(4))[0]
            sig_offset = (sig_offset_raw - 1) * 512

            for key, off in offsets.items():
                f.seek(off)
                metadata[key] = _read_pascal_string(f, encoding)

            f.seek(282)
            xmin = struct.unpack(">f", f.read(4))[0] / 60000.0
            xmax = struct.unpack(">f", f.read(4))[0] / 60000.0

            f.seek(4724)
            intercept = struct.unpack(">d", f.read(8))[0]
            slope = struct.unpack(">d", f.read(8))[0]

            if version == "179":
                tic = _double_array(f, sig_offset)
            else:
                tic = _double_delta_compression(f, sig_offset)

            tic = tic * slope + intercept

        else:
            raise ValueError(f"Unsupported Agilent version: {version}")

    finally:
        if should_close:
            f.close()

    # Create Time Array
    if len(tic) > 1:
        time = np.linspace(xmin, xmax, len(tic))
    else:
        time = np.array([])

    # Build DataFrame
    df = pd.DataFrame({"Time": time, "Signal": tic})

    # Parse Date
    injection_time = _parse_date(metadata.get("date"))
    if pd.isna(injection_time):
        raise ValueError(f"Invalid injection time parsed from {path_str}")

    # Ensure time_unit
    if "time_unit" not in metadata:
        metadata["time_unit"] = "min"

    return Chromatogram(
        data=df,
        injection_time=injection_time,
        metadata=metadata,
        channel=channel,
        path=path_str,
    )


def chromlist_from_dot_d(path_dir: Path) -> list[Chromatogram]:
    """
    Given a path to a Chromeleon .d directory, parses all chromatogram files
    and returns a list of Chromatogram objects.

    Args:
        path_dir (str | Path): Path to the .d directory.

    Returns:
        list[Chromatogram]: List of parsed Chromatogram objects.
    """
    # if the dir doesn't end with .d or is nto a directory, raise an error
    if not path_dir.is_dir() or not path_dir.name.lower().endswith(".d"):
        raise ValueError(f"Provided path is not a valid .d directory: {path_dir}")

    chrom_list = []
    for file in os.listdir(path_dir):
        if file.endswith(".ch"):
            chrom_path = path_dir / file
            chrom = parse_agilent_ch(chrom_path)
            chrom_list.append(chrom)
    if len(chrom_list) == 0:
        log.warning(f"No .ch files found in directory: {path_dir}")
    return chrom_list


def _parse_acmd_channel_mapping(dx_open: zipfile.ZipFile) -> dict[str, str]:
    """
    Parse .acmd XML file from a .dx archive to extract TraceId -> ChannelName mapping.

    Args:
        dx_open: Open ZipFile object

    Returns:
        Dictionary mapping TraceId to ChannelName
    """
    channel_map = {}
    for file in dx_open.namelist():
        if file.lower().endswith(".acmd"):
            try:
                with dx_open.open(file) as acmd_file:
                    tree = ET.parse(acmd_file)
                    root = tree.getroot()
                    # Define namespace
                    ns = {"acmd": "urn:schemas-agilent-com:acmd20"}
                    # Extract signal information
                    for signal in root.findall(".//acmd:Signal", ns):
                        trace_id = signal.find("acmd:TraceId", ns)
                        channel_name = signal.find("acmd:ChannelName", ns)
                        if trace_id is not None and channel_name is not None:
                            channel_map[trace_id.text] = channel_name.text
            except Exception as e:
                log.warning(f"Failed to parse .acmd file: {e}")
            break
    return channel_map


def parse_agilent_dx(file_path) -> list[Chromatogram]:
    """
    Parses Agilent .dx files to a list of Chromatogram objects.

    Args:
        file_path (str | Path): Path to the .dx file.

    Returns:
        list[Chromatogram]: List of parsed Chromatogram objects.
    """
    # check if file is a .dx file
    path = Path(file_path)
    if not path.exists() or not path.is_file() or not path.suffix.lower() == ".dx":
        raise ValueError(f"Provided path is not a valid .dx file: {file_path}")

    # trying to unzip
    with zipfile.ZipFile(path, "r") as dx_open:
        # Parse .acmd file to get channel names
        channel_map = _parse_acmd_channel_mapping(dx_open)

        chrom_list = []
        for file in dx_open.namelist():
            if file.lower().endswith(".ch"):
                with dx_open.open(file) as f:
                    # Try to match filename to channel name
                    # .ch files are typically named with TraceId
                    file_stem = Path(file).stem
                    channel_name = channel_map.get(file_stem) if channel_map else None

                    chrom = parse_agilent_ch(
                        f, file_name=file, channel_name=channel_name
                    )
                    chrom_list.append(chrom)

        if len(chrom_list) == 0:
            log.warning(f"No .ch files found in .dx archive: {file_path}")
        return chrom_list
