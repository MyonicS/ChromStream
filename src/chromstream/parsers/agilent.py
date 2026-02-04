"""
Adapted from Matlab code found at:
https://github.com/chemplexity/chromatography/blob/master/Methods/Import/ImportAgilent.m
"""

import struct
import pandas as pd
import numpy as np
from pathlib import Path
from chromstream.objects import Chromatogram


def read_pascal_string(f, encoding="latin-1"):
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


def delta_compression(f, offset):
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


def double_delta_compression(f, offset):
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


def double_array(f, offset):
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
    # Using numpy for fast reading of doubles
    # Matlab uses little endian for DoubleArray ('double', 'l')
    data = np.fromfile(f, dtype="<d")
    return data.astype(float)


def parse_date(date_str):
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


def parse_agilent_ch(file_path):
    """
    Parses Agilent .ch files to a Chromatogram object.
    Supports versions 8, 81, 179, 181.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    metadata = {}

    with open(path, "rb") as f:
        # Read version string
        version = read_pascal_string(f, "latin-1")
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
                metadata[key] = read_pascal_string(f, encoding)

            tic = delta_compression(f, sig_offset)

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
                metadata[key] = read_pascal_string(f, encoding)

            tic = double_delta_compression(f, sig_offset)

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
                metadata[key] = read_pascal_string(f, encoding)

            f.seek(282)
            xmin = struct.unpack(">f", f.read(4))[0] / 60000.0
            xmax = struct.unpack(">f", f.read(4))[0] / 60000.0

            f.seek(4724)
            intercept = struct.unpack(">d", f.read(8))[0]
            slope = struct.unpack(">d", f.read(8))[0]

            if version == "179":
                tic = double_array(f, sig_offset)
            else:
                tic = double_delta_compression(f, sig_offset)

            tic = tic * slope + intercept

        else:
            raise ValueError(f"Unsupported Agilent version: {version}")

    # Create Time Array
    if len(tic) > 1:
        time = np.linspace(xmin, xmax, len(tic))
    else:
        time = np.array([])

    # Build DataFrame
    df = pd.DataFrame({"Time": time, "Signal": tic})

    # Parse Date
    injection_time = parse_date(metadata.get("date"))
    if pd.isna(injection_time):
        raise ValueError(f"Invalid injection time parsed from {path}")

    # Determine Channel from filename
    channel = path.stem

    # Ensure time_unit
    if "time_unit" not in metadata:
        metadata["time_unit"] = "min"

    return Chromatogram(
        data=df,
        injection_time=injection_time,
        metadata=metadata,
        channel=channel,
        path=str(path),
    )
