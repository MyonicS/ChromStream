import pandas as pd
import re
import logging as log
from pathlib import Path
from chromstream.objects import Chromatogram
from chromstream.objects import ChannelChromatograms
from typing import Optional, Any
from datetime import datetime

# GC txt parsers


def parse_chromeleon_txt(file_path: str | Path) -> tuple[dict[str, str], pd.DataFrame]:
    """
    Parses a txt file exporeted using chromeleon software into a dict of metadata and pd.DataFrame for chromatogram data.

    Args:
        file_path (str | Path): Path to the chromatogram file.

    Returns:
        Tuple[Dict[str, str], pd.DataFrame]: A tuple containing metadata and chromatogram data as a DataFrame.
    """
    metadata = {}
    chromatogram_data_start = None

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Regular expression to match metadata lines
    metadata_pattern = re.compile(r"^(?P<key>[^\t]+?)\s*[:\t]\s*(?P<value>.+)$")

    # Parse metadata
    metadata_section = True
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if metadata_section:
            if line.startswith("Chromatogram Data:"):
                chromatogram_data_start = i + 2  # Skip the header line
                metadata_section = False
                continue
            match = metadata_pattern.match(line)
            if match:
                key = match.group("key").strip()
                value = match.group("value").strip()
                metadata[key] = value
        else:
            break

    # adding injection time as datetime object
    # Note: Some files use "Inject Time", others "Injection Time"
    inject_time_key = None
    if "Inject Time" in metadata:
        inject_time_key = "Inject Time"
    elif "Injection Time" in metadata:
        inject_time_key = "Injection Time"

    if inject_time_key is None:
        log.warning("Inject Time or Injection Time is missing from the metadata.")
    else:
        try:
            metadata["Inject Time"] = parse_inject_time(
                metadata[inject_time_key], metadata
            )
        except Exception as e:
            log.warning(f"Failed to parse '{inject_time_key}': {e}")

    # getting signal unit
    if "Signal Unit" not in metadata:
        log.warning("Signal Unit is missing from the metadata.")
        signal_unit = "unknown"
    else:
        signal_unit = metadata["Signal Unit"]

    # reading chromatogram data
    if chromatogram_data_start is not None:
        # Check first data line to determine number format
        with open(file_path, "r") as f:
            lines = f.readlines()

        first_data_line = lines[chromatogram_data_start].strip()
        comma_decimal_format = "," in first_data_line

        if comma_decimal_format:
            # European format: comma as decimal, dot as thousands
            converters = {
                "Time (min)": lambda x: float(str(x).replace(".", "").replace(",", "."))
                if pd.notna(x) and x != ""
                else float("nan"),
                f"Value ({signal_unit})": lambda x: float(
                    str(x).replace(".", "").replace(",", ".")
                )
                if pd.notna(x) and x != ""
                else float("nan"),
            }
        else:
            # Standard format: dot as decimal, comma as thousands
            converters = {
                "Time (min)": lambda x: float(str(x).replace(",", ""))
                if pd.notna(x) and x != ""
                else float("nan"),
                f"Value ({signal_unit})": lambda x: float(str(x).replace(",", ""))
                if pd.notna(x) and x != ""
                else float("nan"),
            }

        chromatogram_df = pd.read_csv(
            file_path,
            sep="	",
            skiprows=chromatogram_data_start,
            names=["Time (min)", "Step (s)", f"Value ({signal_unit})"],
            na_values=["n.a."],
            usecols=["Time (min)", f"Value ({signal_unit})"],
            converters=converters,
        )
        # adding time units to metadata
        metadata["time_unit"] = "min"
    else:
        log.warning(f"Chromatogram data section not found for {file_path}.")
        chromatogram_df = pd.DataFrame()

    # adding time unit to the metadata

    return metadata, chromatogram_df


### MTO ASCII parser


def parse_MTO_metadata(Path: str | Path) -> dict:
    """
    Parses the metadata section from an MTO ASCII file and returns it as a dictionary.

    Args:
        Path (str | Path): Path to the chromatogram file.

    Returns:
        dict: Dictionary containing metadata with cleaned keys and processed values.
    """
    df_chromatogram_meta = pd.read_csv(
        Path, sep="\t", header=None, skiprows=0, nrows=13
    )

    # Convert metadata DataFrame to dictionary
    metadata = {}
    for _, row in df_chromatogram_meta.iterrows():
        if pd.notna(row[0]):
            # Split on first comma to separate key from values
            parts = str(row[0]).split(",", 1)
            if len(parts) == 2:
                key = parts[0].strip().rstrip(":")
                value = parts[1].strip()
                # If there are multiple comma-separated values, keep as string or convert to list
                if "," in value:
                    # For values like sampling rates, convert to list of numbers where possible
                    try:
                        value_list = [
                            float(x.strip()) for x in value.split(",") if x.strip()
                        ]
                        metadata[key] = (
                            value_list if len(value_list) > 1 else value_list[0]
                        )
                    except ValueError:
                        # If conversion fails, keep as comma-separated string
                        metadata[key] = value
                else:
                    metadata[key] = value
    # adding time unit to metadata
    metadata["time_unit"] = "s"
    return metadata


def parse_MTO_asc(Path: str | Path) -> tuple[Chromatogram, Chromatogram, Chromatogram]:
    """
    Parses an ASCII file. from the MTO setup. The file contains the chromatograms for all channels under each other.

    Args:
        Path (str | Path): Path to the chromatogram file.

    Returns:
        The Chromatogram objects.
    """
    # Reads the ascii file of the injection. Splits in into a metadate frame and a frame containing chromatograms of the 3 channels.
    # Sampling frequency must be equal for all channels.

    df_chromatogram = pd.read_csv(Path, sep="\t", header=None, skiprows=13)
    metadata = parse_MTO_metadata(Path)
    # hard coding the signal unit as mV
    # The metadata contains a field "Y Axis Title"  'mVolts,mVolts,mVolts'
    metadata["Signal Unit"] = "mV"

    # split the chromatogram into the different columns. This is achieved by splitting the frame at indexes matching 1/3 and 2/3 of the lenght
    df_Channel_1 = df_chromatogram.iloc[0 : int(len(df_chromatogram) / 3)].reset_index(
        drop=True
    )
    df_Channel_2 = df_chromatogram.iloc[
        int(len(df_chromatogram) / 3) : int(2 * len(df_chromatogram) / 3)
    ].reset_index(drop=True)
    df_Channel_3 = df_chromatogram.iloc[
        int(2 * len(df_chromatogram) / 3) : len(df_chromatogram)
    ].reset_index(drop=True)

    # combine the 3 Channels into one frame, new index
    df_chromatogram = pd.concat([df_Channel_1, df_Channel_2, df_Channel_3], axis=1)
    sampling_freqs = [
        float(x.strip()) for x in metadata["Sampling Rate"].split(",")[0:2]
    ]

    if len(set(sampling_freqs)) != 1:
        raise ValueError("The sampling frequencies are not equal")

    sampling_freq = 1 / sampling_freqs[0]  # Hz

    df_chromatogram["Time[s]"] = df_chromatogram.index * sampling_freq

    # set time as index
    df_chromatogram = df_chromatogram.set_index("Time[s]")
    df_chromatogram["Time[s]"] = df_chromatogram.index

    # getting injection time
    inj_time = pd.to_datetime(metadata.get("Acquisition Date and Time", ""))

    df_chromatogram.columns = ["FID_L", "FID_M", "TCD", "Time[s]"]
    Chromatogram1 = Chromatogram(
        df_chromatogram[["Time[s]", "FID_L"]],
        injection_time=inj_time,
        metadata=metadata,
        channel="FID_L",
        path=Path,
    )
    Chromatogram2 = Chromatogram(
        df_chromatogram[["Time[s]", "FID_M"]],
        injection_time=inj_time,
        metadata=metadata,
        channel="FID_M",
        path=Path,
    )
    Chromatogram3 = Chromatogram(
        df_chromatogram[["Time[s]", "TCD"]],
        injection_time=inj_time,
        metadata=metadata,
        channel="TCD",
        path=Path,
    )
    return Chromatogram1, Chromatogram2, Chromatogram3


def parse_inject_time(inject_time: str, metadata: dict) -> pd.Timestamp:
    """
    Parses the injeciton time for chromeleon txt files into a pd.Timestamp object.
    The file most likely adopts the datatime format of the machine, meaning it can be very different between machines.
    In some formats, the date is saved seperatly from the datatime, and needs to be combined.

    Args:
        inject_time (pd.Timestamp): The Inject Time timestamp to parse.
        metadata (dict): Metadata dictionary containing additional information.

    Returns:
        pd.Timestamp: Parsed datetime object.
    """
    # Check for format like '7/17/2023 3:35:22 PM +02:00'
    if re.match(
        r"\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} (AM|PM) \+\d{2}:\d{2}", inject_time
    ):
        return pd.to_datetime(inject_time).tz_localize(None)

    # Check for format like '17-1-2023 16:45:42 +01:00'
    if re.match(r"\d{1,2}-\d{1,2}-\d{4} \d{2}:\d{2}:\d{2} \+\d{2}:\d{2}", inject_time):
        return pd.to_datetime(inject_time, format="%d-%m-%Y %H:%M:%S %z").tz_localize(
            None
        )

    # Check for format like '1:43:35 PM' and require metadata for the date
    if re.match(r"\d{1,2}:\d{2}:\d{2} (AM|PM)", inject_time):
        if "Injection Date" in metadata:
            injection_date = metadata["Injection Date"]
            combined_datetime = f"{injection_date} {inject_time}"
            return pd.to_datetime(combined_datetime, format="%m/%d/%Y %I:%M:%S %p")
        else:
            raise ValueError(
                "Injection Date is missing from metadata for AM/PM time format."
            )

    # Check for format like '10:30:07' (24-hour) and require metadata for the date
    if re.match(r"\d{2}:\d{2}:\d{2}$", inject_time):
        if "Injection Date" in metadata:
            injection_date = metadata["Injection Date"]
            # Check if date is in format like '22-Aug-25'
            if re.match(r"\d{1,2}-[A-Za-z]{3}-\d{2}", injection_date):
                combined_datetime = f"{injection_date} {inject_time}"
                return pd.to_datetime(combined_datetime, format="%d-%b-%y %H:%M:%S")
            # Check if date is in format like '12/19/2023'
            elif re.match(r"\d{1,2}/\d{1,2}/\d{4}", injection_date):
                combined_datetime = f"{injection_date} {inject_time}"
                return pd.to_datetime(combined_datetime, format="%m/%d/%Y %H:%M:%S")
            else:
                raise ValueError(
                    f"Unrecognized Injection Date format: {injection_date}"
                )
        else:
            raise ValueError(
                "Injection Date is missing from metadata for time-only Inject Time."
            )
    else:
        try:
            # Attempt to parse as ISO 8601 format
            time = pd.to_datetime(inject_time).tz_localize(None)
            log.info(f"Time format not handled, but succeeded parsing with: {time}")
            return time
        except Exception:
            pass
    raise ValueError(f"Unrecognized Inject Time format: {inject_time}")


# parsing to Chromatogram object


def parse_chromatogram_txt(path: str | Path) -> Chromatogram:
    """
    Parses a txt file exported using chromeleon software into a Chromatogram object.

    Args:
        path (str | Path): Path to the chromatogram file.

    Returns:
        Chromatogram: Parsed Chromatogram object.
    """
    metadata, df = parse_chromeleon_txt(path)
    injection_time = pd.Timestamp(metadata["Inject Time"])
    channel = metadata.get("Channel", "unknown")
    path = Path(path)

    return Chromatogram(
        data=df,
        injection_time=injection_time,
        metadata=metadata,
        channel=channel,
        path=path,
    )


# parsing multiple chromatograms from a list of files or a directory to a ChannelChromatograms object


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


### Log file parsers


def detect_log_type(file_path: str | Path) -> str:
    """
    Detect the type of log file based on its structure and content.

    Args:
        file_path: Path to the log file

    Returns:
        String indicating the log type ('FT', 'HTHPIR', 'LPIR', 'Robert', 'unknown')
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        lines = [
            line.strip() for line in content.split("\n")[:20]
        ]  # Read first 20 lines

    # Check for FT type - has many columns including MFC, valve info
    if any("MFC" in line and "Valve" in line for line in lines):
        return "FT"

    # Check for HTHPIR type - has C2H4/CH4 column and specific format
    if "C2H4/CH4" in content or any("C2H4/CH4" in line for line in lines):
        return "HTHPIR"

    # Check for LPIR type - has N2-bub column
    if any("N2-bub" in line for line in lines):
        return "LPIR"

    # Check for Robert type - has "Power Out %" and "Stirrer" columns
    if any("Power Out %" in line and "Stirrer" in line for line in lines):
        return "Robert"

    return "unknown"


def parse_metadata_section(lines: list) -> dict[str, Any]:
    """
    Parse the metadata section at the top of log files.

    Args:
        lines: List of lines from the file

    Returns:
        Dictionary containing metadata
    """
    metadata = {}

    for line in lines:
        if ":" in line and not line.startswith("Date/Time"):
            # Handle metadata lines like "Name: user" or "Date: 1/17/2023"
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        elif line.strip() and not any(char.isdigit() for char in line.split("\t")[0]):
            # Stop at data lines (which start with dates/times)
            continue
        else:
            break

    return metadata


def parse_log_MTO(file_path: str | Path) -> pd.DataFrame:
    Log = pd.read_csv(file_path, sep="\t", skiprows=1)
    Log = Log[
        [
            "Date",
            "Time",
            "MFC 1 pv",
            "MFC 2 pv",
            "MFC 3 pv",
            "MFC 4 pv",
            "Oven Temperature",
            "v11-reactor",
            "v10-bubbler",
            "v12-gc",
        ]
    ]
    # for the v-11 reactor columns, replace all 0 with 'reactor' else 'bypass'
    Log["v11-reactor"] = Log["v11-reactor"].apply(
        lambda x: "reactor" if x == 0 else "bypass"
    )
    Log.rename(columns={"MFC 1 pv": "N2_flow"}, inplace=True)
    Log.rename(columns={"MFC 4 pv": "He_Bubbler"}, inplace=True)
    Log.rename(columns={"MFC 3 pv": "He_Dilution"}, inplace=True)
    Log["Timestamp"] = pd.to_datetime(
        Log["Date"] + " " + Log["Time"], format="%m/%d/%Y %I:%M:%S %p"
    )
    Log["Timestamp"] = Log.apply(
        lambda row: datetime.strptime(
            row["Date"] + " " + row["Time"], "%m/%d/%Y %I:%M:%S %p"
        ),
        axis=1,
    )
    return Log


def parse_log_type_ft(file_path: str | Path) -> pd.DataFrame:
    """
    Parse FT type log files.

    Args:
        file_path: Path to the FT log file

    Returns:
        Parsed DataFrame with metadata as attributes
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # Find where the data starts
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Date/Time"):
            data_start_idx = i
            break

    if data_start_idx is None:
        raise ValueError("Could not find data section in FT log file")

    # Parse metadata
    metadata = parse_metadata_section(lines[:data_start_idx])

    # Read the data using pandas
    df = pd.read_csv(file_path, sep="\t", skiprows=data_start_idx)

    # Parse datetime
    df["Timestamp"] = pd.to_datetime(df["Date/Time"], format="%d-%b-%Y %H:%M:%S")
    df = df.drop("Date/Time", axis=1)

    # Add metadata as attributes
    df.attrs.update(metadata.items())  # ignore

    return df


def parse_log_type_hthpir(file_path: str | Path) -> pd.DataFrame:
    """
    Parse HTHPIR type log files.

    Args:
        file_path: Path to the HTHPIR log file

    Returns:
        Parsed DataFrame with metadata as attributes
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f.readlines()]  # Keep trailing tabs

    # Find where the data starts - look for actual data rows (start with date)
    data_start_idx = None
    header_lines = []

    for i, line in enumerate(lines):
        # Look for lines that start with a date pattern like "1/17/2023"
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}\t", line):
            data_start_idx = i
            break
        # Collect potential header lines that contain column information
        if (
            "Date" in line
            or "Time" in line
            or "Oven" in line
            or "N2" in line
            or "sp" in line
        ):
            header_lines.append(line)

    if data_start_idx is None:
        raise ValueError("Could not find data section in HTHPIR log file")

    # Parse metadata - everything before the data section
    metadata = parse_metadata_section(lines[:data_start_idx])

    # Manually construct the column names from the header lines
    # The HTHPIR format has split headers, so we need to reconstruct them
    column_names = [
        "Date",
        "Time",
        "Oven sp",
        "Oven temp",
        "Oven ramp",
        "N2 sp",
        "N2 flow",
        "H2 sp",
        "H2 flow",
        "CO2 sp",
        "CO2 flow",
        "C2H4/CH4 sp",
        "C2H4/CH4 flow",
        "O2 sp",
        "O2 pv",
        "Pressure sp",
        "Pressure pv",
    ]

    # Read the data section manually
    data_rows = []
    for line in lines[data_start_idx:]:
        if line.strip():  # Skip empty lines
            row = line.split("\t")
            data_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=column_names[: len(data_rows[0])])

    # Convert numeric columns
    for col in df.columns:
        if col not in ["Date", "Time"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Combine Date and Time columns - handle AM/PM format
    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str)
    )
    df = df.drop(["Date", "Time"], axis=1)

    # Add metadata as attributes
    df.attrs.update(metadata.items())

    return df


def parse_log_type_lpir(file_path: str | Path) -> pd.DataFrame:
    """
    Parse LPIR type log files.

    Args:
        file_path: Path to the LPIR log file

    Returns:
        Parsed DataFrame with metadata as attributes
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # Find where the data starts
    data_start_idx = None
    for i, line in enumerate(lines):
        if "Date\tTime" in line:
            data_start_idx = i
            break

    if data_start_idx is None:
        raise ValueError("Could not find data section in LPIR log file")

    # Parse metadata
    metadata = parse_metadata_section(lines[:data_start_idx])

    # Read the data using pandas
    df = pd.read_csv(file_path, sep="\t", skiprows=data_start_idx)

    # Combine Date and Time columns
    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str)
    )
    df = df.drop(["Date", "Time"], axis=1)

    # Add metadata as attributes
    df.attrs.update(metadata.items())  # ignore

    return df


def parse_log_type_robert(file_path: str | Path) -> pd.DataFrame:
    """
    Parse Robert type log files.

    Args:
        file_path: Path to the Robert log file

    Returns:
        Parsed DataFrame with metadata as attributes
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # Find where the data starts
    data_start_idx = None
    for i, line in enumerate(lines):
        if "Date\tTime" in line:
            data_start_idx = i
            break

    if data_start_idx is None:
        raise ValueError("Could not find data section in Robert log file")

    # Parse metadata
    metadata = parse_metadata_section(lines[:data_start_idx])

    # Read the data using pandas
    df = pd.read_csv(file_path, sep="\t", skiprows=data_start_idx)

    # Combine Date and Time columns
    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str)
    )
    df = df.drop(["Date", "Time"], axis=1)

    # Add metadata as attributes
    df.attrs.update(metadata.items())  # ignore

    return df


def parse_log_file(file_path: str | Path) -> pd.DataFrame:
    """
    Automatically detect and parse any supported log file type. To be extended.

    Args:
        file_path: Path to the log file

    Returns:
        Parsed DataFrame with metadata as attributes

    Raises:
        ValueError: If the file type is not recognized or supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    log_type = detect_log_type(file_path)

    if log_type == "FT":
        return parse_log_type_ft(file_path)
    elif log_type == "HTHPIR":
        return parse_log_type_hthpir(file_path)
    elif log_type == "LPIR":
        return parse_log_type_lpir(file_path)
    elif log_type == "Robert":
        return parse_log_type_robert(file_path)
    else:
        raise ValueError(
            f"Unsupported or unrecognized log file type: {log_type}. Parse the data manually."
        )
