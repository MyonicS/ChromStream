import pandas as pd
import re
import logging as log
from pathlib import Path
from chromstream.objects import Chromatogram
from chromstream.objects import ChannelChromatograms
from typing import Optional


def hello() -> str:
    return "parsers!"


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
        chromatogram_df = pd.read_csv(
            file_path,
            sep="	",
            skiprows=chromatogram_data_start,
            names=["Time (min)", "Step (s)", f"Value ({signal_unit})"],
            na_values=["n.a."],
            usecols=[
                "Time (min)",
                f"Value ({signal_unit})",
            ],  # makes it not read the Step column
            converters={
                "Time (min)": lambda x: float(
                    x.replace(".", "").replace(",", ".")
                    if "," in x and "." in x
                    else x.replace(",", "")
                ),
                f"Value ({signal_unit})": lambda x: float(
                    x.replace(".", "").replace(",", ".")
                    if "," in x and "." in x
                    else x.replace(",", "")
                ),
            },
        )
    else:
        log.warning(f"Chromatogram data section not found for {file_path}.")
        chromatogram_df = pd.DataFrame()
    return metadata, chromatogram_df


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
