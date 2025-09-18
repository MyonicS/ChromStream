"""
Data processing functions for chromatogram analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

if TYPE_CHECKING:
    from .objects import ChannelChromatograms, Chromatogram


def min_subtract(data: pd.DataFrame) -> pd.Series:
    """
    Simple minimum subtraction baseline correction

    Args:
        data: DataFrame containing time and signal columns

    Returns:
        Corrected signal as pandas Series
    """
    signal = data[data.columns[1]]
    return signal - signal.min()


def time_window_baseline(
    data: pd.DataFrame, time_window: tuple[float, float] = (0, 1)
) -> pd.Series:
    """
    Use mean of signal in a specific time window as baseline

    Args:
        data: DataFrame containing time and signal columns
        time_window: Tuple specifying the start and end time of the baseline window. Use the same unit as the chromatogram.

    Returns:
        Corrected signal as pandas Series
    """
    start_time, end_time = time_window
    time_col = data.columns[0]  # "Time (min)"
    signal_col = data.columns[1]

    # Find data points in the specified time window
    mask = (data[time_col] >= start_time) & (data[time_col] <= end_time)
    baseline_value = data.loc[mask, signal_col].mean()

    return data[signal_col] - baseline_value  # type: ignore[operator]


def time_point_baseline(data: pd.DataFrame, time_point: float) -> pd.Series:
    """
    Use signal value at a specific time point as baseline

    Args:
        data: DataFrame containing time and signal columns
        time_point: Time point to use as baseline reference. Use the same unit as the chromatogram.

    Returns:
        Corrected signal as pandas Series
    """
    time_col = data.columns[0]  # "Time (min)"
    signal_col = data.columns[1]

    # Find the closest data point to the specified time
    time_diff = (data[time_col] - time_point).abs()
    closest_index = time_diff.idxmin()
    baseline_value = data.loc[closest_index, signal_col]

    return data[signal_col] - baseline_value  # type: ignore[operator]


def integrate_single_chromatogram(
    chromatogram: Chromatogram, peaklist: dict, column: None | str = None
) -> dict:
    """
    Integrate the signal of a single chromatogram over time.

    Args:
        chromatogram: Chromatogram object containing the data to be analyzed
        peaklist: Dictionary defining the peaks to integrate. Example:
        ```
        Peaks_TCD = {"N2": [20, 26], "H2": [16, 19]}
        ```
        The list values must be in the same unit as the chromatogram.
        column: Optional column name to use for integration. If None, uses second column.

    Returns:
        Dictionary with integrated peak areas and timestamp
    """
    data = chromatogram.data
    time_col = data.columns[0]  # the time column must be the first!
    # need to implement handling of pd.datetime here

    signal_col = column if column is not None else data.columns[1]

    injection_result = {"Timestamp": chromatogram.injection_time}

    for peak_name, (start, end) in peaklist.items():
        # Create a mask for the time window
        mask = (data[time_col] >= start) & (data[time_col] <= end)

        ## Start Jan's peak baseline correction ##
        peak_signal_start = data[mask][signal_col].values[0]
        peak_signal_end = data[mask][signal_col].values[-1]

        baseline_signal = np.linspace(peak_signal_start, peak_signal_end, len(data[mask]))
        data_bsl = data.loc[mask, signal_col] - baseline_signal

        area_bsl = trapezoid(data_bsl, data.loc[mask, time_col])

        injection_result[peak_name + '_bsl'] = area_bsl
        ## End Jan's peak baseline correction ##

        area = trapezoid(data.loc[mask, signal_col], data.loc[mask, time_col])
        injection_result[peak_name] = area

    return injection_result


def integrate_channel(
    chromatogram: ChannelChromatograms, peaklist: dict, column: None | str = None
) -> pd.DataFrame:
    """
    Integrate the signal of a chromatogram over time.

    Args:
        chromatogram: ChannelChromatograms object containing the chromatograms to be analyzed
        peaklist: Dictionary defining the peaks to integrate. Example:
        ```
        Peaks_TCD = {"N2": [20, 26], "H2": [16, 19]}
        ```
        The list values must be in the same unit as the chromatogram.
        column: Optional column name to use for integration. If None, uses second column.
    Returns:
        DataFrame with integrated peak areas for each injection
    """

    results = []

    for chrom in chromatogram.chromatograms.values():
        injection_result = integrate_single_chromatogram(chrom, peaklist, column)
        results.append(injection_result)

    return pd.DataFrame(results)


def get_temp_and_valves_MTO(Integral_Frame, Log):
    """
    For a Dataframe containing chromatogram integrals and a timestamp column,
    add data from a log file.
    """
    integral_copy = Integral_Frame.copy()

    if "Timestamp" not in integral_copy.columns:
        integral_copy = integral_copy.reset_index().rename(
            columns={"index": "Timestamp"}
        )

    # Ensure both DataFrames are sorted by timestamp
    integral_copy = integral_copy.sort_values("Timestamp")
    Log = Log.sort_values("Timestamp")

    # Merge to get all log data at once
    result = pd.merge_asof(
        integral_copy,
        Log[["Timestamp", "Oven Temperature", "v10-bubbler", "v11-reactor"]],
        left_on="Timestamp",
        right_on="Timestamp",
        direction="nearest",
    )

    # Set timestamp as index and return
    return result.set_index("Timestamp")


def add_log_data(
    Integral_Frame: pd.DataFrame, Log: pd.DataFrame, columns: list[str] | all = "all"
) -> pd.DataFrame:
    """
     For a dataframe that contains a timestamp column, data from a log dataframe is added.
     The log dataframe must similarly contain a timestamp column.
     Args:
         Integral_Frame (pd.DataFrame): DataFrame containing e.g. chromatogram integrals.
         Log (pd.DataFrame): DataFrame containing log data with a timestamp column.
         columns (list[str] | 'all', optional): List of columns from the log to add. If 'all', all columns except timestamp are added. Defaults to 'all'.

    Returns:
        pd.DataFrame: DataFrame containing the original dataframe data with log data added.
    """

    # Data validation
    if "Timestamp" not in Integral_Frame.columns:
        raise ValueError("Integral_Frame must contain a 'Timestamp' column.")
    if "Timestamp" not in Log.columns:
        raise ValueError("Log must contain a 'Timestamp' column.")

    # check if the first timestamp of the log is after the first timestamp of the integral frame
    if Log["Timestamp"].min() > Integral_Frame["Timestamp"].max():
        raise ValueError(
            "The first timestamp of the log is after the last timestamp of the "
            "Integral_Frame. Check whether the right files are selected."
        )

    if Log["Timestamp"].max() < Integral_Frame["Timestamp"].min():
        raise ValueError(
            "The last timestamp of the log is before the first timestamp of the "
            "Integral_Frame. Check whether the right files are selected."
        )
    # Ensuring dfs are sorted by timestamp
    Integral_Frame = Integral_Frame.sort_values("Timestamp")
    Log = Log.sort_values("Timestamp")

    if columns == "all":
        # If 'all', add all columns except timestamp
        columns = [col for col in Log.columns if col != "Timestamp"]
    elif not isinstance(columns, list):
        raise ValueError("columns must be a list of column names or 'all'.")

    # Merging the dataframes
    merged = pd.merge_asof(
        Integral_Frame,
        Log[["Timestamp"] + columns],
        on="Timestamp",
        direction="nearest",
    )

    return merged


# To do - seperate integrate chrom function


# Splitting


def split_chromatogram(
    chromatogram: Chromatogram,
    n_injections: int,
    start_offset: int = 0,
    end_offset: int = 0,
    reset_time=True,
) -> list[Chromatogram]:
    """
    When multiple injections are contained in a single chromatogram, this function splits the chromatogram into multiple chromatograms
    Important constraint is the the length of the chromatogram must be divisible by the number of injections.
    The injection time of each split chromatogram is adjusted based on the runtime.
    Note:

    Args:
        chromatogram (Chromatogram): The chromatogram to be split.
        n_injections (int): The number of injections to split the chromatogram into.
        start_offset (int, optional): Number of data points to skip at the start of the chromatogram. Defaults to 0.
        end_offset (int, optional): Number of data points to skip at the end of the chromatogram. Defaults to 0.
        reset_time (bool, optional): Whether to reset the time column to start from 0 for each split chromatogram. Defaults to True.

    Returns:
        list[Chromatogram]: A list of split chromatograms.
    """
    end_index = len(chromatogram.data)
    chrom = (
        chromatogram.data.iloc[start_offset : (end_index - end_offset)]
        .reset_index(drop=True)
        .copy()
    )

    # Check if divisible by n_injections
    if len(chrom) % n_injections != 0:
        raise ValueError(
            f"Cannot split chromatograms, as length is not divisible by {n_injections}. Padding needs to be implemented."
        )

    # Calculate split indices, including the end of the data
    split_indices = [
        i * (len(chrom) // n_injections) for i in range(1, n_injections)
    ] + [len(chrom)]

    split_chromatograms = []
    last_index = 0

    for indx in split_indices:
        # Slice the data for the current segment
        data = chrom.iloc[last_index:indx].reset_index(drop=True).copy()
        last_index = indx

        # Adjust the time column (must be the first column)
        if chromatogram.time_unit == "min":
            injection_time = chromatogram.injection_time + pd.Timedelta(
                minutes=data[data.columns[0]].iloc[0]
            )
        elif chromatogram.time_unit == "s":
            injection_time = chromatogram.injection_time + pd.Timedelta(
                seconds=data[data.columns[0]].iloc[0]
            )
        else:
            raise ValueError(
                f"Unknown time unit {chromatogram.time_unit}, cannot split chromatogram."
            )

        if reset_time:
            # reset the time column to start from 0
            data[data.columns[0]] = (
                data[data.columns[0]] - data[data.columns[0]].iloc[0]
            )

        # Create a new Chromatogram object for the split segment
        from .objects import Chromatogram

        split_chromatogram = Chromatogram(
            data=data,
            injection_time=injection_time,
            metadata=chromatogram.metadata,
            channel=chromatogram.channel,
            path=chromatogram.path,
        )
        split_chromatograms.append(split_chromatogram)

    return split_chromatograms


def list_baseline_functions():
    baseline_functions = ["min_subtract", "time_window_baseline", "time_point_baseline"]
    return "\n".join(baseline_functions)
