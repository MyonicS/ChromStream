"""
Data processing functions for chromatogram analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from chromstream.objects import ChannelChromatograms
from scipy.integrate import trapezoid


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

    return data[signal_col] - baseline_value


def integrate_channel(
    chromatogram: ChannelChromatograms, peaklist: dict
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
    Returns:
        DataFrame with integrated peak areas for each injection
    """
    results = []

    for inj_num, chrom in chromatogram.chromatograms.items():
        data = chrom.data
        time_col = data.columns[0]  # the time column must be the first!
        signal_col = data.columns[1]
        injection_result = {"Timestamp": chrom.injection_time}

        for peak_name, (start, end) in peaklist.items():
            # Create a mask for the time window
            mask = (data[time_col] >= start) & (data[time_col] <= end)

            area = trapezoid(data.loc[mask, signal_col], data.loc[mask, time_col])
            injection_result[peak_name] = area

        results.append(injection_result)

    return pd.DataFrame(results)
