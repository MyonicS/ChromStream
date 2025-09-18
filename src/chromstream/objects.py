from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from chromstream.data_processing import integrate_channel
import logging as log


@dataclass
class Chromatogram:
    """Single chromatogram data for one injection on one channel"""

    data: pd.DataFrame
    injection_time: pd.Timestamp
    metadata: dict
    channel: str
    path: Path | str

    # extra properties
    @property
    def time_unit(self) -> str:
        """Get the time unit from metadata, default to 'min' if not found"""
        if "time_unit" in self.metadata:
            return self.metadata["time_unit"]
        else:
            log.warning("Time unit not found in metadata")
            return "unknown"

    @property
    def signal_unit(self) -> str:
        if "Signal Unit" in self.metadata:
            return self.metadata["Signal Unit"]
        else:
            log.warning("Signal unit not found in metadata")
            return "unknown"

    def plot(self, ax=None, column=None, **kwargs):
        """Plot the chromatogram data"""
        if ax is None:
            fig, ax = plt.subplots()

        # Choose which column to plot (default to second column)
        y_column = self.data.columns[1] if column is None else column
        x_column = self.data.columns[0]

        ax.plot(self.data[x_column], self.data[y_column], **kwargs)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"Chromatogram - {self.channel} - {self.path}")

        return ax

    def apply_baseline(
        self, correction_func, inplace=False, suffix="_BLcorr", **kwargs
    ):
        """
        Apply baseline correction to the chromatogram data

        Args:
            correction_func: Function that takes a pandas DataFrame and returns corrected Series
            inplace (bool): If True, modify the original data. If False, add new column
            suffix (str): Suffix to add to the new column name when inplace=False

        Returns:
            pd.DataFrame: The corrected data (same as self.data if inplace=True)
        """
        signal_column = self.data.columns[1]  # Second column (signal data)

        # Apply the correction function - passes entire DataFrame
        corrected_signal = correction_func(self.data, **kwargs)

        if inplace:
            self.data[signal_column] = corrected_signal
        else:
            new_column_name = signal_column + suffix
            self.data[new_column_name] = corrected_signal

        return self.data

    def integrate_peaks(self, peaklist: dict, column: None | str = None) -> dict:
        """
        Integrate peaks for this chromatogram

        Args:
            peaklist: Dictionary defining the peaks to integrate. Example:
            Peaks_TCD = {"N2": [20, 26], "H2": [16, 19]}
            The list values must be in the same unit as the chromatogram.
            column: Optional column name to use for integration. If None, uses second column.

        Returns:
            Dictionary with integrated peak areas and timestamp
        """
        from .data_processing import integrate_single_chromatogram

        return integrate_single_chromatogram(self, peaklist, column=column)


@dataclass
class ChannelChromatograms:
    """
    Contains data of a single channel with multiple chromatograms

    Args:
        channel: Name of the channel (e.g., 'FID', 'TCD')
        chromatograms: Dictionary mapping chromatogram number to Chromatogram objects
        integrals: DataFrame containing integrated peak areas for each chromatogram (optional)

    Methods:
        add_chromatogram: Add a chromatogram to the channel
        plot: Plot all chromatograms in the channel
        integrate_peaks: Integrate peaks for all chromatograms in the channel, requieres dict of peak limits

    """

    channel: str  # 'FID', 'TCD', etc.
    chromatograms: dict[int, Chromatogram] = field(default_factory=dict)
    integrals: pd.DataFrame | None = None

    def add_chromatogram(self, injection_num: int, chromatogram: Chromatogram):
        """Add a chromatogram for a specific injection"""
        self.chromatograms[injection_num] = chromatogram

    def plot(self, ax=None, colormap="viridis", plot_colorbar=True, **kwargs):
        """Plotting all chromatograms of a channel channel"""
        if ax is None:
            fig, ax = plt.subplots()
        colormap = plt.get_cmap(colormap)
        colors = colormap(np.linspace(0, 1, len(self.chromatograms)))

        for inj_num, chrom in self.chromatograms.items():
            ax.plot(
                chrom.data[chrom.data.columns[0]],
                chrom.data[chrom.data.columns[1]],
                label=f"Injection {inj_num}",
                color=colors[inj_num],
                **kwargs,
            )

        ax.set_xlabel(chrom.data.columns[0])  # type: ignore
        ax.set_ylabel(chrom.data.columns[1])  # type: ignore
        ax.set_title(f"Channel: {self.channel}")
        # add colorbar
        if plot_colorbar:
            sm = plt.cm.ScalarMappable(
                norm=Normalize(vmin=0, vmax=len(self.chromatograms) - 1)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Injection Number")

        return ax

    plot_chromatograms = plot  # alias

    def apply_baseline(
        self, correction_func, inplace=False, suffix="_BLcorr", **kwargs
    ):
        """
        Apply baseline correction to all chromatograms in the channel

        Args:
            correction_func: Function that takes a pandas DataFrame and returns corrected Series
            inplace (bool): If True, modify the original data. If False, add new column
            suffix (str): Suffix to add to the new column name when inplace=False
            **kwargs: Additional arguments to pass to the correction function

        Returns:
            None: Modifies chromatograms in place
        """
        for chrom in self.chromatograms.values():
            chrom.apply_baseline(
                correction_func, inplace=inplace, suffix=suffix, **kwargs
            )

    def integrate_peaks(
        self, peaklist: dict, column: None | str = None, per_peak_baseline: bool = False
    ) -> pd.DataFrame:
        """
        Integrate peaks for all chromatograms in the channel

        Args:
            peaklist: Dictionary defining the peaks to integrate. Example:
            Peaks_TCD = {"N2": [20, 26], "H2": [16, 19]}

            The list values must be in the same unit as the chromatogram.

            column: Optional column name to use for integration. If None, uses second column.

        Returns:
            DataFrame with integrated peak areas for each injection
        """
        self.integrals = integrate_channel(self, peaklist, column=column, per_peak_baseline=per_peak_baseline)
        return self.integrals


@dataclass
class Experiment:
    """Data for a single experiment containing multiple on-line GC channels"""

    name: str
    channels: dict[str, ChannelChromatograms] = field(default_factory=dict)
    experiment_starttime: pd.Timestamp | None = None
    experiment_endtime: pd.Timestamp | None = None
    log: pd.DataFrame | None = None

    # Methods
    @property
    def channel_names(self) -> list[str]:
        """Get a list of channel names in the experiment"""
        return list(self.channels.keys())

    def add_channel(self, channel_name: str, channel_data: ChannelChromatograms):
        """Add a channel to the experiment"""
        self.channels[channel_name] = channel_data

    def add_chromatogram(
        self, chromatogram: Path | str | Chromatogram, channel_name: str | None = None
    ):
        """Add a chromatogram to the experiment, automatically creating the channel if it does not exist

        Args:
            chromatogram (Path | str | Chromatogram): Path to the chromatogram file or a Chromatogram object
            channel_name (Optional[str], optional): Optional channel name to override


        """
        if isinstance(chromatogram, (str, Path)):
            from .parsers import parse_chromatogram_txt

            chrom = parse_chromatogram_txt(chromatogram)
        elif isinstance(chromatogram, Chromatogram):
            chrom = chromatogram
        else:
            raise ValueError(
                "chromatogram must be a file path or a Chromatogram object"
            )

        channel = channel_name if channel_name else chrom.channel

        if channel not in self.channels:
            self.channels[channel] = ChannelChromatograms(channel=channel)

        injection_num = len(self.channels[channel].chromatograms)
        self.channels[channel].add_chromatogram(injection_num, chrom)

    def plot_chromatograms(self, ax=None, channels: str | list = "all", **kwargs):
        if ax is None:
            n_channels_to_plot = (
                len(self.channels) if channels == "all" else len(channels)
            )

            fig, ax = plt.subplots(
                n_channels_to_plot,
                1,
                # figsize=(7, 3.3 / 1.618 * n_channels_to_plot),
                tight_layout=True,
            )
            if n_channels_to_plot == 1:
                ax = [ax]
        if channels == "all":
            channels = list(self.channels.keys())
        for i, channel in enumerate(channels):
            if channel in self.channels:
                self.channels[channel].plot(ax=ax[i], **kwargs)
            else:
                raise ValueError(f"Channel {channel} not found in experiment.")

    def add_log(self, log: str | Path | pd.DataFrame):
        """
        Adds a log dataframe to the experiment, either from a dataframe or from a path to the log file.

        Args:
            log (str | Path | pd.DataFrame): Path to the log file or a DataFrame
        """
        if isinstance(log, (str, Path)):
            from .parsers import parse_log_file

            self.log = parse_log_file(log)
        elif isinstance(log, pd.DataFrame):
            self.log = log
        else:
            raise ValueError("log must be a file path or a DataFrame")

    def plot_log(self, columns: str | list, ax=None, use_exp_time=False):
        """
        Plots specified colums of the experiment log. If use_exp_time is True, the x-axis will be the time since the start of the experiment in minutes.
        Args:
            columns (str | list): Column name or list of column names to plot
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes will be created.
            use_exp_time (bool, optional): Whether to use time since start of experiment as x-axis. Defaults to False.
        """

        if self.log is None:
            raise ValueError("No log data available to plot.")

        if ax is None:
            fig, ax = plt.subplots()

        if isinstance(columns, str):
            columns = [columns]

        if use_exp_time:
            if self.experiment_starttime is None:
                raise ValueError(
                    "Experiment start time is not set. Cannot use experiment time."
                )
            x = (
                pd.to_datetime(self.log["Timestamp"]) - self.experiment_starttime
            ).dt.total_seconds() / 60.0
            x_label = "Experiment Time (min)"
        else:
            x = self.log["Timestamp"]
            x_label = "Timestamp"

        for col in columns:
            if col not in self.log.columns:
                raise ValueError(f"Column {col} not found in log data.")
            ax.plot(x, self.log[col], label=col)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Value")
        ax.set_title("Experiment Log Data")
        ax.legend()

        return ax

    @property
    def log_data(self) -> pd.DataFrame:
        """Get log data, raising an error if not available"""
        if self.log is None:
            raise ValueError(
                "No log data available. Use add_log() to add log data first."
            )
        return self.log
