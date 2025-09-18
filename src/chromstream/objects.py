from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from chromstream.data_processing import integrate_channel
import logging as log
import json


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

    def to_dict(self) -> dict:
        """
        Convert Chromatogram to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the Chromatogram
        """
        # Convert DataFrame to serializable format
        data_dict = {
            "columns": list(self.data.columns),
            "values": self.data.to_numpy().tolist(),
            "index": [str(idx) for idx in self.data.index],  # Convert index to strings
        }

        # Ensure metadata is JSON serializable
        serializable_metadata = {}
        for key, value in self.metadata.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                serializable_metadata[key] = str(value)
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_metadata[key] = value
            else:
                serializable_metadata[key] = str(value)

        return {
            "data": data_dict,
            "injection_time": self.injection_time.isoformat()
            if self.injection_time
            else None,
            "metadata": serializable_metadata,
            "channel": self.channel,
            "path": str(self.path),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Chromatogram:
        """
        Create a Chromatogram from a dictionary (JSON deserialization).

        Args:
            data: Dictionary containing chromatogram data

        Returns:
            Chromatogram object
        """
        # Reconstruct DataFrame with proper index handling
        index_data = data["data"]["index"]

        # Try to convert index back to appropriate type
        try:
            # First try integer conversion
            index_data = [int(float(idx)) for idx in index_data]
        except (ValueError, TypeError):
            try:
                # Then try float conversion
                index_data = [float(idx) for idx in index_data]
            except (ValueError, TypeError):
                # Keep as strings if conversion fails
                pass

        df_data = pd.DataFrame(
            data["data"]["values"], columns=data["data"]["columns"], index=index_data
        )

        # Parse injection time
        injection_time = (
            pd.Timestamp(data["injection_time"])
            if data["injection_time"]
            else pd.Timestamp.now()
        )

        return cls(
            data=df_data,
            injection_time=injection_time,
            metadata=data["metadata"],
            channel=data["channel"],
            path=Path(data["path"]),
        )


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

        # Set labels and title (handle empty channel case)
        if len(self.chromatograms) > 0:
            # Use any chromatogram to get column names
            sample_chrom = next(iter(self.chromatograms.values()))
            ax.set_xlabel(sample_chrom.data.columns[0])
            ax.set_ylabel(sample_chrom.data.columns[1])
        else:
            ax.set_xlabel("Time")
            ax.set_ylabel("Signal")
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
        self, peaklist: dict, column: None | str = None
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
        self.integrals = integrate_channel(self, peaklist, column=column)
        return self.integrals

    def to_dict(self) -> dict:
        """
        Convert ChannelChromatograms to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the ChannelChromatograms
        """
        # Convert chromatograms dict to serializable format
        chromatograms_dict = {
            str(key): chrom.to_dict() for key, chrom in self.chromatograms.items()
        }

        # Convert integrals DataFrame if it exists
        integrals_dict = None
        if self.integrals is not None:
            # Store column dtypes for proper reconstruction
            dtypes = {col: str(dtype) for col, dtype in self.integrals.dtypes.items()}

            # Handle potential timestamp columns in integrals
            integrals_df = self.integrals.copy()
            for col in integrals_df.columns:
                if pd.api.types.is_datetime64_any_dtype(integrals_df[col]):
                    integrals_df[col] = integrals_df[col].dt.strftime(
                        "%Y-%m-%dT%H:%M:%S.%f"
                    )
                elif pd.api.types.is_timedelta64_dtype(integrals_df[col]):
                    integrals_df[col] = integrals_df[col].astype(str)

            integrals_values = integrals_df.to_numpy().tolist()

            integrals_dict = {
                "columns": list(self.integrals.columns),
                "values": integrals_values,
                "index": [
                    str(idx) for idx in self.integrals.index
                ],  # Convert index to strings
                "dtypes": dtypes,
            }

        return {
            "channel": self.channel,
            "chromatograms": chromatograms_dict,
            "integrals": integrals_dict,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChannelChromatograms:
        """
        Create a ChannelChromatograms from a dictionary (JSON deserialization).

        Args:
            data: Dictionary containing channel data

        Returns:
            ChannelChromatograms object
        """
        # Reconstruct chromatograms
        chromatograms = {}
        for key, chrom_data in data["chromatograms"].items():
            chromatograms[int(key)] = Chromatogram.from_dict(chrom_data)

        # Reconstruct integrals DataFrame if it exists
        integrals = None
        if data["integrals"] is not None:
            index_data = data["integrals"]["index"]
            # Try to convert index back to appropriate type
            try:
                # First try integer conversion
                index_data = [int(float(idx)) for idx in index_data]
            except (ValueError, TypeError):
                try:
                    # Then try float conversion
                    index_data = [float(idx) for idx in index_data]
                except (ValueError, TypeError):
                    # Keep as strings if conversion fails
                    pass

            # Create DataFrame first
            integrals = pd.DataFrame(
                data["integrals"]["values"],
                columns=data["integrals"]["columns"],
                index=index_data,
            )

            # Convert columns back to their original dtypes if available
            if "dtypes" in data["integrals"]:
                for col, dtype_str in data["integrals"]["dtypes"].items():
                    if col in integrals.columns:
                        if "datetime64" in dtype_str:
                            integrals[col] = pd.to_datetime(integrals[col])
                        elif "timedelta64" in dtype_str:
                            integrals[col] = pd.to_timedelta(integrals[col])

        return cls(
            channel=data["channel"], chromatograms=chromatograms, integrals=integrals
        )


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

            # Handle empty experiment case
            if n_channels_to_plot == 0:
                fig, ax = plt.subplots()
                ax.text(
                    0.5,
                    0.5,
                    "No channels to plot",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Empty Experiment")
                return

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

    def to_dict(self) -> dict:
        """
        Convert Experiment to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the Experiment
        """
        # Convert channels
        channels_dict = {
            name: channel.to_dict() for name, channel in self.channels.items()
        }

        # Convert log DataFrame if it exists
        log_dict = None
        if self.log is not None:
            # Store column dtypes for proper reconstruction
            dtypes = {col: str(dtype) for col, dtype in self.log.dtypes.items()}

            # Convert DataFrame values, handling timestamp columns
            log_df = self.log.copy()
            for col in log_df.columns:
                if pd.api.types.is_datetime64_any_dtype(log_df[col]):
                    log_df[col] = log_df[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                elif pd.api.types.is_timedelta64_dtype(log_df[col]):
                    log_df[col] = log_df[col].astype(str)

            log_values = log_df.to_numpy().tolist()

            log_dict = {
                "columns": list(self.log.columns),
                "values": log_values,
                "index": [
                    str(idx) for idx in self.log.index
                ],  # Convert index to strings
                "dtypes": dtypes,  # Store column data types
            }

        return {
            "name": self.name,
            "channels": channels_dict,
            "experiment_starttime": self.experiment_starttime.isoformat()
            if self.experiment_starttime
            else None,
            "experiment_endtime": self.experiment_endtime.isoformat()
            if self.experiment_endtime
            else None,
            "log": log_dict,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Experiment:
        """
        Create an Experiment from a dictionary (JSON deserialization).

        Args:
            data: Dictionary containing experiment data

        Returns:
            Experiment object
        """
        # Reconstruct channels
        channels = {}
        for name, channel_data in data["channels"].items():
            channels[name] = ChannelChromatograms.from_dict(channel_data)

        # Parse timestamps
        experiment_starttime = (
            pd.Timestamp(data["experiment_starttime"])
            if data["experiment_starttime"]
            else None
        )
        experiment_endtime = (
            pd.Timestamp(data["experiment_endtime"])
            if data["experiment_endtime"]
            else None
        )

        # Reconstruct log DataFrame if it exists
        log = None
        if data["log"] is not None:
            index_data = data["log"]["index"]
            # Try to convert index back to appropriate type
            try:
                # First try integer conversion
                index_data = [int(float(idx)) for idx in index_data]
            except (ValueError, TypeError):
                try:
                    # Then try float conversion
                    index_data = [float(idx) for idx in index_data]
                except (ValueError, TypeError):
                    # Keep as strings if conversion fails
                    pass

            # Create DataFrame first
            log = pd.DataFrame(
                data["log"]["values"], columns=data["log"]["columns"], index=index_data
            )

            # Convert columns back to their original dtypes if available
            if "dtypes" in data["log"]:
                for col, dtype_str in data["log"]["dtypes"].items():
                    if col in log.columns:
                        if "datetime64" in dtype_str:
                            log[col] = pd.to_datetime(log[col])
                        elif "timedelta64" in dtype_str:
                            log[col] = pd.to_timedelta(log[col])
                        # For other types, pandas will handle them automatically

        return cls(
            name=data["name"],
            channels=channels,
            experiment_starttime=experiment_starttime,
            experiment_endtime=experiment_endtime,
            log=log,
        )

    def to_json(self, file_path: str | Path) -> None:
        """
        Save the Experiment to a JSON file.

        Args:
            file_path: Path where to save the JSON file
        """
        file_path = Path(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, file_path: str | Path) -> Experiment:
        """
        Load an Experiment from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Experiment object
        """
        file_path = Path(file_path)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
