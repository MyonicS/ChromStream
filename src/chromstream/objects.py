from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


@dataclass
class Chromatogram:
    """Single chromatogram data for one injection on one channel"""

    data: pd.DataFrame
    injection_time: pd.Timestamp
    metadata: dict
    channel: str
    path: Path | str

    def plot(self, ax=None, **kwargs):
        """Plot the chromatogram data"""
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.data["Time (min)"], self.data[self.data.columns[1]], **kwargs)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(self.data.columns[1])
        ax.set_title(f"Chromatogram - {self.channel} - {self.path}")

        return ax


@dataclass
class ChannelChromatograms:
    """All data for a single detector channel"""

    channel: str  # 'FID', 'TCD', etc.
    chromatograms: dict[int, Chromatogram] = field(default_factory=dict)

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
                chrom.data["Time (min)"],
                chrom.data[chrom.data.columns[1]],
                label=f"Injection {inj_num}",
                color=colors[inj_num],
                **kwargs,
            )

        ax.set_xlabel("Time (min)")
        ax.set_ylabel(self.chromatograms[0].data.columns[1])
        ax.set_title(f"Channel: {self.channel}")
        # add colorbar
        if plot_colorbar:
            sm = plt.cm.ScalarMappable(
                norm=Normalize(vmin=0, vmax=len(self.chromatograms) - 1)
            )  # type: ignore
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Injection Number")

        return ax


@dataclass
class Experiment:
    """Data for a single experiment containing multiple on-line GC channels"""

    name: str
    channels: dict[str, ChannelChromatograms] = field(default_factory=dict)
    experiment_startime: Optional[pd.Timestamp] = None
    experiment_endtime: Optional[pd.Timestamp] = None

    def add_channel(self, channel_name: str, channel_data: ChannelChromatograms):
        """Add a channel to the experiment"""
        self.channels[channel_name] = channel_data

    def add_chromatogram(
        self,
        chromatogram: Path | str | Chromatogram,
        channel_name: Optional[str] = None,
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

    def plot(self, ax=None, channels: str | list = "all", **kwargs):
        if ax is None:
            n_channels_to_plot = (
                len(self.channels) if channels == "all" else len(channels)
            )

            fig, ax = plt.subplots(
                n_channels_to_plot,
                1,
                figsize=(7, 3.3 / 1.618 * n_channels_to_plot),
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
