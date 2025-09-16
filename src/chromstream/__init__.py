"""Init data"""

from __future__ import annotations

from importlib.metadata import version

from .parsers import *

from .objects import *


# Import objects first
from .objects import ChannelChromatograms

# Then import functions that depend on objects
from .data_processing import *

__all__ = [
    "ChannelChromatograms",
    "integrate_channel",
    "min_subtract",
    "time_window_baseline",
    "time_point_baseline",
    "list_baseline_functions",
]


# Load the version
__version__ = version("chromstream")
