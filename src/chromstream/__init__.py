"""Init data"""

from __future__ import annotations

from importlib.metadata import version

from .parsers import *  

from .objects import *
# Load the version
__version__ = version("chromstream")
