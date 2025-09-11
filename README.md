# ChromStream

A Python package for processing on-line gas chromatography data. ChromStream provides tools to parse, analyze, and visualize chromatographic data from various GC systems.

## Features

- Parse chromatographic data from multiple formats (Chromeleon, FID, etc.)
- Fun with GC
- ...
## Installation


### Install from Git Repository

```bash
# Clone the repository
git clone https://github.com/Quantum-Accelerators/chromstream.git
cd chromstream

# Install in development mode
pip install -e .
```

### Install using uv

If you're using [uv](https://github.com/astral-sh/uv) for fast Python package management:

```bash

# Or install from Git repository
uv add git+https://github.com/Quantum-Accelerators/chromstream.git

# For development installation
git clone https://github.com/Quantum-Accelerators/chromstream.git
cd chromstream
uv add -e .
```

## Quick Start

Here's a simple example of how to set up an experiment and add chromatograms to it:

```python
from chromstream.objects import Experiment
from chromstream import parsers as csp
from pathlib import Path

# Create a new experiment
experiment = Experiment("my_gc_experiment")

# Path to your chromatogram data files
data_path = Path("path/to/your/chromatogram/files")

# Parse and add chromatograms from multiple files
for file_path in data_path.glob("*.txt"):
    # Parse chromatogram data (adjust parser based on your file format)
    metadata, data = csp.parse_chromeleon_txt(file_path)
    
    # Create chromatogram object
    chromatogram = csp.create_chromatogram_from_parsed_data(
        data, metadata, file_path
    )
    
    # Add to experiment
    experiment.add_chromatogram(chromatogram)

# Access chromatograms by channel
channel_data = experiment.get_channel("FID")

# Plot the data
channel_data.plot()
```


## Supported File Formats

ChromStream currently supports parsing data from:

- Chromeleon software exports (`.txt`)
- MTO setup (ascii files)

## Documentation

- to be developed and hosted

## Example Notebooks

Check out the `example_notebooks/` directory for comprehensive examples:

- `example_calibration.ipynb` - GC calibration procedures
- `splitting_examples.ipynb` - Data splitting and processing examples
- `dev_notebook_MTO.ipynb` - MTO setup specific examples

