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
git clone https://git.science.uu.nl/icc-coders/ChromStream

# Install in development mode
pip install -e .
```

### Install using uv

If you're using [uv](https://github.com/astral-sh/uv) for fast Python package management:

```bash

# Or install from Git repository
uv add git+https://git.science.uu.nl/icc-coders/ChromStream

# For development installation
git clone https://git.science.uu.nl/icc-coders/ChromStream
cd chromstream
uv add -e .
```

## Quick Start

Here's a simple example of how to set up an experiment and add chromatograms to it:

```python
from chromstream.objects import Experiment

exp = Experiment(name='hello there')
exp.add_chromatogram('path-to-your-chromatogram') #make a loop for this
exp.plot_chromatograms()
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

