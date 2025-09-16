# ChromStream

A Python package for processing on-line gas chromatography data. ChromStream provides tools to parse, analyze, and visualize chromatographic data from various GC systems, and combine it with data from logfiles such as temperature and pressure.

## Features

- Parse chromatographic data from multiple formats (Chromeleon, FID, etc.)
- Access to data at experiment, channel and chromatogram level
- Quick plotting of chromatograms
- Small selection of baseline corrections, possibility to use custom ones
- Integration using a dict of peaks
- Addition of logfiles

## Installation
Chromstream is presently not availible on PyPi. You can however install it straight from the repository:

### Install from Git Repository

```bash
# Clone the repository
git clone https://github.com/MyonicS/ChromStream
cd ChromStream
# Install in development mode
pip install -e .
```

### Install using uv

If you're using [uv](https://github.com/astral-sh/uv) for fast Python package management:

```bash
uv add git+https://github.com/MyonicS/ChromStream
```
## Quick Start

Check the Quickstart Notebook to see a full demonstration of the most important features of the package. 
Here's a simple example of how to set up an experiment and add chromatograms to it:

```python
from chromstream.objects import Experiment

exp = Experiment(name='hello there')
exp.add_chromatogram('path-to-your-chromatogram') #make a loop for this
exp.plot_chromatograms()
```

To access specific channels:
```python
exp.channels['channel-name'].plot()
```

For specific chromatograms:

```python
exp.channels['channel-name'].chromatograms[0].plot()
```

## Supported File Formats

ChromStream currently supports parsing data from:

- Chromeleon software exports (`.txt`)
- (software names) (ascii files)
- Certain labview log files 

## Documentation

- to be developed and hosted

## Example Notebooks

Check out the `example_notebooks/` directory for comprehensive examples:

- `example_calibration.ipynb` - GC calibration procedures


## Roadmap
- Support for more files formats
- Addition of more data sources such as spectroscopy
- JSON saving and parsings
- tests

## Contributing
This package is in active development. Any help is appreciated. You can submit feature requests or bug reports as issues on the repository.
If you have a specific file format which presently is nto supported please provide an example file.
PRs are more than welcome.

## Authors



