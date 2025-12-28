# Installation

ChromStream is a Python package for processing on-line gas chromatography data. Follow the instructions below to install the package.

### Using pip

```bash
pip install ChromStream
```

### Using uv

If you're using [uv](https://github.com/astral-sh/uv) for fast Python package management:

```bash
uv add ChromStream
```

## Development Installation

For development work or to get the latest features, you can install ChromStream in editable mode:

### Clone and Install in Development Mode

```bash
# Clone the repository
git clone https://github.com/MyonicS/ChromStream
cd ChromStream

# Install in development mode
pip install -e .
```

### Reproduce Exact Development Environment

To reproduce the exact development environment with all dependencies:

```bash
# Clone the repository
git clone https://github.com/MyonicS/ChromStream
cd ChromStream

# Sync the exact development environment with dev and docs dependencies
uv sync --extra dev --extra docs
```

This will install ChromStream along with all development dependencies (testing, linting, documentation, etc.) in the exact versions specified in the lock file.

## Verify Installation

After installation, you can verify that ChromStream is installed correctly by importing it in Python:

```python
import chromstream
print("ChromStream installed successfully!")
```

## Next Steps

Once you have ChromStream installed, check out the [Quickstart Guide](../notebooks/Quickstart.ipynb) to learn how to use the package.
