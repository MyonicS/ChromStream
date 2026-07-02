# Contributing to ChromStream

Thank you for your interest in contributing to ChromStream! This package is in active development and we welcome contributions from the community.

## Ways to Contribute

### Bug Reports
If you find a bug, please submit an issue on our [GitHub repository](https://github.com/MyonicS/ChromStream/issues) with:

- A clear description of the issue
- Steps to reproduce the problem
- Your environment details (Python version, OS, etc.)
- Sample data files if relevant

### Feature Requests
Have an idea for a new feature? Great! Please submit a feature request as an issue with:

- A detailed description of the proposed feature
- Use cases and examples

### File Format Support
If you have a specific file format that ChromStream doesn't currently support:

- Please provide an example file
- Include documentation about the format if available
- Describe the software that generates these files

### Code Contributions
We welcome pull requests! Here's how to get started:

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ChromStream
   cd ChromStream
   ```
3. **Set up the development environment**:
   ```bash
   uv sync --extra dev --extra docs
   ```
   This installs the package together with the development and documentation dependencies.
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Run the test suite** before opening a pull request:
   ```bash
   uv run pytest
   ```
   Useful additional checks:
   ```bash
   uv run ruff check .
   uv run mkdocs build
   ```
   You can also run a narrower test selection while working on a specific area:
   ```bash
   uv run pytest tests/test_hdf5_parser.py
   ```
6. **Make your changes** and add or update tests where needed
7. **Commit your changes** with a clear commit message
8. **Push to your fork** and submit a pull request

## Local Development Notes

- Use `uv run pytest` for the full test suite.
- The tests use the `src/` layout configured in `pyproject.toml`, so run them from the repository root.
- If you are working on the docs or notebooks, verify the documentation build with:
  ```bash
  uv run mkdocs build
  ```

## Development Roadmap

Current priorities include:

- Support for more file formats
- Addition of spectroscopy data sources
- JSON saving and parsing functionality
- Comprehensive tests

## Getting Help

If you need help with development, please do not hesitate to reach out. We appreciate all contributions, no matter how small!
