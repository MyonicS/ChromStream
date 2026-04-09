# AGENTS.md

## Purpose

This repository contains `chromstream`, a Python library for parsing, processing, plotting, and persisting on-line gas chromatography data.

Future agents should treat this as a scientific data library first, not a generic app:

- Preserve parsing correctness over API cleverness.
- Prefer small, test-backed changes.
- Keep public objects and column conventions stable unless a breaking change is intentional.

## Repository Map

- `src/chromstream/__init__.py`: top-level package exports via star imports.
- `src/chromstream/objects.py`: core dataclasses: `Chromatogram`, `ChannelChromatograms`, `Experiment`.
- `src/chromstream/data_processing.py`: baseline correction registry, integration helpers, log merging, chromatogram splitting.
- `src/chromstream/parsers/chromeleon.py`: Chromeleon `.txt` parsing and injection-time parsing.
- `src/chromstream/parsers/agilent.py`: Agilent `.ch`, `.d`, and `.dx` parsing.
- `src/chromstream/parsers/dispatch.py`: single-file parser dispatch for supported chromatogram files.
- `src/chromstream/parsers/other_files.py`: MTO ASCII parser and several log-file parsers.
- `src/chromstream/writers/hdf5_writer.py`: HDF5 export for `Experiment`.
- `tests/`: pytest suite with representative sample files in `tests/testdata/`.
- `docs/` and `mkdocs.yml`: documentation and notebooks, built with MkDocs Material.

## Current Architecture

### Core data model

- `Chromatogram` represents one injection on one channel.
- `ChannelChromatograms` groups many `Chromatogram` objects for one detector/channel.
- `Experiment` groups channels and optional log data.

### Important invariants

- Chromatogram data is stored as a `pandas.DataFrame`.
- The first column is assumed to be retention time.
- The second column is assumed to be signal unless a method explicitly accepts `column=...`.
- Time units are taken from `metadata["time_unit"]`.
- Signal units are taken from `metadata["Signal Unit"]` first, then `metadata["signal_unit"]`.
- Integrated result tables and log tables use a `Timestamp` column.

Many methods rely on column order rather than column names. Do not reorder columns casually.

### Parsing surface

- `parse_chromatogram(path)` in `parsers/dispatch.py` currently supports:
  - Agilent `.ch`
  - Chromeleon `.txt` only when the file content matches expected chromatogram metadata
- `Experiment.add_chromatogram(...)` does not use the dispatch helper for all formats:
  - `.ch` goes to `parse_agilent_ch`
  - all other file paths currently go to `parse_chromatogram_txt`
- `Experiment.add_mult_chromatograms(...)` supports:
  - Agilent `.d` directories
  - Agilent `.dx` archives
  - lists of paths or `Chromatogram` objects
- `other_files.py` contains additional parsers for:
  - MTO ASCII chromatogram exports
  - several log file formats

If you add a new chromatogram format, update the parser, the dispatch layer, and the `Experiment` helpers together.

### Processing surface

- Baseline functions are registered with `@register_baseline`.
- `list_baseline_functions()` relies on registration order.
- Peak integration uses `scipy.integrate.trapezoid`.
- `split_chromatogram()` assumes the sliced chromatogram length is divisible by `n_injections`.
- `add_log_data()` merges on nearest `Timestamp` using `pandas.merge_asof`.

### Persistence

- `Experiment.to_hdf5()` delegates to `write_experiment_hdf5(...)`.
- HDF5 layout is:
  - file attrs for experiment metadata
  - `Channels/<channel>/injections/inj-XXXX`
  - datasets: `retention_time`, `signal`
- Reserved experiment metadata keys in HDF5 export:
  - `schema`
  - `label`
  - `creation_date`
  - `author`
- Metadata values written to HDF5 must be scalar and HDF5-compatible.

## Development Workflow

Primary commands:

- Install dev and docs dependencies: `uv sync --extra dev --extra docs`
- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Build docs locally: `uv run mkdocs build`
- Serve docs locally: `uv run mkdocs serve`

Fallback without `uv`:

- `pip install -e .[dev,docs]`

Baseline status at the time this file was written:

- `uv run pytest` passes
- 44 tests passing

## Testing Expectations

- Add or update tests for every parser, processing, or persistence change.
- Use fixtures in `tests/testdata/` whenever possible.
- Prefer extending the narrow test module that matches the change:
  - parser behavior: `tests/test_parsers.py`
  - object behavior and plotting: `tests/test_objects.py`
  - processing helpers: `tests/test_data_processing.py`
  - HDF5 export: `tests/test_hdf5_writer.py`

For new file-format support, include at least:

- one representative sample file in `tests/testdata/`
- a happy-path parse test
- one failure or unsupported-shape test if practical

## Code Conventions

- The package uses a `src/` layout.
- Python support target is `>=3.9`.
- Ruff, Black-style formatting, mypy, and pyright settings live in `pyproject.toml`.
- Tests run with `pythonpath = "src"` and `--import-mode=importlib`.
- `from __future__ import annotations` is standard across modules.
- Public API is re-exported broadly from package `__init__.py` and parser/writer `__init__.py` files. Be careful when renaming or removing symbols.

### Docstrings

- Keep docstrings concise and practical.
- Prefer Google-style section headers such as `Args:`, `Returns:`, and `Raises:` for new or edited docstrings.
- This matches the MkDocs configuration in `mkdocs.yml`, which renders Python docstrings with `docstring_style: google`.
- Existing modules are somewhat mixed; when touching nearby code, normalize toward the Google-style format instead of introducing a third style.
- Document units, dataframe expectations, and timestamp behavior when they are important to correct scientific use.

## Practical Guardrails

- Preserve dataframe column ordering assumptions unless you are deliberately refactoring all dependent code.
- Preserve `Timestamp` naming across integrations and log merges.
- Be careful with timezone handling. Parsers often normalize to naive `pandas.Timestamp`.
- Avoid changing parser error behavior without tests; users may depend on current `ValueError` and `FileNotFoundError` cases.
- Keep scientific units explicit in metadata when introducing new parsers or transformations.
- Do not touch untracked local artifacts unless the task explicitly requires it. 

## Known Extension Points

Common future changes and the files they usually require:

- New chromatogram file format:
  - parser module under `src/chromstream/parsers/`
  - `src/chromstream/parsers/__init__.py`
  - `src/chromstream/parsers/dispatch.py`
  - `src/chromstream/objects.py` if `Experiment` helpers should auto-detect it
  - tests and docs
- New baseline correction:
  - `src/chromstream/data_processing.py`
  - tests for registration and output behavior
- New persistence format:
  - `src/chromstream/writers/`
  - `src/chromstream/writers/__init__.py`
  - `Experiment` convenience method if needed
  - round-trip or layout tests

## Documentation Notes

- Documentation is configured in `mkdocs.yml`.
- API reference pages are generated via `docs/gen_ref_pages.py`.
- User-facing examples live mostly in notebooks under `docs/notebooks/`.
- If you change public API or supported formats, update both docs and tests in the same change.
