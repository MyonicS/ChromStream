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
- `src/chromstream/parsers/hdf5.py`: strict parsers for ChromStream HDF5 experiment, channel, and chromatogram files.
- `src/chromstream/parsers/other_files.py`: MTO ASCII parser and several log-file parsers.
- `src/chromstream/hdf5_common.py`: shared HDF5 helpers (scalar converters, schema gate, reserved-attr sets, per-channel/per-chromatogram group read/write) reused by the writer and parser.
- `src/chromstream/writers/hdf5_writer.py`: HDF5 export for `Experiment`, `ChannelChromatograms`, and `Chromatogram`.
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
- `parse_experiment_hdf5(path)`, `parse_channel_hdf5(path)`, and `parse_chromatogram_hdf5(path)` read the experiment, channel, and chromatogram HDF5 formats respectively. Each accepts only its exact schema (`chromstream-experiment/v0.1.0`, `chromstream-channel/v0.1.0`, `chromstream-chromatogram/v0.1.0`); schema strings live as `SCHEMA_*` constants in `objects.py`.
- The HDF5 parsers verify that the file is a ChromStream HDF5 file before attempting schema-specific parsing (shared `_require_schema` gate in `hdf5_common.py`).
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

- `Experiment.to_hdf5()`, `ChannelChromatograms.to_hdf5()`, and `Chromatogram.to_hdf5()` delegate to `write_experiment_hdf5(...)`, `write_channel_hdf5(...)`, and `write_chromatogram_hdf5(...)`.
- All three types support strict HDF5 round-tripping via the matching `write_*` / `parse_*` pair. The three writers/parsers share one layout definition through the group helpers in `hdf5_common.py`, so a change to the on-disk format happens in one place.
- HDF5 layouts:
  - Experiment: file attrs for experiment metadata, then `Channels/<channel>/injections/inj-XXXX` with datasets `retention_time`, `signal`.
  - Channel: root *is* the channel group — `name` attr + `injections/inj-XXXX`.
  - Chromatogram: root *is* the chromatogram group — `retention_time`/`signal` datasets plus reserved attrs `schema`, `channel`, `injection_time`, and optional `injection_index`.
- Reserved (structural) attr keys, excluded from the free-form metadata dict, differ per type and live as `RESERVED_*_ATTRS` frozensets in `hdf5_common.py` (experiment: `schema`/`label`/`creation_date`/`author`).
- Metadata handling differs by type: the experiment/channel layout persists only units per chromatogram (reconstructed as `time_unit`/`Signal Unit`); the standalone chromatogram layout persists the full metadata dict key-for-key.
- A standalone chromatogram may store its `injection_index`; `Experiment.add_chromatogram_hdf5(path)` uses it to reassemble chromatogram files into an experiment losslessly, falling back to sequential numbering when the attr is absent.
- HDF5 is intentionally not wired into `parsers/dispatch.py` (an `.h5` file may be an experiment, channel, or chromatogram — ambiguous).
- Metadata values written to HDF5 must be scalar and HDF5-compatible (`Timestamp`/`Path` are normalized to ISO/`str`).

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
- 72 tests passing

## Testing Expectations

- Add or update tests for every parser, processing, or persistence change.
- Use fixtures in `tests/testdata/` whenever possible.
- Prefer extending the narrow test module that matches the change:
  - parser behavior: `tests/test_parsers.py`
  - object behavior and plotting: `tests/test_objects.py`
  - processing helpers: `tests/test_data_processing.py`
  - HDF5 experiment export/parse: `tests/test_hdf5_writer.py`, `tests/test_hdf5_parser.py`
  - HDF5 channel/chromatogram export and parse: `tests/test_hdf5_channel.py`, `tests/test_hdf5_chromatogram.py`
  - HDF5 chromatogram-to-experiment reassembly: `tests/test_hdf5_reassembly.py`

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
- The docs use sample parsing data from `tests/testdata/`, including `tests/testdata/chroms/test_dx.h5` for HDF5 parsing examples.
- If you change public API or supported formats, update both docs and tests in the same change.
