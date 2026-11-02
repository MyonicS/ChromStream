# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Changed

- Chromatogram parsing now uses [chromparser](https://github.com/MyonicS/chromparser). Existing parser functions, signatures, DataFrame columns and metadata keys are kept; chromparser's normalised snake_case metadata keys are added alongside.
- Requires Python >= 3.10 (chromparser's minimum).
- `parse_chromatogram` detects the format from the file content instead of the extension.
- Chromeleon (bug fix): ambiguous dates with an AM/PM time (e.g. `9/7/2023 3:26:11 PM`) are read month first (US locale) regardless of `dayfirst`; previously they were read day first, giving a wrong date. The byte order mark is no longer part of the first metadata key (`"URL"` instead of `"\ufeffURL"`).
- Agilent `.dx`: the injection time comes from the run time in the archive (sub-second precision), and `Chromatogram.path` is the `.dx` archive instead of the `.ch` member name. A `.dx` archive without chromatograms raises a `ValueError` instead of returning an empty list.

### Added

- `parse_chromatograms(path)`: parse all chromatograms of a chromatogram file. Supported formats are maintained in chromparser.
- `Experiment.add_mult_chromatograms` and `Experiment.add_chromatogram` accept all chromatogram files.
- `dayfirst` option for `parse_chromatogram_txt` and `parse_chromeleon_txt`.

## [0.2.0]

### Added

- HDF5 export and parsing for `Experiment`, `ChannelChromatograms` and `Chromatogram` (`to_hdf5()`, `parse_experiment_hdf5`, `parse_channel_hdf5`, `parse_chromatogram_hdf5`), with gzip compression by default.
- `Experiment.add_chromatogram_hdf5()` to reassemble standalone chromatogram files into an experiment.

### Fixed

- Chromeleon inject time: no more `dayfirst` UserWarning; ambiguous dates are explicitly read day first by default.

## [0.1.0]

- Added parsing of Agilent .cd / .dx and .d directories, see new [parsing notebook](https://myonics.github.io/ChromStream/notebooks/parsing.html)
- extra tests
- minor cleanup

## [0.0.2]

- Dependency fixes

## [0.0.1]

### Added

- The initial release!
