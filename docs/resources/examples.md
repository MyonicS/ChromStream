# Example Notebooks

This section contains example notebooks for the ChromStream package. 
Running of these notebooks requires development data which will be uploaded to a seperate repository.

## Available Examples

### [Quickstart Guide](../notebooks/Quickstart.ipynb)
Provides an introduction to ChromStream's core functionality. This notebook covers:
- Setting up experiments
- Loading chromatographic data
- Basic plotting and visualization
- Working with channels and chromatograms

### [Calibration Example](../notebooks/example_calibration.ipynb)
Shows how to calibrate two GC systems:
- A conventional on-line GC
- A specilized system with heated sample loops

### [Cracking Example](../notebooks/cracking_example.ipynb)
Provides an example analysis for cracking of a light hydrocarbon
- Importing, baseline correction and integration
- Applying of an internal standard
- Estimation of partial pressures using a calibration
- Adding data from log files (e.g. Temperature) to the analysis

### [HDF5 Export](../notebooks/exporting_hdf5.ipynb)
Shows the basic HDF5 persistence workflow:
- Exporting an `Experiment` to `.h5`
- Reloading the experiment with the HDF5 parser
- Why the format is useful for compact storage and fast re-loading
