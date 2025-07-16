# GVM Combination Toolkit

This repository contains a simple implementation of the Gamma Variance Model (GVM) for
combining correlated measurements.  The provided `GVMCombination` class builds
the likelihood, minimises it with *Minuit* and computes confidence intervals
using an analytic Bartlett correction.

## Usage

```python
from gvm_toolkit import GVMCombination

# create the combination directly from a configuration file
comb = GVMCombination("input_files/config_lhc.txt")

print("mu hat:", comb.fit_results['mu'])

print("68% CI:", comb.confidence_interval())
print("Goodness of fit:", comb.goodness_of_fit())

# access a dictionary with all input values
info = comb.input_summary()

# modify the input and update the combination
info['central']['ATLAS'] = 173.4
info['systematics']['JES']['ATLAS'] += 0.1
comb.update_inputs(info)
```

Only the construction of the likelihood, numerical minimisation and the
analytical Bartlett correction are implemented.  Plotting routines and code
specific to the top-quark mass combination were removed to keep the toolkit
lightweight and general.

## Configuration File

The combination is driven by a plain text configuration file in INI format.
Blank lines and any text after a ``#`` character are ignored.  The file uses
five sections:

* **[Globals]** – contains general metadata.
  - ``corr_dir``: directory holding correlation matrices.
  - ``combination_name``: a short label for the combination.
  - ``number_of_measurements`` and ``number_of_systematics`` give the expected
    matrix dimensions.
* **[Data]** – lists each measurement in the form ``name  value [stat_err]``.
  When a full statistical covariance matrix is provided the third column is
  omitted and ``stat_cov_path = <file>`` is added.
* **[Systematics Setup]** – one line per systematic with the uncertainty for
  each measurement.
* **[Syst Correlations]** – maps systematic names to the correlation files in
  ``corr_dir``.
* **[Errors-on-Errors]** – optional error-on-error values for each systematic.

See ``input_files/config_lhc.txt`` for a complete example using statistical errors.
``input_files/config_lhc_cov.txt`` shows the equivalent setup with a statistical covariance matrix.
