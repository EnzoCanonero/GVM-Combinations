# GVM Combination Toolkit

This repository contains a simple implementation of the Gamma Variance Model (GVM) for
combining correlated measurements.  The provided `GVMCombination` class builds
the likelihood, minimises it with *Minuit* and computes confidence intervals
using an analytic Bartlett correction.

## Usage

```python
from gvm_toolkit import GVMCombination

# create the combination directly from a configuration file
comb = GVMCombination("input_files/LHC_mass_combination.yaml")

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

The combination is driven by a YAML configuration file with three main
sections:

* ``globals`` – directories containing correlation matrices and statistical
  covariance files.
* ``combination`` – the measurement names and central values.  Each entry may
  include a ``stat_error`` field or the section may define ``stat_cov_path``
  pointing to a covariance matrix.
* ``systematics`` – list of systematics with their shift vectors, associated
  correlation matrix file and optional ``epsilon`` error-on-error value.

See ``input_files/LHC_mass_combination.yaml`` for a complete example using
statistical errors.  ``input_files/LHC_mass_combination_cov.yaml`` shows the
equivalent setup with a statistical covariance matrix.
