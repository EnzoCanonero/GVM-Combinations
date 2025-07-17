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
info = comb.input_data()

# modify the input and update the combination
info['central']['ATLAS'] = 173.4
info['syst']['JES']['values']['ATLAS'] += 0.1
comb.update_data(info)
```

Only the construction of the likelihood, numerical minimisation and the
analytical Bartlett correction are implemented.  Plotting routines and code
specific to the top-quark mass combination were removed to keep the toolkit
lightweight and general.

## Setup

The toolkit requires a few Python packages.  A simple setup script is
provided to install them all::

    ./setup.sh

This installs the dependencies listed in ``requirements.txt`` using
``pip``.  You may also run ``pip install -r requirements.txt`` directly
if preferred.

## Configuration File

The combination is driven by a YAML configuration file with three main
sections:

* ``global`` – directories containing correlation matrices and statistical
  covariance files.  It must also define ``name``, ``n_meas`` and ``n_syst``
  giving the combination name and the expected numbers of measurements and
  systematic sources.
* ``data`` – the measurement names and central values together with their
  statistical uncertainties.  Statistical errors may be given explicitly or a
  ``stat_cov_path`` can provide a covariance matrix.  In either case the
  toolkit constructs the full covariance matrix.
* ``syst`` – list of systematics with their shift vectors, associated
  correlation matrix file and optional ``epsilon`` error-on-error value.

See ``input_files/LHC_mass_combination.yaml`` for a complete example using
statistical errors.  ``input_files/LHC_mass_combination_cov.yaml`` shows the
equivalent setup with a statistical covariance matrix.
