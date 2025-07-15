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
comb.update_inputs(info)
```

Only the construction of the likelihood, numerical minimisation and the
analytical Bartlett correction are implemented.  Plotting routines and code
specific to the top-quark mass combination were removed to keep the toolkit
lightweight and general.

## Configuration File

The combination is driven by a plain text configuration file.  Blank lines and
any text after a ``#`` character are ignored.  Sections are introduced by an
ampersand (``&``) followed by the section name.  The file now uses four
sections:

* **&Combination setup** – contains general metadata.
  - ``Combination Name = <name>``
  - ``Number of Measurements = <n>``
  - ``Measurement names = m1 m2 ...``
* **&Combination data** – provides the measurement values.
  - ``Measurement central values = v1 v2 ...``
  - Either ``Measurement stat errors = s1 s2 ...`` or ``Measurement stat covariance = <path>``.
    The latter must point to a text file with the full ``n × n`` statistical covariance matrix.
* **&Systematics setup** – describes all systematic sources.
  - ``Number of systematics = <n>``
  - One line per systematic: ``name epsilon [path]`` where ``path`` points to an
    optional correlation matrix file.  In this repository the correlation
    matrices are stored under ``input_files/correlations``.
* **&Systematics data** – numeric table with one line per measurement.
  - Each row lists the systematic uncertainties in the order defined above.

See ``input_files/config_lhc.txt`` for a complete example using statistical errors.
``input_files/config_lhc_cov.txt`` shows the equivalent setup with a statistical covariance matrix.
