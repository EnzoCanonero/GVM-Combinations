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
```

Only the construction of the likelihood, numerical minimisation and the
analytical Bartlett correction are implemented.  Plotting routines and code
specific to the top-quark mass combination were removed to keep the toolkit
lightweight and general.

## Configuration File

The combination is driven by a plain text configuration file.  Blank lines and
any text after a ``#`` character are ignored.  Sections are introduced by an
ampersand (``&``) followed by the section name.  The file uses three sections:

* **&Combination setup** – contains general metadata.
  - ``Combination Name = <name>``
  - ``Number of Measurements = <n>``
  - ``Measurement names = m1 m2 ...``
* **&Systematics setup** – describes all systematic sources.
  - ``Number of systematics = <n>``
  - One line per systematic: ``name epsilon [path]`` where ``path`` points to an
    optional correlation matrix file.
* **&Data** – numeric table with one line per measurement.
  - Each row lists the central value, statistical uncertainty and then all
    systematic uncertainties in the order defined above.

See ``input_files/config_lhc.txt`` for a complete example.
