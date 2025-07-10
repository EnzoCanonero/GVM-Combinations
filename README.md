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
