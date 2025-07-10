# GVM Combination Toolkit

This repository contains a simple implementation of the Gamma Variance Model (GVM) for
combining correlated measurements.  The provided `GVMCombination` class builds
the likelihood, minimises it with *Minuit* and computes confidence intervals
using an analytic Bartlett correction.

## Usage

```python
from gvm_toolkit import GVMCombination

# measurements and statistical uncertainties (vector or covariance matrix)
y = [...]
stat = [...]
# stat_cov = np.array([...])  # alternatively provide full covariance

# optional dictionary of systematic uncertainties for each measurement
syst = {
    'sys1': [...],
    'sys2': [...],
}

# correlation matrices; if omitted a diagonal matrix is assumed
correlations = {
    'sys1': rho_sys1,
    # 'sys2' : defaults to diagonal correlation
}

# specify which systematics have an errors-on-errors parameter epsilon
uncertain = {
    'sys1': 0.3,
}

comb = GVMCombination(y, stat, syst, correlations, uncertain)
# or simply GVMCombination(y, stat) if no systematics are present

print("mu hat:", comb.fit_results['mu'])

# minimise only the nuisance parameters while fixing mu
comb.minimize(fixed={'mu': 172.5})
print("68% CI:", comb.confidence_interval())
print("Goodness of fit:", comb.goodness_of_fit())
```

Only the construction of the likelihood, numerical minimisation and the
analytical Bartlett correction are implemented.  Plotting routines and code
specific to the top-quark mass combination were removed to keep the toolkit
lightweight and general.
