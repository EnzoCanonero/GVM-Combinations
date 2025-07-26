# Toy Correlation Tutorial

In this notebook we explore how different correlation assumptions impact a simple two-measurement combination. Both measurements provide an estimate of the same quantity with central values +1 and -1. Their statistical uncertainties are $\sqrt{2}$ and we introduce a single systematic source also of magnitude $\sqrt{2}$. All error-on-error terms ($\epsilon$) are set to zero so only the correlations influence the result.

```python
import os, sys

script_dir = os.getcwd()
gvm_root = os.path.abspath(os.path.join(script_dir, "../../"))
if gvm_root not in sys.path:
    sys.path.insert(0, gvm_root)

from gvm_toolkit import GVMCombination
```

The following cells build a `GVMCombination` for each correlation scenario. After running the fit we print the estimated mean (`mu_hat`), the 68% confidence interval and the goodness-of-fit (chi-square) value.

## 1. Decorrelated case
Here the systematic uncertainty is defined as independent for each measurement (no correlation matrix).

`decorrelated.yaml`
```yaml
global:
  name: decorrelated
  n_meas: 2
  n_syst: 1
  corr_dir: tutorials/toy/correlations

data:
  measurements:
    - label: m1
      central: 1.0
      stat_error: 1.4142
    - label: m2
      central: -1.0
      stat_error: 1.4142

syst:
  - name: sys1
    shifts: [1.4142, 1.4142]
    type: independent
    epsilon: 0.0
```

```python
comb = GVMCombination('correlations/decorrelated.yaml')
mu_hat = comb.fit_results['mu']
ci_low, ci_high, _ = comb.confidence_interval()
chi2 = comb.goodness_of_fit()
print(f'decorrelated: mu_hat={mu_hat:.4f}, CI=({ci_low:.4f}, {ci_high:.4f}), chi2={chi2:.3f}')
```

```
decorrelated: mu_hat=0.0000, CI=(-1.4188, 1.4138), chi2=0.500
```

## 2. Correlation examples
We now compare three different correlation matrices for the systematic uncertainty:
1. **Diagonal**: off-diagonal terms are zero so the systematic acts independently.
2. **Fully correlated**: all coefficients are one so the systematic behaves as one shared nuisance parameter.
3. **Hybrid**: the off-diagonal coefficient is 0.5 representing a partial correlation.

### Diagonal
`diag_corr.yaml`
```yaml
global:
  name: diag_corr
  n_meas: 2
  n_syst: 1
  corr_dir: tutorials/toy/correlations

data:
  measurements:
    - label: m1
      central: 1.0
      stat_error: 1.4142
    - label: m2
      central: -1.0
      stat_error: 1.4142

syst:
  - name: sys1
    shifts: [1.4142, 1.4142]
    type:
      dependent: diagonal
    epsilon: 0.0
```
The diagonal option creates the identity matrix:
```
1 0
0 1
```

### Fully correlated
`full_corr.yaml`
```yaml
global:
  name: full_corr
  n_meas: 2
  n_syst: 1
  corr_dir: tutorials/toy/correlations

data:
  measurements:
    - label: m1
      central: 1.0
      stat_error: 1.4142
    - label: m2
      central: -1.0
      stat_error: 1.4142

syst:
  - name: sys1
    shifts: [1.4142, 1.4142]
    type:
      dependent: ones
    epsilon: 0.0
```
The `ones` option yields a matrix filled with ones:
```
1 1
1 1
```

### Hybrid
`hybrid_corr.yaml`
```yaml
global:
  name: hybrid_corr
  n_meas: 2
  n_syst: 1
  corr_dir: tutorials/toy/correlations

data:
  measurements:
    - label: m1
      central: 1.0
      stat_error: 1.4142
    - label: m2
      central: -1.0
      stat_error: 1.4142

syst:
  - name: sys1
    shifts: [1.4142, 1.4142]
    type:
      dependent: hybrid_corr.txt
    epsilon: 0.0
```
`hybrid_corr.txt`
```
1 0.5
0.5 1
```

```python
comb = GVMCombination('correlations/diag_corr.yaml')
mu_hat = comb.fit_results['mu']
ci_low, ci_high, _ = comb.confidence_interval()
chi2 = comb.goodness_of_fit()
print(f'diag_corr: mu_hat={mu_hat:.4f}, CI=({ci_low:.4f}, {ci_high:.4f}), chi2={chi2:.3f}')

comb = GVMCombination('correlations/full_corr.yaml')
mu_hat = comb.fit_results['mu']
ci_low, ci_high, _ = comb.confidence_interval()
chi2 = comb.goodness_of_fit()
print(f'full_corr: mu_hat={mu_hat:.4f}, CI=({ci_low:.4f}, {ci_high:.4f}), chi2={chi2:.3f}')

comb = GVMCombination('correlations/hybrid_corr.yaml')
mu_hat = comb.fit_results['mu']
ci_low, ci_high, _ = comb.confidence_interval()
chi2 = comb.goodness_of_fit()
print(f'hybrid_corr: mu_hat={mu_hat:.4f}, CI=({ci_low:.4f}, {ci_high:.4f}), chi2={chi2:.3f}')
```

```
diag_corr: mu_hat=0.0000, CI=(-1.4188, 1.4138), chi2=0.500
full_corr: mu_hat=0.0000, CI=(-1.7375, 1.7325), chi2=1.000
hybrid_corr: mu_hat=0.0000, CI=(-1.5888, 1.5813), chi2=0.667
```

## 3. Non-diagonal statistical covariance
The same systematic is fully correlated as above, but the statistical uncertainties are supplied via a covariance matrix with non-zero off-diagonal terms.

`stat_cov.yaml`
```yaml
global:
  name: stat_cov
  n_meas: 2
  n_syst: 1
  corr_dir: tutorials/toy/correlations

data:
  stat_cov_path: stat_cov.txt
  measurements:
    - label: m1
      central: 1.0
    - label: m2
      central: -1.0

syst:
  - name: sys1
    shifts: [1.4142, 1.4142]
    type:
      dependent: ones
    epsilon: 0.0
```
`stat_cov.txt`
```
2 1
1 2
```
The fully correlated systematic corresponds to a matrix of ones:
```
1 1
1 1
```

```python
comb = GVMCombination('correlations/stat_cov.yaml')
mu_hat = comb.fit_results['mu']
ci_low, ci_high, _ = comb.confidence_interval()
chi2 = comb.goodness_of_fit()
print(f'stat_cov: mu_hat={mu_hat:.4f}, CI=({ci_low:.4f}, {ci_high:.4f}), chi2={chi2:.3f}')
```

```
stat_cov: mu_hat=0.0000, CI=(-1.8788, 1.8713), chi2=2.000
```

## Tip: modifying the combination
It can be useful to adjust the input of an existing combination and rerun the fit
without editing the YAML file.  The `input_data()` method returns a dictionary
summarising the current configuration, which can then be modified and passed back
to `update_data()` before re-fitting.

```python
comb = GVMCombination('correlations/decorrelated.yaml')

# retrieve current setup
info = comb.input_data()

# modify measurement and systematic values
info['data']['measurements']['m1']['central'] = 2.0
info['syst']['sys1']['values']['m1'] = 0.5

# replace the correlation matrix and mark the systematic as dependent
info['syst']['sys1']['type'] = 'dependent'
info['syst']['sys1']['corr'] = np.array([[1.0, 0.3], [0.3, 1.0]])

# update the combination and refit
comb.update_data(info)
comb.fit_results = comb.minimize()
print(f"updated mu_hat={comb.fit_results['mu']:.4f}")
```

## 4. Effect of error-on-error
We now vary the error-on-error parameter ($\epsilon$) from 0 to 0.6 while also testing three different pairs of measurements $(\pm1, \pm2, \pm3)$. For each correlation scenario we plot the fitted central value, confidence interval size and goodness-of-fit.

### Decorrelated
```python
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

eps_grid = np.linspace(0., 0.6, 7)
y_vals = [1, 2, 3]
comb = GVMCombination('correlations/decorrelated.yaml')
base_info = comb.input_data()
cv = {y: [] for y in y_vals}
ci = {y: [] for y in y_vals}
gof = {y: [] for y in y_vals}
for y in y_vals:
    for eps in eps_grid:
        info = deepcopy(base_info)
        info['data']['measurements'][0]['central'] = float(y)
        info['data']['measurements'][1]['central'] = -float(y)
        info['syst']['sys1']['epsilon'] = float(eps)
        comb.update_data(info)
        cv[y].append(comb.fit_results['mu'])
        ci[y].append(comb.confidence_interval()[2])
        gof[y].append(comb.goodness_of_fit())

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, cv[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel('Central value', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("decorrelated_cv.pdf")
plt.show()

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, ci[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel('68% half-size CI', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("decorrelated_ci.pdf")
plt.show()

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, gof[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel(r'$\\chi^2$', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("decorrelated_gof.pdf")
plt.show()
```


### Diagonal correlation
```python
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

eps_grid = np.linspace(0., 0.6, 7)
y_vals = [1, 2, 3]
comb = GVMCombination('correlations/diag_corr.yaml')
base_info = comb.input_data()
cv = {y: [] for y in y_vals}
ci = {y: [] for y in y_vals}
gof = {y: [] for y in y_vals}
for y in y_vals:
    for eps in eps_grid:
        info = deepcopy(base_info)
        info['data']['measurements'][0]['central'] = float(y)
        info['data']['measurements'][1]['central'] = -float(y)
        info['syst']['sys1']['epsilon'] = float(eps)
        comb.update_data(info)
        cv[y].append(comb.fit_results['mu'])
        ci[y].append(comb.confidence_interval()[2])
        gof[y].append(comb.goodness_of_fit())

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, cv[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel('Central value', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("diag_cv.pdf")
plt.show()

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, ci[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel('68% half-size CI', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("diag_ci.pdf")
plt.show()

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, gof[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel(r'$\\chi^2$', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("diag_gof.pdf")
plt.show()
```


### Fully correlated
```python
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

eps_grid = np.linspace(0., 0.6, 7)
y_vals = [1, 2, 3]
comb = GVMCombination('correlations/full_corr.yaml')
base_info = comb.input_data()
cv = {y: [] for y in y_vals}
ci = {y: [] for y in y_vals}
gof = {y: [] for y in y_vals}
for y in y_vals:
    for eps in eps_grid:
        info = deepcopy(base_info)
        info['data']['measurements'][0]['central'] = float(y)
        info['data']['measurements'][1]['central'] = -float(y)
        info['syst']['sys1']['epsilon'] = float(eps)
        comb.update_data(info)
        cv[y].append(comb.fit_results['mu'])
        ci[y].append(comb.confidence_interval()[2])
        gof[y].append(comb.goodness_of_fit())

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, cv[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel('Central value', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("full_cv.pdf")
plt.show()

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, ci[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel('68% half-size CI', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("full_ci.pdf")
plt.show()

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, gof[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel(r'$\\chi^2$', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("full_gof.pdf")
plt.show()
```


### Hybrid correlation
```python
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

eps_grid = np.linspace(0., 0.6, 7)
y_vals = [1, 2, 3]
comb = GVMCombination('correlations/hybrid_corr.yaml')
base_info = comb.input_data()
cv = {y: [] for y in y_vals}
ci = {y: [] for y in y_vals}
gof = {y: [] for y in y_vals}
for y in y_vals:
    for eps in eps_grid:
        info = deepcopy(base_info)
        info['data']['measurements'][0]['central'] = float(y)
        info['data']['measurements'][1]['central'] = -float(y)
        info['syst']['sys1']['epsilon'] = float(eps)
        comb.update_data(info)
        cv[y].append(comb.fit_results['mu'])
        ci[y].append(comb.confidence_interval()[2])
        gof[y].append(comb.goodness_of_fit())

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, cv[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel('Central value', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("hybrid_cv.pdf")
plt.show()

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, ci[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel('68% half-size CI', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("hybrid_ci.pdf")
plt.show()

plt.figure(figsize=(11,7))
for y in y_vals:
    plt.plot(eps_grid, gof[y], '--o', label=f'y=\u00b1{y}')
plt.xlabel(r'$\\epsilon$', fontsize=24)
plt.ylabel(r'$\\chi^2$', fontsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("hybrid_gof.pdf")
plt.show()
```

