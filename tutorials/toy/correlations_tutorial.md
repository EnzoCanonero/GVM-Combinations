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


```python
comb = GVMCombination('correlations/decorrelated.yaml')
mu_hat = comb.fit_results['mu']
ci_low, ci_high, _ = comb.confidence_interval()
chi2 = comb.goodness_of_fit()
print(f'decorrelated: mu_hat={mu_hat:.4f}, CI=({ci_low:.4f}, {ci_high:.4f}), chi2={chi2:.3f}')

```

## 2. Correlation examples
We now compare three different correlation matrices for the systematic uncertainty:
1. **Diagonal**: off-diagonal terms are zero so the systematic acts independently.
2. **Fully correlated**: all coefficients are one so the systematic behaves as one shared nuisance parameter.
3. **Hybrid**: the off-diagonal coefficient is 0.5 representing a partial correlation.



```python
for cfg in ['diag_corr.yaml', 'full_corr.yaml', 'hybrid_corr.yaml']:
    comb = GVMCombination(f'correlations/{cfg}')
    mu_hat = comb.fit_results['mu']
    ci_low, ci_high, _ = comb.confidence_interval()
    chi2 = comb.goodness_of_fit()
    print(f'{cfg}: mu_hat={mu_hat:.4f}, CI=({ci_low:.4f}, {ci_high:.4f}), chi2={chi2:.3f}')

```

## 5. Non-diagonal statistical covariance
The same systematic is fully correlated as above, but the statistical uncertainties are supplied via a covariance matrix with non-zero off-diagonal terms.


```python
comb = GVMCombination('correlations/stat_cov.yaml')
mu_hat = comb.fit_results['mu']
ci_low, ci_high, _ = comb.confidence_interval()
chi2 = comb.goodness_of_fit()
print(f'stat_cov: mu_hat={mu_hat:.4f}, CI=({ci_low:.4f}, {ci_high:.4f}), chi2={chi2:.3f}')

```

> **Tip**
> You can modify the combination without creating a new configuration file by retrieving the current input with `comb.input_data()`, editing the returned dictionary and then calling `comb.update_data()`.
