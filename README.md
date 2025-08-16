# GVM Combination Toolkit

This repository contains a simple implementation of the Gamma Variance Model (GVM) for
combining correlated measurements.  The provided `GVMCombination` class builds
the likelihood, minimises it with *Minuit* and computes confidence intervals
using an analytic Bartlett correction.

The toolkit constructs the likelihood

$$
\ell_p(\mu,\boldsymbol{\theta}) = -\frac{1}{2}\sum_{i,j=1}^N
\left(y_i-\mu-\sum_{s=1}^M \Gamma_i^{s}\theta^i_s\right)W_{ij}^{-1}
\left(y_j-\mu-\sum_{s=1}^M \Gamma_j^{s}\theta^j_s\right)
-\frac{1}{2}\sum_{s=1}^M \left(N+\frac{1}{2\varepsilon_s^2}\right)
\log\!\left[1 + 2\varepsilon_s^2 \sum_{i,j=1}^N \theta^i_s
\left(\rho^{(s)}\right)_{ij}^{-1} \theta^j_s\right]\,.
$$


Here $\mu$ is the parameter of interest, $\boldsymbol{\theta}$ are nuisance parameters for each systematic source $s$, and $\Gamma_i^{s}$ encodes the effect of $s$ on measurement $i$. The covariance matrix entering the likelihood is defined as

$$
W_{ij}=V_{ij}+ \sum_{s\in\{\varepsilon_s=0\}} U_{ij}^{(s)}\, ,
$$

with

$$
U_{ij}^{(s)}=\Gamma_i^{s}\Gamma_j^{s}\sigma_{u_s}^2\, ,
$$

and $V$ the covariance matrix among measurements. The sum runs over systematics whose error-on-error $\varepsilon_s$ vanishes; their nuisance parameters are profiled out in a BLUE-like combination. $\rho^{(s)}$ is the correlation matrix due to source $s$. More details can be found in [arXiv:2407.05322](https://arxiv.org/abs/2407.05322).

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
info['data']['measurements']['ATLAS']['central'] = 173.4
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
* ``syst`` – list of systematics. Each has a ``shift`` block with ``value``
  listing the shifts for each measurement and ``correlation`` specifying
  ``diagonal``, ``ones`` or a path to a correlation matrix. An optional
  ``error-on-error`` block specifies the ``value`` and ``type``
  (``dependent`` or ``independent``). For ``independent`` systematics the
  ``value`` may be either a list giving one epsilon per measurement or a
  single number which is applied to all measurements. If omitted, the value
  defaults to ``0`` and the type to ``dependent``.
A minimal configuration:

```yaml
global:
  name: Example
  n_meas: 2
  n_syst: 1

data:
  measurements:
    - label: A
      central: 1.0
      stat_error: 0.1
    - label: B
      central: 1.2
      stat_error: 0.1

syst:
  - name: scale
    shift:
      value: [0.05, -0.02]
      correlation: diagonal
    error-on-error:
      value: 0.02
      type: dependent
```


See ``input_files/LHC_mass_combination.yaml`` for a complete example using
statistical errors.  ``input_files/LHC_mass_combination_cov.yaml`` shows the
equivalent setup with a statistical covariance matrix.
