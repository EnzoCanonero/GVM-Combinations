# GVM Combination Toolkit

This repository provides a simple implementation of the Gamma Variance Model (GVM) for
combining correlated measurements. The `GVMCombination` class constructs the likelihood,
performs the minimisation with *Minuit*, and computes confidence intervals using an
analytic Bartlett correction.

The likelihood is defined as

$$
\ell_p(\mu,\boldsymbol{\theta}) = -\frac{1}{2}\sum_{i,j=1}^N
\left(y_i-\mu-\sum_{p=1} \Gamma_i^{p}\theta^i_p\right)W_{ij}^{-1}
\left(y_j-\mu-\sum_{p=1} \Gamma_j^{p}\theta^j_p\right)
-\frac{1}{2}\sum_{p=1} \left(N+\frac{1}{2\varepsilon_p^2}\right)
\log\!\left[1 + 2\varepsilon_p^2 \sum_{i,j=1}^N \theta^i_p
\left(\rho^{(p)}\right)_{ij}^{-1} \theta^j_p\right] .
$$

Here, $\mu$ is the parameter of interest, and $\boldsymbol{\theta}$ denotes the nuisance
parameters associated with each systematic source $p$, for which the corresponding
error-on-error $\varepsilon_p$ is greater than zero. The coefficients $\Gamma_i^{p}$ quantify 
the uncertainty induced by systematic source $p$ on measurement $i$, while $\rho^{(p)}$ is 
the correlation matrix describing correlations among measurements due to source $p$.

All other sources of systematic uncertainty, together with the statistical errors,
are encoded in a BLUE-like covariance matrix defined as

$$
W_{ij}=V_{ij}+ \sum_{s\in\{\varepsilon_s=0\}} U_{ij}^{(s)} ,
$$

with

$$
U_{ij}^{(s)}=\rho^{(s)}_{ij}\Gamma_i^{s}\Gamma_j^{s}.
$$

Here, $V$ is the statistical covariance matrix of the measurements,
and $U^{(s)}_{ij}$ is the contribution from systematic source $s$, with
$\rho^{(s)}$ again the corresponding correlation matrix. The sum runs over
systematic sources with vanishing error-on-error $(\varepsilon_s = 0)$; their
nuisance parameters are profiled out, as in a BLUE-like combination.

Further details can be found in \href{https://arxiv.org/abs/2407.05322}{arXiv:2407.05322}.


## Setup

The toolkit requires a few Python packages.  A simple setup script is
provided to install them all::

    ./setup.sh

This installs the dependencies listed in ``requirements.txt`` using
``pip``.

## Usage

```python
from gvm_toolkit import GVMCombination

comb = GVMCombination("path/to/config.yaml")
```

A comprehensive introductory tutorial is available in the [toy](tutorials/toy) folder. The top mass example from the [paper](https://arxiv.org/abs/2407.05322) can be found in the [top-mass](tutorials/top-mass) folder.

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

More information can be found in the [toy](tutorials/toy) folder.