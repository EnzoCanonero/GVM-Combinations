import os
import numpy as np
import yaml
import warnings
from .fit_results import FitResult
from .minuit_wrapper import minimize as _minimize
from .config import (
    validate_input_data,
    input_data as InputData,
)

class GVMCombination:
    """General combination tool using the Gamma Variance model."""

    def __init__(self, data):
        # Accept pre-built input_data (from gvm.config)
        input_data = data
        validate_input_data(input_data)

        self._input_data = input_data

        # prepared empty state
        self.V_inv = None
        self.C_inv = {}
        self.Gamma = {}
        self.fit_results = None
        
    # from_yaml removed. Build input via gvm.config and pass to constructor.
    
    def prepare(self):
        """Validate inputs and compute internal matrices for likelihood.
        This must be called before calling likelihood or fitting methods
        if the instance was not prepared yet.
        """
        validate_input_data(self._input_data)
        self.V_inv, self.C_inv, self.Gamma = self._compute_likelihood_matrices()
        return self
    
    def fit(self, fixed=None, update=True):
        """Run the minimisation and store the result in ``self.fit_results``.
        This is a convenience wrapper around ``minimize`` that ensures the
        instance is prepared before fitting.
        """
        if self.V_inv is None or not self.Gamma:
            self.prepare()
        return self.minimize(fixed=fixed, update=update)

    # ------------------------------------------------------------------
    # Input-data accessors
    # ------------------------------------------------------------------
    @property
    def name(self):
        return self._input_data.name

    @property
    def n_meas(self):
        return self._input_data.n_meas

    @property
    def n_syst(self):
        return self._input_data.n_syst

    @property
    def measurements(self):
        return self._input_data.measurements

    @property
    def V_stat(self):
        return self._input_data.V_stat

    @property
    def syst(self):
        return self._input_data.syst

    @property
    def corr(self):
        return self._input_data.corr

    @property
    def eoe_type(self):
        return self._input_data.eoe_type

    @property
    def uncertain_systematics(self):
        return self._input_data.uncertain_systematics

    def get_input_data(self, copy: bool = False):
        """Return the current input_data object.

        If ``copy`` is True, return a shallow copy with arrays copied to
        avoid in-place external mutations affecting the combination.
        """
        if not copy:
            return self._input_data
        s = self._input_data
        from dataclasses import replace
        return replace(
            s,
            measurements=dict(s.measurements),
            V_stat=np.array(s.V_stat, copy=True),
            syst={k: np.array(v, copy=True) for k, v in s.syst.items()},
            corr={k: np.array(v, copy=True) for k, v in s.corr.items()},
            eoe_type=dict(s.eoe_type),
            uncertain_systematics={k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                                   for k, v in s.uncertain_systematics.items()},
        )

    def set_input_data(self, data, refit: bool = True):
        """Replace the combination input with a new input_data object.

        Validates, rebuilds internal matrices, and optionally refits.
        """
        validate_input_data(data)
        self._input_data = data
        self.V_inv, self.C_inv, self.Gamma = self._compute_likelihood_matrices()
        if refit:
            self.fit_results = self.minimize()
        return self
    
    # ------------------------------------------------------------------
    # Build Likelihood Matrices
    # ------------------------------------------------------------------
    def _reduce_corr(self, rho, src_name=None):
        n = rho.shape[0]
        groups = []
        visited = set()
        for i in range(n):
            if i not in visited:
                group = [i]
                for j in range(i + 1, n):
                    if abs(rho[i, j]) == 1:
                        group.append(j)
                        visited.add(j)
                groups.append(group)
                visited.add(i)

        rsize = len(groups)
        reduced = np.zeros((rsize, rsize))
        Gamma = np.zeros((n, rsize))

        for new_i, group in enumerate(groups):
            for j in group:
                sign = 1.0
                if rho[group[0], j] == -1:
                    sign = -1.0
                Gamma[j, new_i] = sign

        for new_i, gi in enumerate(groups):
            for new_j, gj in enumerate(groups):
                vec = [rho[i, j] * Gamma[i, new_i] * Gamma[j, new_j]
                       for i in gi for j in gj]
                reduced[new_i, new_j] = np.mean(vec)

        eig = np.linalg.eigvalsh(reduced)
        m = eig.min()
        if m <= 0:
            offset = abs(m) + 0.01
            np.fill_diagonal(reduced, reduced.diagonal() + offset)
            if src_name:
                warnings.warn(
                    f'Negative eigenvalue {m:.4e} in systematic "{src_name}"; '
                    f'adding {offset:.4e} to diagonal for regularisation.')
        return reduced, Gamma

    # ------------------------------------------------------------------
    def _compute_likelihood_matrices(self):
        n = len(self.measurements)
        V_stat = self.V_stat
        V_syst = np.zeros((n, n))
        for src, rho in self.corr.items():
            if src not in self.uncertain_systematics:
                sigma = self.syst[src]
                V_syst += np.outer(sigma, sigma) * rho
        V_blue = V_stat + V_syst
        V_inv = np.linalg.inv(V_blue)

        C_inv = {}
        Gamma_factors = {}
        for src, sigma in self.syst.items():
            if src in self.uncertain_systematics:
                rho = self.corr[src]
                red, Gamma = self._reduce_corr(rho, src_name=src)
                for i in range(Gamma.shape[0]):
                    for j in range(Gamma.shape[1]):
                        if Gamma[i, j] != 0:
                            Gamma[i, j] *= sigma[i]
                zero_cols = np.all(Gamma == 0, axis=0)
                Gamma = Gamma[:, ~zero_cols]
                if np.any(zero_cols):
                    red = red[~zero_cols][:, ~zero_cols]
                C_inv[src] = np.linalg.inv(red)
                Gamma_factors[src] = Gamma
        return V_inv, C_inv, Gamma_factors
    
    # ------------------------------------------------------------------
    # Config End
    # ------------------------------------------------------------------
    def nll(self, mu, *thetas):
        """Negative log-likelihood."""
        thetas = list(thetas)

        adj = np.sum([self.Gamma[k] @ thetas[i]
                      for i, k in enumerate(self.Gamma)], axis=0) if thetas else 0
        y_vals = np.fromiter(self.measurements.values(), dtype=float)
        v = y_vals - mu - adj
        chi2_y = v @ self.V_inv @ v

        chi2_u = 0.0
        keys = list(self.Gamma.keys())
        for i, k in enumerate(keys):
            theta = np.asarray(thetas[i])
            eps = self.uncertain_systematics[k]
            if self.eoe_type.get(k, 'dependent') == 'dependent':
                Cinv = self.C_inv[k]
                N_s = len(theta)
                if eps > 0:
                    chi2_u += (N_s + 1.0 / (2.0 * eps ** 2)) * np.log(1. + 2. * eps ** 2 * theta @ Cinv @ theta)
                else:
                    chi2_u += theta @ Cinv @ theta
            else:
                eps = np.asarray(eps)
                mask = eps > 0
                if np.any(mask):
                    chi2_u += np.sum((1 + 1.0 / (2.0 * eps[mask] ** 2)) * np.log(1. + 2. * eps[mask] ** 2 * theta[mask] ** 2))
                if np.any(~mask):
                    chi2_u += np.sum(theta[~mask] ** 2)
        return 0.5 * (chi2_y + chi2_u)

    # ------------------------------------------------------------------
    def minimize(self, fixed=None, update=True):
        """Minimise the negative log-likelihood.

        Parameters
        ----------
        fixed : dict, optional
            Dictionary mapping parameter names to fixed values.  Any parameter
            not listed here is treated as free.
        update : bool, optional
            If True, store the fit result in ``self.fit_results``.
        """
        fixed = fixed or {}

        # Build full parameter list
        names = ['mu']
        for key in self.Gamma:
            for j in range(self.Gamma[key].shape[1]):
                names.append(f'{key}_{j}')

        y_vals = np.fromiter(self.measurements.values(), dtype=float)
        initial = [np.mean(y_vals)] + [0.] * (len(names) - 1)

        # Determine which parameters are free
        free_idx = []
        free_names = []
        x0 = []
        for i, n in enumerate(names):
            if n not in fixed:
                free_idx.append(i)
                free_names.append(n)
                x0.append(initial[i])

        # If no parameters remain free after applying the fixed values,
        # directly evaluate the negative log-likelihood without calling a
        # minimiser.
        if len(x0) == 0:
            params = list(initial)
            for n, val in fixed.items():
                params[names.index(n)] = val
            mu = params[0]
            theta_flat = params[1:]
            thetas = []
            i = 0
            for key in self.Gamma:
                npar = self.Gamma[key].shape[1]
                thetas.append(np.array(theta_flat[i:i+npar]))
                i += npar
            nll_val = self.nll(mu, *thetas)
            result = FitResult(
                mu=mu,
                thetas=np.array(theta_flat),
                nll=nll_val,
            )
            if update:
                self.fit_results = result
            return result

        def f(arr):
            params = list(initial)
            for val, idx in zip(arr, free_idx):
                params[idx] = val
            for n, val in fixed.items():
                params[names.index(n)] = val
            mu = params[0]
            theta_flat = params[1:]
            thetas = []
            i = 0
            for key in self.Gamma:
                npar = self.Gamma[key].shape[1]
                thetas.append(np.array(theta_flat[i:i+npar]))
                i += npar
            return self.nll(mu, *thetas)

        m = _minimize(f, x0, free_names, errordef=0.5)

        # Collect fitted values
        values = dict(zip(names, initial))
        for val, idx in zip(m.values, free_idx):
            values[names[idx]] = val
        for n, v in fixed.items():
            values[n] = v

        result = FitResult(
            mu=values['mu'],
            thetas=np.array([values[n] for n in names[1:]]),
            nll=m.fval,
        )
        if update:
            self.fit_results = result
        return result

    # ------------------------------------------------------------------
    def likelihood_ratio(self, mu):
        best = self.fit_results or self.minimize()
        nll_best = best.nll if isinstance(best, FitResult) else self.nll(best['mu'], *best['thetas'])
        res_mu = self.minimize(fixed={'mu': mu}, update=False)
        nll_mu = res_mu.nll if isinstance(res_mu, FitResult) else res_mu['nll']
        return 2 * (nll_mu - nll_best)

    # ------------------------------------------------------------------
    def compute_FIM(self, S):
        keys = list(self.C_inv.keys())
        sizes = [self.C_inv[k].shape[0] for k in keys]

        tot = sum(sizes)
        F = np.zeros((1 + tot, 1 + tot))
        F[0, 0] = np.sum(self.V_inv)
        start_idx = np.cumsum([0] + sizes[:-1])
        idxs = [np.arange(sz) + s + 1 for sz, s in zip(sizes, start_idx)]

        V_G = {k: self.V_inv @ self.Gamma[k] for k in keys}
        for i, k in enumerate(keys):
            idx = idxs[i]
            F[0, idx] = F[idx, 0] = V_G[k].sum(axis=0)

        for i, ks in enumerate(keys):
            idx_s = idxs[i]
            Gs = self.Gamma[ks]
            Cinv_s = self.C_inv[ks]
            S_s = S[i]
            for j, kp in enumerate(keys):
                idx_p = idxs[j]
                GsVinGp = Gs.T @ V_G[kp]
                if ks == kp:
                    if self.eoe_type.get(kp, 'dependent') == 'dependent':
                        F[np.ix_(idx_s, idx_p)] = GsVinGp + (1.0 / S_s) * Cinv_s
                    else:
                        F[np.ix_(idx_s, idx_p)] = (
                            GsVinGp + Cinv_s * (1.0 / S_s)[:, None]
                        )
                else:
                    F[np.ix_(idx_s, idx_p)] = GsVinGp
        return F

    # ------------------------------------------------------------------
    def bartlett_correction(self):
        """Return Bartlett corrections for the profile likelihood ratio and
        goodness-of-fit statistics.

        Returns
        -------
        tuple of float
            ``(b_profile, b_chi2)`` where ``b_profile`` rescales the likelihood
            ratio used for confidence intervals and ``b_chi2`` rescales the
            goodness-of-fit statistic.
        """

        if len(self.C_inv) == 0:
            # No nuisance parameters: both corrections reduce to their
            # asymptotic values.  ``self.measurements`` holds the measured values.
            return 1.0, float(len(self.measurements) - 1)

        thetas = self.fit_results.thetas if isinstance(self.fit_results, FitResult) else self.fit_results['thetas']
        keys = list(self.C_inv.keys())
        sizes = [self.C_inv[k].shape[0] for k in keys]
        idx = np.cumsum([0] + sizes)
        thetas = [np.asarray(thetas[idx[i]:idx[i+1]]) for i in range(len(keys))]
        eps = [np.asarray(self.uncertain_systematics[k], dtype=float) for k in keys]
        C_inv_list = [self.C_inv[k] for k in keys]
        N_s = sizes
        S = []
        for th, k, C, e, N in zip(thetas, keys, C_inv_list, eps, N_s):
            if self.eoe_type.get(k, 'dependent') == 'dependent':
                S.append((1 + 2 * e ** 2 * th @ C @ th) / (1 + 2 * e ** 2 * N))
            else:
                S.append((1 + 2 * e ** 2 * th ** 2) / (1 + 2 * e ** 2))
        F = self.compute_FIM(S)
        W_full = np.linalg.inv(F)[1:, 1:]
        W_theta = np.linalg.inv(F[1:, 1:])
        start_idx = idx[:-1]
        b_lik = b_theta = b_chi2 = 0.0
        for s, (th, Cinv, e, N, S_s) in enumerate(zip(thetas, C_inv_list, eps, N_s, S)):
            si = start_idx[s]
            ei = si + N
            W_s = W_full[si:ei, si:ei]
            Wt_s = W_theta[si:ei, si:ei]
            if self.eoe_type.get(keys[s], 'dependent') == 'dependent':
                trWC = np.trace(W_s @ Cinv)
                trWCWC = np.trace(W_s @ Cinv @ W_s @ Cinv)
                trW_t_C = np.trace(Wt_s @ Cinv)
                trW_t_CWC = np.trace(Wt_s @ Cinv @ Wt_s @ Cinv)
                b_lik += (
                    (4 * e ** 2 / S_s) * trWC
                    - (2 * e ** 2 / S_s ** 2) * trWCWC
                    + (e ** 2 / S_s ** 2) * (trWC ** 2)
                )
                b_theta += (
                    (4 * e ** 2 / S_s) * trW_t_C
                    - (2 * e ** 2 / S_s ** 2) * trW_t_CWC
                    + (e ** 2 / S_s ** 2) * (trW_t_C ** 2)
                )
                b_chi2 += (2 * N + N ** 2) * e ** 2
            else:
                diag_W = np.diag(W_s)
                diag_W_sq = diag_W ** 2
                diag_Wt = np.diag(Wt_s)
                diag_Wt_sq = diag_Wt ** 2
                e2 = e ** 2
                b_lik += np.sum((4 * e2 / S_s) * diag_W - (e2 / S_s ** 2) * diag_W_sq)
                b_theta += np.sum((4 * e2 / S_s) * diag_Wt - (e2 / S_s ** 2) * diag_Wt_sq)
                b_chi2 += np.sum(3 * e2)
                
        b_profile = 1 + b_lik - b_theta
        b_chi2 = float(len(self.measurements) - 1) + b_chi2 - b_lik
        return b_profile, b_chi2

    # ------------------------------------------------------------------
    def confidence_interval(self, step=0.01, tol=0.001, max_iter=1000):
        b_profile, _ = self.bartlett_correction()
        fit = self.fit_results or self.minimize()
        mu_hat = fit.mu if isinstance(fit, FitResult) else fit['mu']
        q0 = self.likelihood_ratio(mu_hat)
        up = mu_hat
        q_up = q0
        down = mu_hat
        q_down = q0
        it = 0
        while q_up <= b_profile and it < max_iter:
            up += step
            q_up = self.likelihood_ratio(up)
            it += 1
        it = 0
        while q_down <= b_profile and it < max_iter:
            down -= step
            q_down = self.likelihood_ratio(down)
            it += 1
        step /= 2
        it = 0
        while abs(q_up - b_profile) > tol and it < max_iter:
            if q_up > b_profile:
                up -= step
            else:
                up += step
            q_up = self.likelihood_ratio(up)
            step /= 2
            it += 1
        step = step if step > 0 else 0.001
        it = 0
        while abs(q_down - b_profile) > tol and it < max_iter:
            if q_down > b_profile:
                down += step
            else:
                down -= step
            q_down = self.likelihood_ratio(down)
            step /= 2
            it += 1
        return down, up, 0.5*(up - down)
    
    # ------------------------------------------------------------------
    def goodness_of_fit(self, mu=None, thetas=None):
        """Return the goodness-of-fit ``-2 * nll``.

        Parameters
        ----------
        mu : float, optional
            Mean parameter.  If ``None`` use the fitted value.
        thetas : array-like, optional
            Nuisance parameters.  If ``None`` use the fitted values.
        """
        fit = self.fit_results if self.fit_results else self.minimize()
        if mu is None:
            mu = fit.mu if isinstance(fit, FitResult) else fit['mu']
        if thetas is None:
            thetas = fit.thetas if isinstance(fit, FitResult) else fit['thetas']

        # ``fit['thetas']`` is stored as a flat array.  Split it into one
        # array per systematic before calling ``nll``.  Handle the case
        # where no nuisance parameters are present.
        thetas = np.asarray(thetas)
        if thetas.size == 0:
            q = 2 * self.nll(mu)
            _, b_chi2 = self.bartlett_correction()
            return q * (len(self.measurements) - 1) / b_chi2
        if not isinstance(thetas[0], (list, np.ndarray)):
            keys = list(self.C_inv.keys())
            sizes = [self.C_inv[k].shape[0] for k in keys]
            idx = np.cumsum([0] + sizes)
            thetas = [np.asarray(thetas[idx[i]:idx[i+1]])
                      for i in range(len(keys))]

        q = 2 * self.nll(mu, *thetas)
        _, b_chi2 = self.bartlett_correction()
        return q * (len(self.measurements) - 1) / b_chi2
