import os
import numpy as np
import yaml
from iminuit import Minuit
import warnings

class GVMCombination:
    """General combination tool using the Gamma Variance model."""

    def __init__(self, config_file):
        cfg = self._parse_config(config_file)

        self.name = cfg['name']
        self.measurements = cfg['measurements']
        self.n_meas = cfg['n_meas']
        self.n_syst = cfg['n_syst']
        self.y = np.asarray(cfg['data']['central'], dtype=float)
        self.stat = np.asarray(cfg['data']['stat'], dtype=float)

        self.syst = {k: np.asarray(v, dtype=float)
                     for k, v in cfg['data']['systematics'].items()}

        self.corr = {}
        for k, info in cfg['systematics'].items():
            path = info['path']
            if path:
                mat = np.loadtxt(path, dtype=float)
                if not np.allclose(mat, mat.T, rtol=1e-7, atol=1e-8):
                    diff = np.argwhere(~np.isclose(mat, mat.T, rtol=1e-7, atol=1e-8))
                    for i, j in diff:
                        if i < j:
                            warnings.warn(
                                f'Correlation matrix "{k}" asymmetric for measurements '
                                f'{self.measurements[i]} and {self.measurements[j]}: '
                                f'{mat[i, j]} vs {mat[j, i]}')
                self.corr[k] = mat
            else:
                self.corr[k] = np.eye(len(self.y))

        self.uncertain_systematics = {
            k: info['epsilon'] for k, info in cfg['systematics'].items()
            if info['epsilon'] != 0.0
        }

        self._validate_dimensions()

        self.V_inv, self.C_inv, self.Gamma = self._compute_likelihood_matrices()
        self.fit_results = self.minimize()

    # ------------------------------------------------------------------
    def _parse_config(self, path):
        """Parse a YAML configuration file."""

        base_dir = os.path.dirname(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        cfg = {'systematics': {}}
        glob = data.get('globals', {})
        corr_dir = glob.get('corr_dir', '')
        stat_cov_dir = glob.get('stat_cov_dir', '')

        combo = data.get('combination', {})
        cfg['name'] = combo.get('name', '')
        meas_entries = combo.get('measurements', [])
        labels, central, stat_err = [], [], []
        for m in meas_entries:
            labels.append(m['label'])
            central.append(float(m['central']))
            if 'stat_error' in m:
                stat_err.append(float(m['stat_error']))

        stat_cov_path = combo.get('stat_cov_path')
        if stat_cov_path:
            stat_cov_path = stat_cov_path.replace('${globals.corr_dir}', corr_dir)
            stat_cov_path = stat_cov_path.replace('${globals.stat_cov_dir}', stat_cov_dir)
            if not os.path.isabs(stat_cov_path):
                cand = os.path.join(base_dir, stat_cov_path)
                if os.path.exists(cand):
                    stat_cov_path = cand
            stat = np.loadtxt(stat_cov_path, dtype=float)
            if stat.shape != (len(labels), len(labels)):
                raise ValueError('Stat covariance must be %dx%d' % (len(labels), len(labels)))
            if not np.allclose(stat, stat.T, rtol=1e-7, atol=1e-8):
                diff = np.argwhere(~np.isclose(stat, stat.T, rtol=1e-7, atol=1e-8))
                for i, j in diff:
                    if i < j:
                        warnings.warn(
                            f'Stat covariance asymmetric for measurements {labels[i]} and {labels[j]}: '
                            f'{stat[i, j]} vs {stat[j, i]}')
        elif stat_err:
            if len(stat_err) != len(labels):
                raise ValueError(f'Expected {len(labels)} stat errors, found {len(stat_err)}')
            stat = stat_err
        else:
            raise ValueError('Measurement stat errors or covariance required')

        syst_values = {}
        for item in data.get('systematics', []):
            name = item['name']
            shifts = [float(x) for x in item['shifts']]
            if len(shifts) != len(labels):
                raise ValueError(f'Systematic {name} must have {len(labels)} values')
            path_corr = item.get('corr_file')
            if path_corr:
                path_corr = path_corr.replace('${globals.corr_dir}', corr_dir)
                if corr_dir and not os.path.isabs(path_corr):
                    path_corr = os.path.join(corr_dir, path_corr)
            eps = float(item.get('epsilon', 0.0))
            syst_values[name] = shifts
            cfg['systematics'][name] = {'path': path_corr, 'epsilon': eps}

        cfg['measurements'] = labels
        cfg['n_meas'] = len(labels)
        cfg['n_syst'] = len(syst_values)
        cfg['data'] = {'central': central, 'stat': stat, 'systematics': syst_values}
        return cfg

    # ------------------------------------------------------------------
    def _validate_dimensions(self):
        """Validate internal array dimensions against ``n_meas`` and ``n_syst``."""
        if len(self.measurements) != self.n_meas:
            raise ValueError(
                f'Expected {self.n_meas} measurements, got {len(self.measurements)}')

        if len(self.syst) != self.n_syst:
            raise ValueError(
                f'Expected {self.n_syst} systematics, got {len(self.syst)}')

        if self.y.shape[0] != self.n_meas:
            raise ValueError(
                f'Central values vector must have {self.n_meas} elements')

        if self.stat.ndim == 1:
            if self.stat.shape[0] != self.n_meas:
                raise ValueError(
                    f'Stat error vector must have {self.n_meas} elements')
        elif self.stat.ndim == 2:
            if self.stat.shape != (self.n_meas, self.n_meas):
                raise ValueError(
                    f'Stat covariance must be {self.n_meas}x{self.n_meas}')
        else:
            raise ValueError('Stat errors must be a 1D or 2D array')

        for name, arr in self.syst.items():
            if arr.shape[0] != self.n_meas:
                raise ValueError(
                    f'Systematic {name} must have {self.n_meas} values')

        if len(self.corr) != self.n_syst:
            raise ValueError(
                f'Expected {self.n_syst} correlation matrices, got {len(self.corr)}')

        for name, mat in self.corr.items():
            if mat.shape != (self.n_meas, self.n_meas):
                raise ValueError(
                    f'Correlation matrix {name} must be {self.n_meas}x{self.n_meas}')

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
        n = self.y.size
        if self.stat.ndim == 2:
            V_stat = self.stat
        else:
            V_stat = np.diag(self.stat ** 2)
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
    def nll(self, mu, *thetas, y=None, u=None, s=None):
        """Negative log-likelihood."""
        y = self.y if y is None else y
        if u is None:
            u = np.zeros(sum(mat.shape[0] for mat in self.C_inv.values()))
        if s is None:
            s = np.ones(len(self.C_inv))

        thetas = list(thetas)
        seg_u = []
        start = 0
        for key in self.C_inv:
            npar = self.C_inv[key].shape[0]
            seg_u.append(u[start:start + npar])
            start += npar

        adj = np.sum([self.Gamma[k] @ thetas[i]
                      for i, k in enumerate(self.Gamma)], axis=0) if thetas else 0
        v = y - mu - adj
        chi2_y = v @ self.V_inv @ v

        chi2_u = 0.0
        for i, k in enumerate(self.C_inv):
            theta = thetas[i] - seg_u[i]
            Cinv = self.C_inv[k]
            eps = self.uncertain_systematics[k]
            N_s = len(theta)
            if eps > 0:
                chi2_u += (N_s + 1./(2.*eps**2)) * np.log(1. + 2.*eps**2/s[i]*theta @ Cinv @ theta)
            else:
                chi2_u += (theta @ Cinv @ theta) / s[i]
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
        for key in self.C_inv:
            for j in range(self.C_inv[key].shape[0]):
                names.append(f'{key}_{j}')

        initial = [np.mean(self.y)] + [0.] * (len(names) - 1)

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
            for key in self.C_inv:
                npar = self.C_inv[key].shape[0]
                thetas.append(np.array(theta_flat[i:i+npar]))
                i += npar
            nll_val = self.nll(mu, *thetas)
            result = {
                'mu': mu,
                'thetas': np.array(theta_flat),
                'nll': nll_val,
            }
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
            for key in self.C_inv:
                npar = self.C_inv[key].shape[0]
                thetas.append(np.array(theta_flat[i:i+npar]))
                i += npar
            return self.nll(mu, *thetas)

        if hasattr(Minuit, "from_array_func"):
            m = Minuit.from_array_func(
                f,
                x0,
                name=free_names,
                errordef=0.5,
            )
        else:
            # iminuit >=3 removed ``from_array_func``. Use constructor instead.
            m = Minuit(
                f,
                x0,
                name=free_names,
            )
            m.errordef = 0.5
        m.migrad()

        # Collect fitted values
        values = dict(zip(names, initial))
        for val, idx in zip(m.values, free_idx):
            values[names[idx]] = val
        for n, v in fixed.items():
            values[n] = v

        result = {
            'mu': values['mu'],
            'thetas': np.array([values[n] for n in names[1:]]),
            'nll': m.fval,
        }
        if update:
            self.fit_results = result
        return result

    # ------------------------------------------------------------------
    def input_data(self):
        """Return dictionaries summarising the combination input.

        The returned dictionary contains the following keys:

        ``central``
            Mapping measurement name -> central value.
        ``stat_error``
            Mapping measurement name -> statistical error.
        ``stat_corr``
            Mapping (name1, name2) -> statistical correlation coefficient.
        ``systematics``
            Mapping systematic name -> dict of measurement values.
        ``syst_corr``
            Mapping systematic name -> dict keyed by ``(name1, name2)`` with
            correlation coefficients.
        ``epsilon``
            Mapping systematic name -> error-on-error.
        """

        meas = self.measurements
        n = len(meas)

        if self.stat.ndim == 2:
            stat_cov = self.stat
            stat_err = np.sqrt(np.diag(stat_cov))
        else:
            stat_err = np.asarray(self.stat)
            stat_cov = np.diag(stat_err ** 2)

        out = {}
        out['central'] = dict(zip(meas, self.y))
        out['stat_error'] = dict(zip(meas, stat_err))

        stat_corr = {}
        for i in range(n):
            for j in range(n):
                denom = stat_err[i] * stat_err[j]
                rho = stat_cov[i, j] / denom if denom != 0 else 0.0
                stat_corr[(meas[i], meas[j])] = rho
        out['stat_corr'] = stat_corr

        syst_value = {}
        for sname, vals in self.syst.items():
            syst_value[sname] = {m: vals[i] for i, m in enumerate(meas)}
        out['systematics'] = syst_value

        syst_corr = {}
        for sname, rho in self.corr.items():
            pairs = {}
            for i in range(n):
                for j in range(n):
                    pairs[(meas[i], meas[j])] = float(rho[i, j])
            syst_corr[sname] = pairs
        out['syst_corr'] = syst_corr

        epsilon = {k: self.uncertain_systematics.get(k, 0.0)
                   for k in self.syst.keys()}
        out['epsilon'] = epsilon

        return out

    # ------------------------------------------------------------------
    def update_data(self, info):
        """Update combination input from a summary dictionary.

        ``info`` should follow the same structure as returned by
        :meth:`input_summary`.  Only the provided values are updated.
        """

        idx = {m: i for i, m in enumerate(self.measurements)}

        if 'central' in info:
            for m, v in info['central'].items():
                self.y[idx[m]] = float(v)

        if 'stat_error' in info or 'stat_corr' in info:
            n = len(self.measurements)
            errors = {
                m: (np.sqrt(self.stat[i, i]) if self.stat.ndim == 2 else self.stat[i])
                for m, i in idx.items()
            }
            errors.update(info.get('stat_error', {}))
            if 'stat_corr' in info:
                cov = np.zeros((n, n), float)
                for m in self.measurements:
                    cov[idx[m], idx[m]] = errors[m] ** 2
                for (m1, m2), rho in info['stat_corr'].items():
                    s1 = errors[m1]
                    s2 = errors[m2]
                    cov[idx[m1], idx[m2]] = rho * s1 * s2
                    cov[idx[m2], idx[m1]] = cov[idx[m1], idx[m2]]
                self.stat = cov
            else:
                self.stat = np.array([errors[m] for m in self.measurements], float)

        if 'systematics' in info:
            for sname, per_meas in info['systematics'].items():
                if sname not in self.syst:
                    continue
                arr = self.syst[sname]
                for m, val in per_meas.items():
                    arr[idx[m]] = float(val)

        if 'syst_corr' in info:
            for sname, corr_map in info['syst_corr'].items():
                mat = self.corr.get(sname, np.eye(len(self.measurements)))
                for (m1, m2), rho in corr_map.items():
                    i, j = idx[m1], idx[m2]
                    mat[i, j] = mat[j, i] = float(rho)
                self.corr[sname] = np.asarray(mat, float)

        if 'epsilon' in info:
            for s, e in info['epsilon'].items():
                e = float(e)
                if e == 0:
                    self.uncertain_systematics.pop(s, None)
                else:
                    self.uncertain_systematics[s] = e

        self._validate_dimensions()

        # Recompute matrices and refit after updates
        self.V_inv, self.C_inv, self.Gamma = self._compute_likelihood_matrices()
        self.fit_results = self.minimize()

    # ------------------------------------------------------------------
    def likelihood_ratio(self, mu):
        best = self.fit_results or self.minimize()
        nll_best = best['nll'] if 'nll' in best else self.nll(best['mu'], *best['thetas'])
        res_mu = self.minimize(fixed={'mu': mu}, update=False)
        nll_mu = res_mu['nll']
        return 2 * (nll_mu - nll_best)

    # ------------------------------------------------------------------
    def compute_FIM(self, S=None):
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
            S_s = 1.0 if S is None else S[i]
            for j, kp in enumerate(keys):
                idx_p = idxs[j]
                GsVinGp = Gs.T @ V_G[kp]
                if ks == kp:
                    F[np.ix_(idx_s, idx_p)] = GsVinGp + (1./S_s)*Cinv_s
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
            # asymptotic values.  ``self.y`` holds the measured values.
            return 1.0, float(len(self.y) - 1)

        thetas = self.fit_results['thetas']
        keys = list(self.C_inv.keys())
        sizes = [self.C_inv[k].shape[0] for k in keys]
        idx = np.cumsum([0] + sizes)
        thetas = [thetas[idx[i]:idx[i+1]] for i in range(len(keys))]
        eps = [self.uncertain_systematics[k] for k in keys]
        C_inv_list = [self.C_inv[k] for k in keys]
        N_s = sizes
        S = np.array([(1 + 2*e**2 * th @ C @ th)/(1 + 2*e**2*N)
                       for th, C, e, N in zip(thetas, C_inv_list, eps, N_s)])
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
            trWC = np.trace(W_s @ Cinv)
            trWCWC = np.trace(W_s @ Cinv @ W_s @ Cinv)
            trW_t_C = np.trace(Wt_s @ Cinv)
            trW_t_CWC = np.trace(Wt_s @ Cinv @ Wt_s @ Cinv)
            b_lik += (4*e**2/S_s)*trWC - (2*e**2/S_s**2)*trWCWC + (e**2/S_s**2)*(trWC**2)
            b_theta += (4*e**2/S_s)*trW_t_C - (2*e**2/S_s**2)*trW_t_CWC + (e**2/S_s**2)*(trW_t_C**2)
            b_chi2 += (2*N + N**2) * e**2
        b_profile = 1 + b_lik - b_theta
        b_chi2 = float(len(self.y) - 1) + b_chi2 - b_lik
        return b_profile, b_chi2

    # ------------------------------------------------------------------
    def confidence_interval(self, step=0.01, tol=0.001, max_iter=1000):
        b_profile, _ = self.bartlett_correction()
        fit = self.fit_results or self.minimize()
        mu_hat = fit['mu']
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
            mu = fit['mu']
        if thetas is None:
            thetas = fit['thetas']

        # ``fit['thetas']`` is stored as a flat array.  Split it into one
        # array per systematic before calling ``nll``.  Handle the case
        # where no nuisance parameters are present.
        thetas = np.asarray(thetas)
        if thetas.size == 0:
            q = 2 * self.nll(mu)
            _, b_chi2 = self.bartlett_correction()
            return q * (len(self.y) - 1) / b_chi2
        if not isinstance(thetas[0], (list, np.ndarray)):
            keys = list(self.C_inv.keys())
            sizes = [self.C_inv[k].shape[0] for k in keys]
            idx = np.cumsum([0] + sizes)
            thetas = [np.asarray(thetas[idx[i]:idx[i+1]])
                      for i in range(len(keys))]

        q = 2 * self.nll(mu, *thetas)
        _, b_chi2 = self.bartlett_correction()
        return q * (len(self.y) - 1) / b_chi2