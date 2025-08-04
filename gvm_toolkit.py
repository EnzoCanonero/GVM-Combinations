import os
import numpy as np
import yaml
from iminuit import Minuit
import warnings

class GVMCombination:
    """General combination tool using the Gamma Variance model."""

    def __init__(self, config_file):
        cfg = self._parse_config(config_file)

        glob = cfg['global']
        self.name = glob['name']
        self.n_meas = glob['n_meas']
        self.n_syst = glob['n_syst']

        meas_dict = cfg['data']['measurements']
        self.measurements = list(meas_dict.keys())
        self.y = np.array([meas_dict[m]['central'] for m in self.measurements],
                          dtype=float)
        self.V_stat = np.asarray(cfg['data']['V_stat'], dtype=float)

        self.syst = {
            sname: np.array([cfg['syst'][sname]['values'][m]
                             for m in self.measurements], dtype=float)
            for sname in cfg['syst']
        }

        self.corr = {s: np.asarray(cfg['syst'][s]['corr'], dtype=float)
                     for s in cfg['syst']}

        self.eoe_type = {
            s: cfg['syst'][s]['error-on-error']['type']
            for s in cfg['syst']
        }

        self.uncertain_systematics = {
            s: cfg['syst'][s]['error-on-error']['value']
            for s in cfg['syst'] if cfg['syst'][s]['error-on-error']['value'] != 0.0
        }

        self._validate_combination()

        self.V_inv, self.C_inv, self.Gamma = self._compute_likelihood_matrices()
        self.fit_results = self.minimize()

    # ------------------------------------------------------------------
    def _parse_config(self, path):
        """Parse a YAML configuration file."""

        base_dir = os.path.dirname(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        cfg = {}
        try:
            glob = data['global']
            corr_dir = glob.get('corr_dir', '')
            name = glob['name']
            n_meas = int(glob['n_meas'])
            n_syst = int(glob['n_syst'])
        except KeyError as exc:
            raise KeyError(
                'Global configuration must define "name", "n_meas" and "n_syst"'
            ) from exc

        cfg['global'] = {
            'name': name,
            'n_meas': n_meas,
            'n_syst': n_syst,
        }

        try:
            combo = data['data']
            meas_entries = combo['measurements']
        except KeyError as exc:
            raise KeyError('Data configuration must define "measurements"') from exc
        labels, central, stat_err = [], [], []
        for m in meas_entries:
            try:
                label = m['label']
            except KeyError as exc:
                raise KeyError('Each measurement requires a "label"') from exc
            try:
                cent = m['central']
            except KeyError as exc:
                raise KeyError(
                    f'Measurement "{label}" must define "central"') from exc
            labels.append(label)
            central.append(float(cent))
            if 'stat_error' in m:
                stat_err.append(float(m['stat_error']))

        stat_cov_path = combo.get('stat_cov_path')
        if stat_cov_path:
            stat_cov_path = stat_cov_path.replace('${global.corr_dir}', corr_dir)
            if not os.path.isabs(stat_cov_path):
                stat_cov_path = os.path.join(corr_dir, stat_cov_path)
            V_stat = np.loadtxt(stat_cov_path, dtype=float)
        elif stat_err:
            V_stat = np.diag(np.array(stat_err, dtype=float) ** 2)
        else:
            raise KeyError('Measurement stat errors or covariance required')

        try:
            syst_entries = data['syst']
        except KeyError as exc:
            raise KeyError('Configuration must define "syst" section') from exc

        meas_map = {m: i for i, m in enumerate(labels)}
        syst_dict = {}
        for item in syst_entries:
            name = item['name']

            try:
                shift = item['shift']
                shift_vals = shift['value']
            except KeyError as exc:
                raise KeyError(
                    f'Systematic "{name}" must define "shift.value"'
                ) from exc
            shifts = [float(x) for x in shift_vals]

            try:
                corr_spec = shift['correlation']
            except KeyError as exc:
                raise KeyError(
                    f'Systematic "{name}" must define "shift.correlation"'
                ) from exc
            if corr_spec == 'diagonal':
                corr = np.eye(n_meas)
            elif corr_spec == 'ones':
                corr = np.ones((n_meas, n_meas))
            else:
                path_corr = corr_spec.replace('${global.corr_dir}', corr_dir)
                if not os.path.isabs(path_corr):
                    path_corr = os.path.join(corr_dir, path_corr)
                corr = np.loadtxt(path_corr, dtype=float)

            try:
                eoe = item['error-on-error']
                eps_val = float(eoe['value'])
                eps_type = eoe['type']
            except KeyError as exc:
                raise KeyError(
                    f'Systematic "{name}" must define "error-on-error.value" and "error-on-error.type"'
                ) from exc
            if eps_type not in ('dependent', 'independent'):
                raise ValueError(
                    f'Systematic "{name}" has unrecognised error-on-error type "{eps_type}"'
                )

            val_map = {lab: shifts[meas_map[lab]] for lab in labels}
            syst_dict[name] = {
                'values': val_map,
                'error-on-error': {'value': eps_val, 'type': eps_type},
                'corr': corr,
            }

        meas_data = {
            lab: {'central': c, 'stat': np.sqrt(V_stat[i, i])}
            for i, (lab, c) in enumerate(zip(labels, central))
        }
        cfg['data'] = {'measurements': meas_data, 'V_stat': V_stat}
        cfg['syst'] = syst_dict
        return cfg

    # ------------------------------------------------------------------
    def _validate_combination(self):
        """Validate consistency of the combination inputs."""
        if len(self.measurements) != self.n_meas:
            raise ValueError(
                f'Expected {self.n_meas} measurements, got {len(self.measurements)}')

        if len(self.syst) != self.n_syst:
            raise ValueError(
                f'Expected {self.n_syst} systematics, got {len(self.syst)}')

        if self.y.shape[0] != self.n_meas:
            raise ValueError(
                f'Central values vector must have {self.n_meas} elements')

        if self.V_stat.shape != (self.n_meas, self.n_meas):
            raise ValueError(
                f'Stat covariance must be {self.n_meas}x{self.n_meas}')
        diff = np.argwhere(~np.isclose(self.V_stat, self.V_stat.T, rtol=1e-7, atol=1e-8))
        for i, j in diff:
            if i < j:
                warnings.warn(
                    f'Stat covariance asymmetric for measurements '
                    f'{self.measurements[i]} and {self.measurements[j]}: '
                    f'{self.V_stat[i, j]} vs {self.V_stat[j, i]}')

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
            diff = np.argwhere(~np.isclose(mat, mat.T, rtol=1e-7, atol=1e-8))
            for i, j in diff:
                if i < j:
                    warnings.warn(
                        f'Correlation matrix "{name}" asymmetric for measurements '
                        f'{self.measurements[i]} and {self.measurements[j]}: '
                        f'{mat[i, j]} vs {mat[j, i]}')
            if self.eoe_type.get(name, 'dependent') == 'independent':
                if not np.allclose(mat, np.eye(self.n_meas)):
                    raise ValueError(
                        f'Systematic {name} has independent error-on-error but correlation is not diagonal')

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
    def nll(self, mu, *thetas):
        """Negative log-likelihood."""
        thetas = list(thetas)

        adj = np.sum([self.Gamma[k] @ thetas[i]
                      for i, k in enumerate(self.Gamma)], axis=0) if thetas else 0
        v = self.y - mu - adj
        chi2_y = v @ self.V_inv @ v

        chi2_u = 0.0
        keys = list(self.Gamma.keys())
        for i, k in enumerate(keys):
            theta = thetas[i]
            eps = self.uncertain_systematics[k]
            if self.eoe_type.get(k, 'dependent') == 'dependent':
                Cinv = self.C_inv[k]
                N_s = len(theta)
                if eps > 0:
                    chi2_u += (N_s + 1.0 / (2.0 * eps ** 2)) * np.log(1. + 2. * eps ** 2 * theta @ Cinv @ theta)
                else:
                    chi2_u += theta @ Cinv @ theta
            else:
                if eps > 0:
                    chi2_u += np.sum((1 + 1.0 / (2.0 * eps ** 2)) * np.log(1. + 2. * eps ** 2 * theta ** 2))
                else:
                    chi2_u += theta @ theta
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
            for key in self.Gamma:
                npar = self.Gamma[key].shape[1]
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
            for key in self.Gamma:
                npar = self.Gamma[key].shape[1]
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
        """Return a dictionary summarising the current combination input."""

        meas = self.measurements

        cfg = {
            'global': {
                'name': self.name,
                'n_meas': self.n_meas,
                'n_syst': self.n_syst,
            },
            'data': {
                'measurements': {
                    m: {
                        'central': float(self.y[i]),
                        'stat': float(np.sqrt(self.V_stat[i, i]))
                    }
                    for i, m in enumerate(meas)
                },
                'V_stat': self.V_stat.copy(),
            },
            'syst': {
                sname: {
                    'shift': {
                        'value': {
                            m: float(self.syst[sname][i])
                            for i, m in enumerate(meas)
                        },
                        'correlation': self.corr[sname].copy(),
                    },
                    'error-on-error': {
                        'value': self.uncertain_systematics.get(sname, 0.0),
                        'type': self.eoe_type[sname],
                    },
                }
                for sname in self.syst
            },
        }

        return cfg

    # ------------------------------------------------------------------
    def update_data(self, info):
        """Update combination input from a summary dictionary.

        ``info`` should follow the same structure as returned by
        :meth:`input_data`.  Only the provided values are updated.
        """

        idx = {m: i for i, m in enumerate(self.measurements)}

        data = info.get('data', {})
        meas_info = data.get('measurements', {})
        for m, vals in meas_info.items():
            if 'central' in vals:
                self.y[idx[m]] = float(vals['central'])
            if 'stat' in vals:
                self.V_stat[idx[m], idx[m]] = float(vals['stat']) ** 2

        if 'V_stat' in data:
            V_stat = np.asarray(data['V_stat'], dtype=float)
            self.V_stat = V_stat

        syst_info = info.get('syst', {})
        for sname, entry in syst_info.items():
            if sname not in self.syst:
                continue

            shift = entry.get('shift')
            if shift:
                if 'value' in shift:
                    for m, val in shift['value'].items():
                        self.syst[sname][idx[m]] = float(val)
                if 'correlation' in shift:
                    mat = np.asarray(shift['correlation'], dtype=float)
                    self.corr[sname] = mat

            if 'error-on-error' in entry:
                eoe = entry['error-on-error']
                if 'type' in eoe:
                    if eoe['type'] not in ('dependent', 'independent'):
                        raise ValueError(
                            f'Systematic "{sname}" has unrecognised error-on-error type "{eoe["type"]}"'
                        )
                    self.eoe_type[sname] = eoe['type']
                if 'value' in eoe:
                    e = float(eoe['value'])
                    if e == 0:
                        self.uncertain_systematics.pop(sname, None)
                    else:
                        self.uncertain_systematics[sname] = e

        self._validate_combination()

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
                b_lik += np.sum((4 * e ** 2 / S_s) * diag_W - (e ** 2 / S_s ** 2) * diag_W_sq)
                b_theta += np.sum((4 * e ** 2 / S_s) * diag_Wt - (e ** 2 / S_s ** 2) * diag_Wt_sq)
                b_chi2 += 3 * N * e ** 2
                
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
