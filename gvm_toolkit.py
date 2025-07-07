import numpy as np
from iminuit import Minuit

class GVMCombination:
    """General combination tool using the Gamma Variance model."""

    def __init__(self, measurements, stat, systematics, correlations,
                 uncertain_systematics=None):
        self.y = np.asarray(measurements, dtype=float)
        self.stat = np.asarray(stat, dtype=float)
        self.syst = {k: np.asarray(v, dtype=float) for k, v in systematics.items()}
        self.corr = {k: np.asarray(m, dtype=float) for k, m in correlations.items()}
        self.uncertain_systematics = uncertain_systematics or {}

        self.V_inv, self.C_inv, self.Gamma = self._compute_likelihood_matrices()
        self.fit_results = self.minimize()

    # ------------------------------------------------------------------
    def _reduce_corr(self, rho):
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
            np.fill_diagonal(reduced, reduced.diagonal() + abs(m) + 0.01)
        return reduced, Gamma

    # ------------------------------------------------------------------
    def _compute_likelihood_matrices(self):
        n = self.y.size
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
                red, Gamma = self._reduce_corr(rho)
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

        def f(arr):
            params = list(initial)
            for val, idx in zip(arr, free_idx):
                params[idx] = val
            for n, val in fixed.items():
                params[names.index(n)] = val
            mu = params[0]
            thetas = params[1:]
            return self.nll(mu, *thetas)

        m = Minuit.from_array_func(
            f,
            x0,
            name=free_names,
            errordef=0.5,
        )
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
    def likelihood_ratio(self, mu):
        best = self.fit_results or self.minimize()
        nll_best = best['nll'] if 'nll' in best else self.nll(best['mu'], *best['thetas'])
        res_mu = self.minimize(fixed={'mu': mu}, update=False)
        nll_mu = res_mu['nll']
        return 2 * (nll_mu - nll_best)

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
        return -2 * self.nll(mu, *thetas)

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
        b_lik = b_theta = 0
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
        b_profile = 1 + b_lik - b_theta
        return b_profile

    # ------------------------------------------------------------------
    def confidence_interval(self, step=0.01, tol=0.001, max_iter=1000):
        b = self.bartlett_correction()
        fit = self.fit_results or self.minimize()
        mu_hat = fit['mu']
        q0 = self.likelihood_ratio(mu_hat)
        up = mu_hat
        q_up = q0
        down = mu_hat
        q_down = q0
        it = 0
        while q_up <= b and it < max_iter:
            up += step
            q_up = self.likelihood_ratio(up)
            it += 1
        it = 0
        while q_down <= b and it < max_iter:
            down -= step
            q_down = self.likelihood_ratio(down)
            it += 1
        step /= 2
        it = 0
        while abs(q_up - b) > tol and it < max_iter:
            if q_up > b:
                up -= step
            else:
                up += step
            q_up = self.likelihood_ratio(up)
            step /= 2
            it += 1
        step = step if step > 0 else 0.001
        it = 0
        while abs(q_down - b) > tol and it < max_iter:
            if q_down > b:
                down += step
            else:
                down -= step
            q_down = self.likelihood_ratio(down)
            step /= 2
            it += 1
        return down, up, 0.5*(up - down)
