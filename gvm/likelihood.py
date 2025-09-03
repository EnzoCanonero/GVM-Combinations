# Likelihood utilities: NLL, Fisher information, Bartlett corrections.
import numpy as np
from .fit_results import FitResult


def nll(comb, mu, *thetas):
    """Negative log-likelihood for a combination instance.

    Parameters
    ----------
    comb : GVMCombination-like
        Object exposing attributes: measurements, V_inv, Gamma, C_inv,
        uncertain_systematics, eoe_type.
    mu : float
        Parameter of interest.
    thetas : sequence of arrays
        Nuisance parameters grouped per systematic.
    """
    thetas = list(thetas)

    adj = np.sum([comb.Gamma[k] @ thetas[i]
                  for i, k in enumerate(comb.Gamma)], axis=0) if thetas else 0
    y_vals = np.fromiter(comb.measurements.values(), dtype=float)
    v = y_vals - mu - adj
    chi2_y = v @ comb.V_inv @ v

    chi2_u = 0.0
    keys = list(comb.Gamma.keys())
    for i, k in enumerate(keys):
        theta = np.asarray(thetas[i])
        eps = comb.uncertain_systematics[k]
        if comb.eoe_type.get(k, 'dependent') == 'dependent':
            Cinv = comb.C_inv[k]
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


def compute_FIM(comb, S):
    """Compute the Fisher Information Matrix for the combination.

    Parameters
    ----------
    comb : GVMCombination-like
        Object exposing attributes: V_inv, Gamma, C_inv, eoe_type.
    S : list of floats or arrays
        Scale factors per systematic used in Bartlett correction.
    """
    keys = list(comb.C_inv.keys())
    sizes = [comb.C_inv[k].shape[0] for k in keys]

    tot = sum(sizes)
    F = np.zeros((1 + tot, 1 + tot))
    F[0, 0] = np.sum(comb.V_inv)
    start_idx = np.cumsum([0] + sizes[:-1])
    idxs = [np.arange(sz) + s + 1 for sz, s in zip(sizes, start_idx)]

    V_G = {k: comb.V_inv @ comb.Gamma[k] for k in keys}
    for i, k in enumerate(keys):
        idx = idxs[i]
        F[0, idx] = F[idx, 0] = V_G[k].sum(axis=0)

    for i, ks in enumerate(keys):
        idx_s = idxs[i]
        Gs = comb.Gamma[ks]
        Cinv_s = comb.C_inv[ks]
        S_s = S[i]
        for j, kp in enumerate(keys):
            idx_p = idxs[j]
            GsVinGp = Gs.T @ V_G[kp]
            if ks == kp:
                if comb.eoe_type.get(kp, 'dependent') == 'dependent':
                    F[np.ix_(idx_s, idx_p)] = GsVinGp + (1.0 / S_s) * Cinv_s
                else:
                    F[np.ix_(idx_s, idx_p)] = (
                        GsVinGp + Cinv_s * (1.0 / S_s)[:, None]
                    )
            else:
                F[np.ix_(idx_s, idx_p)] = GsVinGp
    return F


def bartlett_correction(comb):
    """Return Bartlett corrections for profile LR and GOF.

    Returns
    -------
    tuple of float
        (b_profile, b_chi2)
    """
    if len(comb.C_inv) == 0:
        return 1.0, float(len(comb.measurements) - 1)

    thetas = comb.fit_results.thetas if isinstance(comb.fit_results, FitResult) else comb.fit_results['thetas']
    keys = list(comb.C_inv.keys())
    sizes = [comb.C_inv[k].shape[0] for k in keys]
    idx = np.cumsum([0] + sizes)
    thetas = [np.asarray(thetas[idx[i]:idx[i+1]]) for i in range(len(keys))]
    eps = [np.asarray(comb.uncertain_systematics[k], dtype=float) for k in keys]
    C_inv_list = [comb.C_inv[k] for k in keys]
    N_s = sizes
    S = []
    for th, k, C, e, N in zip(thetas, keys, C_inv_list, eps, N_s):
        if comb.eoe_type.get(k, 'dependent') == 'dependent':
            S.append((1 + 2 * e ** 2 * th @ C @ th) / (1 + 2 * e ** 2 * N))
        else:
            S.append((1 + 2 * e ** 2 * th ** 2) / (1 + 2 * e ** 2))
    F = compute_FIM(comb, S)
    W_full = np.linalg.inv(F)[1:, 1:]
    W_theta = np.linalg.inv(F[1:, 1:])
    start_idx = idx[:-1]
    b_lik = b_theta = b_chi2 = 0.0
    for s, (th, Cinv, e, N, S_s) in enumerate(zip(thetas, C_inv_list, eps, N_s, S)):
        si = start_idx[s]
        ei = si + N
        W_s = W_full[si:ei, si:ei]
        Wt_s = W_theta[si:ei, si:ei]
        if comb.eoe_type.get(keys[s], 'dependent') == 'dependent':
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
    b_chi2 = float(len(comb.measurements) - 1) + b_chi2 - b_lik
    return b_profile, b_chi2
