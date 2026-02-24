"""Lambda scan over error-on-error scaling.

Scans a global scaling factor lambda that multiplies all error-on-error
values uniformly, tracking how mu_hat, CI, and GOF evolve.

Usage:
    cd runs/toy_2meas && python ../scan.py
    python runs/scan.py --config runs/toy_2meas/input/toy2.yaml
    python runs/scan.py --config runs/toy_4meas/input/toy4.yaml --lambda-range 0 2.0
"""
import argparse
import glob as globmod
import sys
from pathlib import Path
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gvm import GVMCombination, build_input_data


def find_default_config():
    """Auto-discover the YAML config in input/. Error if not exactly one."""
    yamls = sorted(globmod.glob("input/*.yaml")) + sorted(globmod.glob("input/*.yml"))
    if len(yamls) == 0:
        raise FileNotFoundError("No .yaml files found in input/.")
    if len(yamls) > 1:
        raise RuntimeError(
            f"Multiple .yaml files found in input/: {yamls}. "
            "Use --config to specify which one."
        )
    return yamls[0]


def parse_args():
    p = argparse.ArgumentParser(description="Lambda scan over error-on-error.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to the YAML config (default: auto-discover in input/).")
    p.add_argument("--output", type=str, default="output",
                   help="Output directory (default: output/).")
    p.add_argument("--lambda-range", type=float, nargs=2, default=[0.0, 1.2],
                   metavar=("MIN", "MAX"),
                   help="Lambda scan range (default: 0.0 1.2).")
    p.add_argument("--n-points", type=int, default=14,
                   help="Number of scan points (default: 14).")
    return p.parse_args()


def main():
    args = parse_args()
    config = Path(args.config).resolve() if args.config else Path(find_default_config()).resolve()
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=True)

    lam_min, lam_max = args.lambda_range
    lambda_grid = np.linspace(lam_min, lam_max, args.n_points)

    data = build_input_data(str(config))
    comb = GVMCombination(data)
    base_info = comb.get_input_data(copy=True)

    # Read base epsilons from the YAML config (values at lambda=1)
    base_eps = {}
    for sname in base_info.syst:
        if sname in base_info.uncertain_systematics:
            base_eps[sname] = base_info.uncertain_systematics[sname]
        else:
            base_eps[sname] = 0.0

    cv, lo_1, hi_1, lo_2, hi_2, significances = [], [], [], [], [], []

    for lam in lambda_grid:
        info = deepcopy(base_info)
        for sname, eps0 in base_eps.items():
            scaled = lam * eps0
            info.uncertain_systematics[sname] = scaled
        comb.set_input_data(info)
        cv.append(comb.fit_results.mu)
        l1, u1, _ = comb.confidence_interval(cl_val=0.683)
        l2, u2, _ = comb.confidence_interval(cl_val=0.955)
        lo_1.append(l1); hi_1.append(u1)
        lo_2.append(l2); hi_2.append(u2)

        chi2_val = comb.goodness_of_fit()
        p = 1 - chi2.cdf(chi2_val, df=data.n_meas - 1)
        significances.append(norm.ppf(1 - p / 2))

    # ── scan plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lambda_grid, cv, "k--o", label="Central Value")
    ax.fill_between(lambda_grid, lo_2, hi_2, color="yellow", alpha=0.25, label="95.5% CI")
    ax.fill_between(lambda_grid, lo_1, hi_1, color="green", alpha=0.5, label="68.3% CI")
    ax.set_xlabel(r"$\lambda$", fontsize=24)
    ax.set_ylabel(r"$\mu$", fontsize=22)
    ax.set_xlim(lam_min, lam_max)
    all_lo = min(min(lo_2), min(cv))
    all_hi = max(max(hi_2), max(cv))
    y_span = all_hi - all_lo
    ax.set_ylim(all_lo - 0.15 * y_span, all_hi + 0.15 * y_span)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output / "scan.png", dpi=150)
    plt.close(fig)
    print(f"Scan plot saved to {output / 'scan.png'}")

    # ── GOF plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(lambda_grid, significances, "--o")
    ax.set_xlabel(r"$\lambda$", fontsize=24)
    ax.set_ylabel("Significance", fontsize=20)
    ax.set_xlim(lam_min, lam_max)
    ax.set_ylim(bottom=0, top=max(significances) * 1.15)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output / "gof.png", dpi=150)
    plt.close(fig)
    print(f"GOF plot saved to {output / 'gof.png'}")


if __name__ == "__main__":
    main()
