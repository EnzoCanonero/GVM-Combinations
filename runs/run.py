"""Run a GVM combination and produce summary results.

Usage:
    cd runs/toy_2meas && python ../run.py
    python runs/run.py --config runs/toy_2meas/input/toy2.yaml
"""
import argparse
import glob as globmod
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
import yaml

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
    p = argparse.ArgumentParser(description="Run a GVM combination.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to the YAML config (default: auto-discover in input/).")
    p.add_argument("--output", type=str, default="output",
                   help="Output directory (default: output/).")
    return p.parse_args()


def main():
    args = parse_args()
    config = Path(args.config).resolve() if args.config else Path(find_default_config()).resolve()
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=True)

    data = build_input_data(str(config))
    comb = GVMCombination(data)
    comb.fit()

    mu = comb.fit_results.mu
    lo_1, hi_1, hw_1 = comb.confidence_interval(cl_val=0.683)
    lo_2, hi_2, hw_2 = comb.confidence_interval(cl_val=0.955)
    gof = comb.goodness_of_fit()
    p_val = 1 - chi2.cdf(gof, df=data.n_meas - 1)
    sig = norm.ppf(1 - p_val / 2)

    # ── terminal output ──────────────────────────────────────────────
    print(f"\n=== GVM Combination: {data.name} ===")
    print(f"mu_hat      = {mu:.4f}")
    print(f"68.3% CI    = ({lo_1:.4f}, {hi_1:.4f}),  half-width = {hw_1:.4f}")
    print(f"95.5% CI    = ({lo_2:.4f}, {hi_2:.4f}),  half-width = {hw_2:.4f}")
    print(f"GOF chi2    = {gof:.3f}")
    print(f"p-value     = {p_val:.4f}")
    print(f"significance = {sig:.2f} sigma")

    # ── results file ─────────────────────────────────────────────────
    results = {
        "name": data.name,
        "mu_hat": float(mu),
        "confidence_interval_68": {
            "lower": float(lo_1), "upper": float(hi_1),
            "half_width": float(hw_1),
        },
        "confidence_interval_95": {
            "lower": float(lo_2), "upper": float(hi_2),
            "half_width": float(hw_2),
        },
        "goodness_of_fit": {
            "chi2": float(gof),
            "p_value": float(p_val),
            "significance": float(sig),
        },
    }
    with open(output / "results.yaml", "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    # ── summary plot ─────────────────────────────────────────────────
    labels = data.labels
    centrals = np.array([data.measurements[l] for l in labels])
    stat_diag = np.sqrt(np.diag(data.V_stat))

    syst_sq = np.zeros(data.n_meas)
    for src, sigma in data.syst.items():
        syst_sq += sigma ** 2
    total_err = np.sqrt(stat_diag ** 2 + syst_sq)

    x_pos = np.arange(data.n_meas)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhspan(lo_2, hi_2, color="yellow", alpha=0.25, label="95.5% CI")
    ax.axhspan(lo_1, hi_1, color="green", alpha=0.5, label="68.3% CI")
    ax.axhline(mu, color="red", linewidth=1.2, label=r"MLE for $\mu$")
    ax.errorbar(x_pos, centrals, yerr=total_err, fmt="o", color="blue",
                capsize=5, markersize=7, label="Data Points")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("Value", fontsize=16)
    ax.legend(fontsize=12, loc="upper right")
    all_lo = [lo_2, min(centrals - total_err)]
    all_hi = [hi_2, max(centrals + total_err)]
    y_span = max(all_hi) - min(all_lo)
    ax.set_ylim(min(all_lo) - 0.15 * y_span, max(all_hi) + 0.15 * y_span)
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {output / 'summary.png'}")


if __name__ == "__main__":
    main()
