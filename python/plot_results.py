"""
Plot PARAFAC results and Python vs MATLAB validation.

Generates two figures saved to python/figures/:
  1. modeb_profiles.png  — time-of-day loading profiles for each component
                           Python (tol=1e-8) overlaid on MATLAB reference
  2. tcc_validation.png  — TCC heatmap: per-week and reference model

Usage:
  python plot_results.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
LOADINGS_CSV = os.path.join(_HERE, "..", "matlab", "data", "loadings_all.csv")
ALL_DAYS_DIR = os.path.join(_HERE, "data", "output_tol1e8", "all_days")
PER_WEEK_DIR = os.path.join(_HERE, "data", "output", "per_week")
FIGURES_DIR = os.path.join(_HERE, "figures")

TRAINING_SOURCE = "2019-08-19_2019-09-22"
N_COMP = 5
COMP_COLS = [f"comp_{i+1}" for i in range(N_COMP)]
WEEK_MAP = {
    "1": "week01", "2": "week02", "3": "week03",
    "4": "week04", "5": "week05",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tcc_matrix(a, b):
    a_n = a / np.linalg.norm(a, axis=0, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=0, keepdims=True)
    return a_n.T @ b_n


def best_alignment(mat):
    row_ind, col_ind = linear_sum_assignment(-mat)
    return row_ind, col_ind, mat[row_ind, col_ind]


def load_matlab(df, type_, week, mode):
    sub = df[
        (df["source"] == TRAINING_SOURCE)
        & (df["type"] == type_)
        & (df["week"] == week)
        & (df["mode"] == mode)
    ].sort_values("row")
    return sub[COMP_COLS].values.astype(float)


def load_python(path):
    return pd.read_csv(path).values.astype(float)


def minutes_to_time(minutes):
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


# ---------------------------------------------------------------------------
# Figure 1: modeB time-of-day profiles (all_days, Python vs MATLAB)
# ---------------------------------------------------------------------------

def plot_modeb_profiles(df):
    mat_b = load_matlab(df, "reference", "reference", "B")
    py_b = load_python(os.path.join(ALL_DAYS_DIR, "modeB_all_days.csv"))

    tcc = tcc_matrix(mat_b, py_b)
    row_ind, col_ind, tcc_vals = best_alignment(tcc)
    py_b_aligned = py_b[:, col_ind]

    minutes = np.arange(1440)
    x_ticks = np.arange(0, 1441, 120)
    x_labels = [minutes_to_time(m) for m in x_ticks]

    fig, axes = plt.subplots(N_COMP, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "modeB — Time-of-day profiles (all_days model)\n"
        "Python tol=1e-8 vs MATLAB reference",
        fontsize=12,
    )

    def norm01(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else v

    colors = plt.cm.tab10.colors
    for i, ax in enumerate(axes):
        py_col = py_b_aligned[:, i]
        mat_col = mat_b[:, row_ind[i]]

        ax.plot(minutes, norm01(mat_col), color=colors[i], lw=1.2,
                label=f"MATLAB comp {row_ind[i]+1}")
        ax.plot(minutes, norm01(py_col), color=colors[i], lw=1.2,
                linestyle="--", alpha=0.75,
                label=f"Python comp {col_ind[i]+1}")

        ax.set_ylabel(f"Comp {i+1}\nTCC={tcc_vals[i]:.3f}", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylim(-0.05, 1.15)
        ax.yaxis.set_major_locator(mticker.NullLocator())

    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("Time of day")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "modeb_profiles.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: TCC validation heatmap
# ---------------------------------------------------------------------------

def plot_tcc_heatmap(df):
    row_labels = [f"Week {w}" for w in range(1, 6)] + ["Reference"]
    tcc_mean_b = []
    tcc_vals_b = []

    for week_num, py_label in WEEK_MAP.items():
        mat_b = load_matlab(df, "weekly", week_num, "B")
        py_b = load_python(
            os.path.join(PER_WEEK_DIR, py_label, f"modeB_{py_label}.csv")
        )
        _, _, vals = best_alignment(tcc_matrix(mat_b, py_b))
        tcc_vals_b.append(sorted(vals, reverse=True))
        tcc_mean_b.append(vals.mean())

    mat_b = load_matlab(df, "reference", "reference", "B")
    py_b = load_python(os.path.join(ALL_DAYS_DIR, "modeB_all_days.csv"))
    _, _, vals = best_alignment(tcc_matrix(mat_b, py_b))
    tcc_vals_b.append(sorted(vals, reverse=True))
    tcc_mean_b.append(vals.mean())

    data = np.array(tcc_vals_b)  # (6, 5)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, vmin=0.98, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="TCC")

    ax.set_xticks(range(N_COMP))
    ax.set_xticklabels([f"Comp {i+1}" for i in range(N_COMP)])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(
        "TCC: Python vs MATLAB — modeB (time profiles)\n"
        "Columns sorted by TCC descending per model"
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black")

    for i, mean in enumerate(tcc_mean_b):
        ax.text(N_COMP + 0.6, i, f"μ={mean:.4f}", va="center", fontsize=8)

    ax.set_xlim(-0.5, N_COMP + 1.2)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "tcc_validation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Loading {LOADINGS_CSV} …")
    df = pd.read_csv(LOADINGS_CSV, low_memory=False)
    df["week"] = df["week"].astype(str)

    print("Plotting modeB time profiles …")
    plot_modeb_profiles(df)

    print("Plotting TCC validation heatmap …")
    plot_tcc_heatmap(df)

    print("Done.")


if __name__ == "__main__":
    main()
