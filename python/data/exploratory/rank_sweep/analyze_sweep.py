"""Analyze the completed rank sweep: same-rank TCC comparisons.

Methodology (matches validate_matlab.py / plot_results.py conventions
already trusted in this project):
  - TCC = cosine similarity between L2-normalized factor columns.
  - Hungarian algorithm (linear_sum_assignment) finds the component
    permutation maximizing summed TCC — handles PARAFAC's column-order
    ambiguity.
  - Alignment is computed on modeB (time-of-day, the most identifying,
    least degenerate mode: 1440 points vs modeC's 2). The SAME alignment
    is then reused to score modeC (down/up split) — this asks "does the
    matched component also agree on its download/upload mix", not an
    independently-optimized modeC alignment (which would be close to
    meaningless with only 2 points per component).

Three pairings, at MATCHED rank (not a fixed reference):
  2019wk1 vs wan10   (does WAN look like 2019 at the same rank?)
  2019wk1 vs wan21   (same, second WAN week)
  wan10   vs wan21   (do two WAN weeks agree with each other?)

modeC comparison is only possible where BOTH sides of a pair have modeC
saved. 2019wk1 (all 5 ranks) and wan10_rank3 were fit before the
modeA/C-saving fix and are modeB-only — those pairs get modeB TCC only,
flagged in the CSV.

Outputs (this directory):
  tcc_summary.csv   — every matched-component TCC, modeB and modeC
  tcc_summary.png   — mean TCC vs rank, one line per pair, per mode
  loadings_rank{R}_{pairA}_vs_{pairB}.png — matched modeB curve overlays,
                      one figure per (pair, rank), for visual verification
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
RANKS = [3, 4, 5, 6, 7]
PAIRS = [("2019wk1", "wan10"), ("2019wk1", "wan21"), ("wan10", "wan21")]
COLORS = {"2019wk1_wan10": "tab:blue", "2019wk1_wan21": "tab:orange",
          "wan10_wan21": "tab:green"}
LABELS = {"2019wk1": "2019 wk1", "wan10": "WAN wk10", "wan21": "WAN wk21"}


def load_mode(name, rank, mode):
    path = f"{OUT}/mode{mode}_{name}_rank{rank}.npy"
    return np.load(path) if os.path.exists(path) else None


def tcc_matrix(a, b):
    a_n = a / np.linalg.norm(a, axis=0, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=0, keepdims=True)
    return a_n.T @ b_n


def best_alignment(mat):
    row_ind, col_ind = linear_sum_assignment(-mat)
    return row_ind, col_ind, mat[row_ind, col_ind]


rows = []  # for CSV
summary = {}  # (pair, rank) -> {"B": mean_tcc, "C": mean_tcc or None}

for name_a, name_b in PAIRS:
    pair_key = f"{name_a}_{name_b}"
    for rank in RANKS:
        B_a, B_b = load_mode(name_a, rank, "B"), load_mode(name_b, rank, "B")
        C_a, C_b = load_mode(name_a, rank, "C"), load_mode(name_b, rank, "C")
        if B_a is None or B_b is None:
            continue

        matB = tcc_matrix(B_a, B_b)
        r_ind, c_ind, tccB_vals = best_alignment(matB)

        tccC_vals = None
        if C_a is not None and C_b is not None:
            matC = tcc_matrix(C_a, C_b)
            # reuse modeB's alignment, don't re-optimize on modeC
            tccC_vals = matC[r_ind, c_ind]

        for i, (ra, rb) in enumerate(zip(r_ind, c_ind)):
            rows.append({
                "pair": pair_key, "rank": rank,
                "comp_a_idx": ra, "comp_b_idx": rb,
                "tcc_modeB": tccB_vals[i],
                "tcc_modeC": tccC_vals[i] if tccC_vals is not None else "",
            })

        summary[(pair_key, rank)] = {
            "B": tccB_vals.mean(),
            "C": tccC_vals.mean() if tccC_vals is not None else None,
        }

        # --- per-(pair, rank) modeB overlay plot, Hungarian-matched, then
        # amplitude-rescaled (same convention as validate_matlab.py / fig 7
        # of the report: PARAFAC's scale is arbitrarily split across modes,
        # so two independently-fit models can have wildly different raw
        # modeB magnitudes even with identical shape. Shape-matching (TCC)
        # is scale-invariant and already correct; for the PLOT, name_b's
        # curve is rescaled by one scalar per component (ratio of means)
        # so amplitude is visually comparable — never per-curve min-max,
        # which would distort each curve's own relative dynamics).
        n = len(r_ind)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3), squeeze=False)
        for i, (ra, rb) in enumerate(zip(r_ind, c_ind)):
            ax = axes[0, i]
            curve_a = B_a[:, ra]
            curve_b_raw = B_b[:, rb]
            scale = curve_a.mean() / curve_b_raw.mean()
            curve_b = curve_b_raw * scale
            ax.plot(curve_a, label=name_a, color="tab:blue", lw=1.2)
            ax.plot(curve_b, label=f"{name_b} (×{scale:.2f})",
                    color="tab:red", lw=1.2, alpha=0.8)
            ax.set_title(f"c{ra}↔c{rb}  TCC={tccB_vals[i]:.3f}",
                         fontsize=9)
            ax.set_xticks([0, 360, 720, 1080, 1439])
            ax.set_xticklabels(["00h", "06h", "12h", "18h", "24h"],
                               fontsize=7)
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f"{name_a} vs {name_b}  —  rank {rank}  "
                    f"(shape-matched, amplitude-rescaled per component "
                    f"— not min-max, mean TCC={tccB_vals.mean():.3f})",
                    fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(f"{OUT}/loadings_rank{rank}_{pair_key}.png", dpi=110)
        plt.close(fig)

# --- CSV ---
with open(f"{OUT}/tcc_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pair", "rank", "comp_a_idx",
                                      "comp_b_idx", "tcc_modeB", "tcc_modeC"])
    w.writeheader()
    w.writerows(rows)

# --- summary plot: mean TCC vs rank ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for name_a, name_b in PAIRS:
    pair_key = f"{name_a}_{name_b}"
    pair_label = f"{LABELS[name_a]} vs {LABELS[name_b]}"
    color = COLORS[pair_key]
    ranks_b = [r for r in RANKS if (pair_key, r) in summary]
    means_b = [summary[(pair_key, r)]["B"] for r in ranks_b]
    ax1.plot(ranks_b, means_b, "o-", color=color, label=pair_label)

    ranks_c = [r for r in RANKS
              if (pair_key, r) in summary and summary[(pair_key, r)]["C"] is not None]
    means_c = [summary[(pair_key, r)]["C"] for r in ranks_c]
    if ranks_c:
        ax2.plot(ranks_c, means_c, "o-", color=color, label=pair_label)

ax1.set_title("modeB (time-of-day) — mean matched TCC vs rank")
ax1.set_xlabel("rank")
ax1.set_ylabel("mean TCC")
ax1.set_ylim(0, 1)
ax1.axhline(0.95, ls="--", color="gray", lw=0.8)
ax1.axhline(0.85, ls=":", color="gray", lw=0.8)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

ax2.set_title("modeC (down/up split) — mean matched TCC vs rank\n"
             "(only where both sides have modeC saved)")
ax2.set_xlabel("rank")
ax2.set_ylabel("mean TCC")
ax2.set_ylim(0, 1)
ax2.axhline(0.95, ls="--", color="gray", lw=0.8)
ax2.axhline(0.85, ls=":", color="gray", lw=0.8)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"{OUT}/tcc_summary.png", dpi=130)
plt.close(fig)

print("Wrote tcc_summary.csv, tcc_summary.png, and per-rank loading plots.")
print()
print(f"{'pair':20s} {'rank':5s} {'meanTCC_B':10s} {'meanTCC_C':10s}")
for (pair_key, rank), v in sorted(summary.items()):
    c_str = f"{v['C']:.4f}" if v["C"] is not None else "  n/a"
    print(f"{pair_key:20s} {rank:<5d} {v['B']:<10.4f} {c_str}")
