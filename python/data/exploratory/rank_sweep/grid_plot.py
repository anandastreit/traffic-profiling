"""Grid of all 15 fits' own modeB components — one panel per (dataset, rank).

First attempt at this (per-component, matched to one of the 5 anchor
components individually) was tried and rejected: with ranks 3-7 all being
matched against only 5 anchor slots, most components across most fits
landed on the SAME best-matching anchor slot (checked directly: anchor
component 1 alone was the top match for ~40% of all components tested),
collapsing the whole grid to one dominant colour and hiding real
structure instead of revealing it — worse than the original, not better.

This version applies ONE scalar per fit instead of one per component:
each fit's modeB is rescaled by (mean of the whole anchor model, 2019wk1
rank5) / (mean of this fit's whole modeB) — a single number per panel,
not per curve. This is coarser than the true per-component fix (it can't
correct PARAFAC's per-component scale indeterminacy within a fit, only
each fit's overall magnitude relative to the anchor), but it removes the
one real, correctable source of cross-panel incomparability (arbitrary
overall magnitude differences between independently-run fits) without
introducing a matching artefact. Colour goes back to component slot
(arbitrary within a fit, NOT matched across panels — unlike the
Hungarian-matched pairwise comparison figures elsewhere in this report,
which remain the rigorous per-component comparison tool). All 15 panels
share one y-axis so height is now meaningfully comparable panel-to-panel.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["2019wk1", "wan10", "wan21"]
RANKS = [3, 4, 5, 6, 7]
ROW_LABELS = {"2019wk1": "2019 wk1", "wan10": "WAN wk10", "wan21": "WAN wk21"}
CMAP = plt.get_cmap("tab10")

ANCHOR_NAME, ANCHOR_RANK = "2019wk1", 5
anchor = np.load(f"{OUT}/modeB_{ANCHOR_NAME}_rank{ANCHOR_RANK}.npy")
anchor_mean = anchor.mean()

# Pass 1: one scalar per fit (mean of the whole anchor / mean of this fit's
# whole modeB), applied uniformly to every component in that fit — then the
# global y-max so all panels can share one axis.
panel_data = {}
global_max = 0.0
for name in DATASETS:
    for rank in RANKS:
        B = np.load(f"{OUT}/modeB_{name}_rank{rank}.npy")
        scale = anchor_mean / B.mean()
        rescaled = B * scale
        panel_data[(name, rank)] = (rescaled, scale)
        global_max = max(global_max, rescaled.max())

fig, axes = plt.subplots(len(DATASETS), len(RANKS),
                         figsize=(3.0 * len(RANKS), 2.3 * len(DATASETS)),
                         squeeze=False, sharey=True)

for i, name in enumerate(DATASETS):
    for j, rank in enumerate(RANKS):
        ax = axes[i, j]
        rescaled, scale = panel_data[(name, rank)]
        for r in range(rescaled.shape[1]):
            ax.plot(rescaled[:, r], color=CMAP(r), lw=1.0)
        if i == 0:
            ax.set_title(f"rank {rank}", fontsize=10)
        if j == 0:
            ax.set_ylabel(ROW_LABELS[name], fontsize=10)
        ax.text(0.02, 0.95, f"×{scale:.2f}", transform=ax.transAxes,
                fontsize=7, color="gray", va="top")
        ax.set_xticks([0, 720, 1439])
        ax.set_xticklabels(["00h", "12h", "24h"], fontsize=6)
        ax.set_ylim(0, global_max * 1.05)

fig.suptitle("Every fit's own modeB components, each fit rescaled by ONE factor "
            "(its overall mean vs the 2019wk1 rank-5 anchor's) — one shared "
            "y-axis so height is comparable across panels; colour = component "
            "slot within that fit only, not matched across panels", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}/grid_own_models.png", dpi=120)
plt.close(fig)
print("wrote grid_own_models.png, global_max=", global_max)
