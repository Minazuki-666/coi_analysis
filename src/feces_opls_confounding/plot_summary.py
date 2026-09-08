# %%
"""Figures for the two result tables written by confounding.py.

  confounding_loadings.csv -> the per-bin loadings back on the chemical shift axis
  confounding_summary.csv  -> the scalar verdict of the confounding assessment

Both read the CSVs rather than refitting, so the figures always match the tables.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")
sns.set_context("talk", font_scale=0.9)

C_UNADJ = "#4d4d4d"
C_MATCH = "#2c7fb8"
# %%
heredir = os.path.dirname(os.path.abspath(__file__))
figdir = os.path.join(heredir, "fig")
os.makedirs(figdir, exist_ok=True)

load = pd.read_csv(os.path.join(heredir, "confounding_loadings.csv"))
summ = pd.read_csv(os.path.join(heredir, "confounding_summary.csv"),
                   header=None, index_col=0).squeeze("columns")
print(f"loadings: {len(load)} bins, {load['ppm'].max():.2f}–{load['ppm'].min():.2f} ppm")
print(summ.to_string())
# %%
def break_gaps(ppm, *series):
  """Insert NaN across the removed water region so lines are not drawn through it."""
  step = np.median(np.abs(np.diff(ppm)))
  cut = np.flatnonzero(np.abs(np.diff(ppm)) > 1.5 * step) + 1
  return [np.insert(s.astype(float), cut, np.nan) for s in (ppm, *series)]


d = load.sort_values("ppm", ascending=False)
x, unadj, mean, lo, hi, width = break_gaps(
  d["ppm"].to_numpy(), d["p_corr_unadjusted"], d["p_corr_matched_mean"],
  d["p_corr_matched_lo95"], d["p_corr_matched_hi95"], d["width95"])

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})

ax = axes[0]
ax.fill_between(x, lo, hi, color=C_MATCH, alpha=0.25, linewidth=0,
                label=f"matched: central 95% of {int(summ['n_draws'])} draws")
ax.plot(x, mean, color=C_MATCH, linewidth=1.6, label="matched: mean")
ax.plot(x, unadj, color=C_UNADJ, linewidth=1.2, linestyle="--",
        label=f"unadjusted (n={int(summ['n_joined'])})")
ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
ax.set_ylabel("p(corr)[1]")
ax.legend(frameon=False, fontsize=11, ncol=3, loc="lower left")
ax.set_title("Feeding-type adjustment across the spectrum — "
             f"Newborn (timing {int(summ['timing'])}, 1+{int(summ['n_ortho'])})",
             fontsize=13)

ax = axes[1]
# the adjustment only matters where it moves a loading further than the draws do
half = width / 2
ax.fill_between(x, -half, half, color="#bdbdbd", alpha=0.45, linewidth=0,
                label="±½ of the 95% draw interval")
ax.plot(x, mean - unadj, color="#c0392b", linewidth=1.4,
        label="matched mean − unadjusted")
ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
outside = np.nansum(np.abs(mean - unadj) > half)
ax.set_xlim(np.nanmax(x), np.nanmin(x))
ax.set_xlabel("chemical shift [ppm]")
ax.set_ylabel("difference")
ax.legend(frameon=False, fontsize=11, ncol=2, loc="upper left")
ax.set_title(f"{int(outside)} / {len(load)} bins move further than the draw noise",
             fontsize=12)

fig.tight_layout()
fig.savefig(os.path.join(figdir, "loadings_spectrum.svg"))
plt.show()
# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

# --- the ladder: one change at a time ---------------------------------------
ax = axes[0]
rungs = [("full", "Q2_full", summ["n_full"], "unadjusted, every sample"),
         ("joined", "Q2_joined", summ["n_joined"], "unadjusted, feeding data only"),
         ("matched", "Q2_matched", summ["n_matched"], "feeding-type matched"),
         ("random", "Q2_random", summ["n_matched"], "same n, feeding ignored")]
colors = [C_UNADJ, C_UNADJ, C_MATCH, "#7f7f7f"]
ypos = np.arange(len(rungs))[::-1]
ax.barh(ypos, [summ[k] for _, k, _, _ in rungs], height=0.55, color=colors, alpha=0.85)
for yp, (name, k, n, _) in zip(ypos, rungs):
  ax.text(summ[k] + 0.008, yp, f"{summ[k]:.3f}", va="center", fontsize=11)
ax.set_yticks(ypos, [f"{name}\n(n={int(n)})" for name, _, n, _ in rungs], fontsize=11)
ax.set_xlim(0, max(summ[k] for _, k, _, _ in rungs) * 1.25)
ax.set_xlabel("Q2")
ax.set_title("One change at a time\n(single draw, seed=0)", fontsize=12)

# --- effect of matching, relative to the draw-to-draw spread ----------------
ax = axes[1]
metrics = [("Q2", "Q2_cohen_d"), ("R2Y", "R2Y_cohen_d"),
           ("r of p(corr)[1]", "r_pcorr_cohen_d"),
           ("top-10 overlap", "top_overlap_cohen_d")]
ypos = np.arange(len(metrics))[::-1]
vals = [summ[k] for _, k in metrics]
ax.axvspan(-0.2, 0.2, color="#2ca02c", alpha=0.12, label="|d| < 0.2 (negligible)")
ax.axvline(0, color="gray", linewidth=1)
for v, yp in zip(vals, ypos):
  ax.plot([0, v], [yp, yp], color="#999999", linewidth=1.2, zorder=1)
ax.scatter(vals, ypos, s=110, color=C_MATCH, edgecolor="black",
           linewidth=0.6, zorder=3)
for v, yp in zip(vals, ypos):
  ax.text(v + 0.035, yp, f"{v:+.2f}", ha="left", va="center", fontsize=11)
ax.set_yticks(ypos, [m for m, _ in metrics], fontsize=11)
ax.set_xlim(-0.35, max(vals) * 1.55)
ax.set_ylim(-0.75, len(metrics) - 0.5)
ax.set_xlabel("Cohen's d  (matched − random)")
ax.legend(frameon=False, fontsize=10, loc="lower left")
ax.set_title(f"Effect of matching over {int(summ['n_draws'])} draws\n"
             "positive = matching helps slightly", fontsize=12)

# --- is feeding type a confounder at all? -----------------------------------
ax = axes[2]
tests = [(f"feeding type × sec-type\n$\\chi^2$ = {summ['chi2_feeding_vs_sec']:.2f}",
          summ["p_chi2"]),
         (f"$t_{{pred}}$ by feeding type\nKruskal H = "
          f"{summ['kruskal_tpred_by_feeding']:.2f}", summ["p_kruskal"])]
ypos = np.arange(len(tests))[::-1]
ax.barh(ypos, [p for _, p in tests], height=0.45, color="#2ca02c", alpha=0.6)
ax.axvline(0.05, color="#c0392b", linestyle="--", linewidth=1.4, label="p = 0.05")
for (_, p), yp in zip(tests, ypos):
  ax.text(p + 0.02, yp, f"p = {p:.3f}", va="center", fontsize=11)
ax.set_yticks(ypos, [t for t, _ in tests], fontsize=11)
ax.set_xlim(0, 1)
ax.set_xlabel("p value")
ax.legend(frameon=False, fontsize=10, loc="lower right")
ax.set_title("A confounder must be associated\nwith both — neither holds",
             fontsize=12)

fig.suptitle("Is infant feeding type confounding the sec / non-sec model?",
             fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "confounding_summary.svg"))
plt.show()
# %%
print(f"\nbins moving further than the draw noise: {int(outside)} / {len(load)}")
print("widest 95% draw intervals (most draw-sensitive bins):")
print(load.nlargest(8, "width95")[
  ["ppm", "p_corr_unadjusted", "p_corr_matched_mean", "width95"]]
  .round(3).to_string(index=False))
