# %%
"""Distance-based redundancy analysis (Legendre & Anderson 1999) of the genus
profiles, constrained by sec-type, one model per timing.

db-RDA is the constrained counterpart of the unconstrained PCoA in PCoA.py:
the same Bray-Curtis Gower matrix is split into the part explained by sec-type
and the residual, so the first axis is the direction of maximum sec/non-sec
separation rather than of maximum total variation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

sns.set_theme(style="white")
sns.set_context("talk", font_scale=0.9)

N_PERM = 9999
SEED = 0
PALETTE = {"sec": "#d62728", "non-sec": "#1f77b4"}
GROUPS = ["sec", "non-sec"]   # fixed order so the legend matches across timings
# spread of each group's own scatter, in standard deviations along the
# ellipse axes (2 SD covers ~86% of a bivariate normal)
ELLIPSE_N_SD = 2.0
N_ARROWS = 6   # taxa drawn on the biplot, ranked by r2 against the two axes

# how each timing is labelled in the figures, and the slug used in file names
TIMING_LABELS = {
  3: ("Newborn", "newborn"),
  4: ("1 mo.", "1mo"),
  5: ("4-5 mos.", "4-5mos"),
}
# %%
heredir = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.dirname(os.path.dirname(heredir))
datadir = os.path.join(basedir, "data")
figdir = os.path.join(heredir, "fig")
os.makedirs(figdir, exist_ok=True)
# %%
df = pd.read_csv(os.path.join(datadir, "20260901_Genus_RelativeAb.csv"), header=0)
# the file ends with a blank line and an unlabelled column-summary row
df = df.dropna(subset=["ID"]).reset_index(drop=True)
df["timing"] = df["timing"].astype(int)

taxa = df.columns[6:]
X_all = df.iloc[:, 6:].to_numpy(dtype=float)
# taxa absent from every sample carry no Bray-Curtis information
present = X_all.sum(axis=0) > 0
taxa = taxa[present]
X_all = X_all[:, present]
labels_all = df["sec-type"].to_numpy()
timing_all = df["timing"].to_numpy()


def short_name(lineage):
  """Last named rank of a '|'-separated GTDB lineage.

  Ranks above genus are flagged - g__Bifidobacterium becomes Bifidobacterium,
  but f__Enterobacteriaceae_A becomes "Enterobacteriaceae_A (f)" so a bin that
  was never resolved to a genus is not read as one.
  """
  for field in reversed(lineage.split("|")):
    rank, _, name = field.partition("__")
    if name and name != "_":
      # trailing numeric suffixes (g__Bacteroides_H_123456) only add noise
      parts = name.split("_")
      while len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
      name = "_".join(parts)
      return name if rank == "g" else f"{name} ({rank})"
  # lineages with no named rank at all, e.g. "Unassigned|__|__|..."
  return lineage.split("|")[0]
# %%
def gower(D):
  """Gower's centred matrix G = -0.5 J D^2 J, the response of a db-RDA."""
  n = D.shape[0]
  A = -0.5 * D ** 2
  J = np.eye(n) - np.ones((n, n)) / n
  G = J @ A @ J
  return (G + G.T) / 2


def dbrda(D, labels):
  """db-RDA of a distance matrix on one categorical constraint.

  Follows vegan::dbrda: the Gower matrix is used directly, without a
  correction for the negative eigenvalues that Bray-Curtis produces, so the
  total inertia is the trace of G and the constrained fraction is exact.

  Returns the constrained axis, the first residual (unconstrained) axis, the
  variation each carries, and the permutation test of the constraint.
  """
  n = D.shape[0]
  G = gower(D)
  groups = np.unique(labels)
  # dummy coding of the factor: k levels give k-1 columns, centred so that the
  # hat matrix is orthogonal to the intercept already removed by the centring
  Z = np.column_stack([(labels == g).astype(float) for g in groups[1:]])
  Z = Z - Z.mean(axis=0)
  m = Z.shape[1]
  H = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
  I = np.eye(n)

  G_fit = H @ G @ H
  G_res = (I - H) @ G @ (I - H)
  total, constrained = np.trace(G), np.trace(G_fit)
  residual = total - constrained

  def eig_axes(M):
    val, vec = np.linalg.eigh((M + M.T) / 2)
    order = np.argsort(val)[::-1]
    return val[order], vec[:, order]

  val_fit, vec_fit = eig_axes(G_fit)
  val_res, vec_res = eig_axes(G_res)
  # weighted-average site scores (vegan's default display): the raw data
  # projected onto the axis, WA = G U / sqrt(lambda)
  wa = (G @ vec_fit[:, :m]) / np.sqrt(val_fit[:m])
  mds = vec_res[:, :2] * np.sqrt(val_res[:2])

  f_obs = (constrained / m) / (residual / (n - m - 1))
  rng = np.random.default_rng(SEED)
  perm = np.empty(N_PERM)
  for i in range(N_PERM):
    idx = rng.permutation(n)
    Gp = G[np.ix_(idx, idx)]
    c = np.trace(H @ Gp @ H)
    perm[i] = (c / m) / ((total - c) / (n - m - 1))
  p = (np.sum(perm >= f_obs) + 1) / (N_PERM + 1)

  r2 = constrained / total
  return {
    "dbRDA1": wa[:, 0], "MDS1": mds[:, 0], "MDS2": mds[:, 1],
    "constrained_var": val_fit[:m] / total, "mds_var": val_res[:2] / total,
    "R2": r2, "R2_adj": 1 - (1 - r2) * (n - 1) / (n - m - 1),
    "F": f_obs, "p": p, "df1": m, "df2": n - m - 1,
    "centroids": {g: wa[labels == g, 0].mean() for g in groups},
  }


def taxa_fit(X, axes, names, n_top=N_ARROWS):
  """Correlation of each taxon with the two plotted axes (an envfit biplot).

  Abundances are square-root transformed first, which stabilises the variance
  of proportions; the arrow is the correlation vector and its length sqrt(r2).
  """
  Xs = np.sqrt(X)
  Xs = Xs - Xs.mean(axis=0)
  sd = Xs.std(axis=0)
  keep = sd > 0
  A = axes - axes.mean(axis=0)
  corr = (Xs[:, keep].T @ A) / (len(A) * sd[keep][:, None] * A.std(axis=0))
  r2 = (corr ** 2).sum(axis=1)
  order = np.argsort(r2)[::-1][:n_top]
  idx = np.flatnonzero(keep)[order]
  return pd.DataFrame({
    "taxon": [short_name(names[i]) for i in idx],
    "lineage": [names[i] for i in idx],
    "corr_dbRDA1": corr[order, 0], "corr_axis2": corr[order, 1],
    "r2": r2[order],
  })


def spread_labels(fig, labels, max_pass=60):
  """Push overlapping arrow labels apart vertically, a few points at a time.

  The biplot arrows fan out from one origin, so several labels can land on the
  same spot; nudging their offsets is easier to read than dropping taxa.
  """
  to_points = 72 / fig.dpi
  for _ in range(max_pass):
    fig.canvas.draw()
    boxes = [t.get_window_extent() for t in labels]
    moved = False
    for i in range(len(labels)):
      for j in range(i + 1, len(labels)):
        a, b = boxes[i], boxes[j]
        if a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0:
          continue
        shift = (min(a.y1, b.y1) - max(a.y0, b.y0) + 2) / 2 * to_points
        up, down = (i, j) if a.y0 >= b.y0 else (j, i)
        for k, sign in ((up, 1), (down, -1)):
          x, y = labels[k].xyann
          labels[k].xyann = (x, y + sign * shift)
        moved = True
    if not moved:
      return


def data_ellipse(ax, x, y, color, n_sd=ELLIPSE_N_SD):
  """Ellipse covering the group's own scatter (vegan's ordiellipse kind="sd")."""
  if len(x) < 3:
    return
  cov = np.cov(x, y)
  eigval, eigvec = np.linalg.eigh(cov)
  order = np.argsort(eigval)[::-1]
  eigval, eigvec = eigval[order], eigvec[:, order]
  angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
  width, height = 2 * n_sd * np.sqrt(eigval)
  ax.add_patch(Ellipse((x.mean(), y.mean()), width, height, angle=angle,
                       facecolor=color, alpha=0.12, edgecolor=color,
                       linestyle="--", linewidth=1.4, zorder=1))
# %%
def run_timing(timing):
  label, slug = TIMING_LABELS[timing]
  mask = timing_all == timing
  X, y = X_all[mask], labels_all[mask]

  D = squareform(pdist(X, metric="braycurtis"))
  res = dbrda(D, y)
  axes = np.column_stack([res["dbRDA1"], res["MDS1"]])
  fit = taxa_fit(X, axes, taxa.to_numpy())

  fig, ax = plt.subplots(figsize=(7.6, 6.6))
  ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
  for cls in GROUPS:
    sel = y == cls
    data_ellipse(ax, axes[sel, 0], axes[sel, 1], PALETTE[cls])
    ax.scatter(axes[sel, 0], axes[sel, 1], s=55, color=PALETTE[cls],
               edgecolor="white", linewidth=0.5, alpha=0.75, zorder=3,
               label=f"{cls} (n={sel.sum()})")
  for cls in GROUPS:
    ax.axvline(res["centroids"][cls], color=PALETTE[cls], linewidth=1.2,
               linestyle=":", zorder=2)

  # arrows share the axis units, so scale them to the width of the cloud
  span = 0.42 * min(np.ptp(axes[:, 0]), np.ptp(axes[:, 1]))
  arrows = fit.assign(
    dx=fit["corr_dbRDA1"] * span, dy=fit["corr_axis2"] * span)
  labels = []
  for row in arrows.itertuples():
    ax.annotate("", xy=(row.dx, row.dy), xytext=(0, 0), zorder=5,
                arrowprops=dict(arrowstyle="->", color="#444444", linewidth=1.2))
    labels.append(ax.annotate(
      row.taxon, xy=(row.dx, row.dy), fontsize=8.5, color="#222222",
      xytext=(5 if row.dx >= 0 else -5, 0), textcoords="offset points",
      zorder=6, ha="left" if row.dx >= 0 else "right", va="center",
      bbox=dict(boxstyle="square,pad=0.1", facecolor="white",
                edgecolor="none", alpha=0.7)))
  spread_labels(fig, labels)

  ax.set_xlabel(f"db-RDA1 ({res['constrained_var'][0] * 100:.1f}%, constrained)")
  ax.set_ylabel(f"MDS1 ({res['mds_var'][0] * 100:.1f}%, unconstrained)")
  ax.set_title(f"{label} — db-RDA on Bray-Curtis ~ sec-type")
  ax.text(0.02, 0.02,
          f"constrained: pseudo-F({res['df1']},{res['df2']}) = {res['F']:.2f}, "
          f"$R^2$ = {res['R2']:.3f} (adj. {res['R2_adj']:.3f}), "
          f"p = {res['p']:.4f}\n"
          f"dotted line = group centroid, ellipse = {ELLIPSE_N_SD:g} SD, "
          f"arrows = {N_ARROWS} best-fitting genera",
          transform=ax.transAxes, fontsize=9.5, va="bottom", linespacing=1.5)
  ymin, ymax = ax.get_ylim()
  ax.set_ylim(ymin - 0.10 * (ymax - ymin), ymax + 0.16 * (ymax - ymin))
  ax.legend(loc="upper right", frameon=False, fontsize=11)
  fig.tight_layout()
  fig.savefig(os.path.join(figdir, f"dbrda_bray_{slug}.svg"))
  plt.show()

  scores = pd.DataFrame({
    "ID": df.loc[mask, "ID"].to_numpy(), "sec-type": y,
    "dbRDA1": res["dbRDA1"], "MDS1": res["MDS1"], "MDS2": res["MDS2"],
  })
  scores.to_csv(os.path.join(heredir, f"dbrda_scores_{slug}.csv"), index=False)
  fit.to_csv(os.path.join(heredir, f"dbrda_taxa_{slug}.csv"), index=False)

  print(f"[{label}] n={mask.sum()} "
        f"({(y == 'sec').sum()} sec / {(y == 'non-sec').sum()} non-sec)")
  print(f"  db-RDA  pseudo-F({res['df1']},{res['df2']}) = {res['F']:.3f}  "
        f"R2 = {res['R2']:.4f}  adj.R2 = {res['R2_adj']:.4f}  "
        f"p = {res['p']:.4f}  ({N_PERM} permutations)")
  print(f"  constrained axis carries {res['constrained_var'][0] * 100:.1f}% of "
        f"the total inertia; MDS1 {res['mds_var'][0] * 100:.1f}%")
  print("  best-fitting genera on db-RDA1: "
        + ", ".join(f"{r.taxon} ({r.corr_dbRDA1:+.2f})"
                    for r in fit.head(5).itertuples()))

  return {
    "timing": timing, "label": label, "n": int(mask.sum()),
    "n_sec": int((y == "sec").sum()), "n_non_sec": int((y == "non-sec").sum()),
    "dbRDA1_var": res["constrained_var"][0], "MDS1_var": res["mds_var"][0],
    "F": res["F"], "R2": res["R2"], "R2_adj": res["R2_adj"], "p": res["p"],
  }
# %%
summary = pd.DataFrame([run_timing(t) for t in TIMING_LABELS])
# three timings tested for the same effect
order = np.argsort(summary["p"].to_numpy())
ranked = summary["p"].to_numpy()[order] * len(order) / (np.arange(len(order)) + 1)
bh = np.minimum.accumulate(ranked[::-1])[::-1]
summary.loc[summary.index[order], "q"] = np.minimum(bh, 1.0)

summary.to_csv(os.path.join(heredir, "dbrda_summary.csv"), index=False)
print(summary[["label", "n", "F", "R2", "R2_adj", "p", "q"]].to_string(index=False))
