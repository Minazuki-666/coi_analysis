# %%
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
# %%
def pcoa(D):
  """Classical (Gower) principal coordinate analysis of a distance matrix.

  Returns the coordinates on the axes with positive eigenvalues, the fraction
  of variation each explains, and the full eigenvalue spectrum (Bray-Curtis is
  not Euclidean, so the tail of negative eigenvalues is expected).
  """
  n = D.shape[0]
  A = -0.5 * D ** 2
  J = np.eye(n) - np.ones((n, n)) / n
  G = J @ A @ J
  eigval, eigvec = np.linalg.eigh((G + G.T) / 2)
  order = np.argsort(eigval)[::-1]
  eigval, eigvec = eigval[order], eigvec[:, order]
  pos = eigval > 1e-10
  coords = eigvec[:, pos] * np.sqrt(eigval[pos])
  # explained fraction is over the positive eigenvalues only, as in vegan/skbio
  explained = eigval[pos] / eigval[pos].sum()
  return coords, explained, eigval


def permanova(D, labels, n_perm=N_PERM, seed=SEED):
  """One-way PERMANOVA (Anderson 2001) on a distance matrix."""
  n = D.shape[0]
  groups = np.unique(labels)
  a = len(groups)
  D2 = D ** 2
  total = D2.sum() / 2 / n   # sum of squares about the grand centroid

  def within(lab):
    ss = 0.0
    for g in groups:
      idx = np.flatnonzero(lab == g)
      ss += D2[np.ix_(idx, idx)].sum() / 2 / len(idx)
    return ss

  def pseudo_f(lab):
    ssw = within(lab)
    return ((total - ssw) / (a - 1)) / (ssw / (n - a))

  f_obs = pseudo_f(labels)
  rng = np.random.default_rng(seed)
  perm = np.array([pseudo_f(rng.permutation(labels)) for _ in range(n_perm)])
  p = (np.sum(perm >= f_obs) + 1) / (n_perm + 1)
  # R2 = SSA / SST, the share of the Bray-Curtis variation explained by sec-type
  r2 = 1 - within(labels) / total
  return {"F": f_obs, "p": p, "R2": r2, "df1": a - 1, "df2": n - a, "perm": perm}


def permdisp(D, labels, n_perm=N_PERM, seed=SEED):
  """PERMDISP (Anderson 2006): are the within-group dispersions homogeneous?

  Distances to the group centroid are taken in the full principal coordinate
  space; the negative eigenvalue axes enter with a minus sign, as in
  vegan::betadisper.
  """
  n = D.shape[0]
  A = -0.5 * D ** 2
  J = np.eye(n) - np.ones((n, n)) / n
  G = J @ A @ J
  eigval, eigvec = np.linalg.eigh((G + G.T) / 2)
  keep = np.abs(eigval) > 1e-10
  coords = eigvec[:, keep] * np.sqrt(np.abs(eigval[keep]))
  sign = np.sign(eigval[keep])

  groups = np.unique(labels)
  dist = np.empty(n)
  for g in groups:
    idx = np.flatnonzero(labels == g)
    centroid = coords[idx].mean(axis=0)
    d2 = ((coords[idx] - centroid) ** 2 * sign).sum(axis=1)
    dist[idx] = np.sqrt(np.abs(d2))

  def anova_f(d):
    grand = d.mean()
    ssb = sum(len(d[labels == g]) * (d[labels == g].mean() - grand) ** 2
              for g in groups)
    ssw = sum(((d[labels == g] - d[labels == g].mean()) ** 2).sum()
              for g in groups)
    return (ssb / (len(groups) - 1)) / (ssw / (n - len(groups)))

  f_obs = anova_f(dist)
  rng = np.random.default_rng(seed)
  # the residuals are permuted by shuffling the sample-to-group assignment
  perm = np.empty(n_perm)
  for i in range(n_perm):
    perm[i] = anova_f(dist[rng.permutation(n)])
  return {"F": f_obs, "p": (np.sum(perm >= f_obs) + 1) / (n_perm + 1),
          "dispersion": {g: dist[labels == g].mean() for g in groups}}


def data_ellipse(ax, x, y, color, n_sd=ELLIPSE_N_SD):
  """Ellipse covering the group's own scatter (vegan's ordiellipse kind="sd").

  The semi-axes are n_sd standard deviations of the group's scores along the
  principal axes of its covariance.
  """
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
  ax.scatter([x.mean()], [y.mean()], s=90, marker="X", color=color,
             edgecolor="black", linewidth=0.8, zorder=4)


# %%
def run_timing(timing):
  label, slug = TIMING_LABELS[timing]
  mask = timing_all == timing
  X, y = X_all[mask], labels_all[mask]

  D = squareform(pdist(X, metric="braycurtis"))
  coords, explained, eigval = pcoa(D)
  res = permanova(D, y)
  disp = permdisp(D, y)

  fig, ax = plt.subplots(figsize=(7.2, 6.4))
  ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
  for cls in GROUPS:
    sel = y == cls
    data_ellipse(ax, coords[sel, 0], coords[sel, 1], PALETTE[cls])
    ax.scatter(coords[sel, 0], coords[sel, 1], s=55, color=PALETTE[cls],
               edgecolor="white", linewidth=0.5, alpha=0.75, zorder=3,
               label=f"{cls} (n={sel.sum()})")
  ax.set_xlabel(f"PCo1 ({explained[0] * 100:.1f}%)")
  ax.set_ylabel(f"PCo2 ({explained[1] * 100:.1f}%)")
  ax.set_title(f"{label} — Bray-Curtis PCoA")
  ax.text(0.02, 0.02,
          f"PERMANOVA: pseudo-F({res['df1']},{res['df2']}) = {res['F']:.2f}, "
          f"$R^2$ = {res['R2']:.3f}, p = {res['p']:.4f}\n"
          f"X = group centroid, ellipse = {ELLIPSE_N_SD:g} SD data ellipse",
          transform=ax.transAxes, fontsize=10, va="bottom", linespacing=1.5)
  ax.set_aspect("equal", adjustable="datalim")
  # headroom so the legend does not sit on the cloud
  ymin, ymax = ax.get_ylim()
  ax.set_ylim(ymin - 0.10 * (ymax - ymin), ymax + 0.14 * (ymax - ymin))
  ax.legend(loc="upper right", frameon=False, fontsize=11)
  fig.tight_layout()
  fig.savefig(os.path.join(figdir, f"pcoa_bray_{slug}.svg"))
  plt.show()

  scores = pd.DataFrame(coords[:, :5],
                        columns=[f"PCo{i + 1}" for i in range(5)])
  scores.insert(0, "sec-type", y)
  scores.insert(0, "ID", df.loc[mask, "ID"].to_numpy())
  scores.to_csv(os.path.join(heredir, f"pcoa_scores_{slug}.csv"), index=False)

  print(f"[{label}] n={mask.sum()} "
        f"({(y == 'sec').sum()} sec / {(y == 'non-sec').sum()} non-sec)")
  print(f"  PERMANOVA  pseudo-F = {res['F']:.3f}  R2 = {res['R2']:.4f}  "
        f"p = {res['p']:.4f}  ({N_PERM} permutations)")
  print(f"  PERMDISP   F = {disp['F']:.3f}  p = {disp['p']:.4f}  "
        f"mean distance to centroid: "
        + ", ".join(f"{g} {v:.3f}" for g, v in disp["dispersion"].items()))
  print(f"  negative eigenvalues: {(eigval < -1e-10).sum()} "
        f"(|min| / max eigenvalue = {abs(eigval.min()) / eigval.max():.3f})")

  return {
    "timing": timing, "label": label, "n": int(mask.sum()),
    "n_sec": int((y == "sec").sum()), "n_non_sec": int((y == "non-sec").sum()),
    "PCo1_explained": explained[0], "PCo2_explained": explained[1],
    "permanova_F": res["F"], "permanova_R2": res["R2"], "permanova_p": res["p"],
    "permdisp_F": disp["F"], "permdisp_p": disp["p"],
    "dispersion_sec": disp["dispersion"]["sec"],
    "dispersion_non_sec": disp["dispersion"]["non-sec"],
  }
# %%
summary = pd.DataFrame([run_timing(t) for t in TIMING_LABELS])
# three timings tested for the same effect
order = np.argsort(summary["permanova_p"].to_numpy())
ranked = summary["permanova_p"].to_numpy()[order] * len(order) / (np.arange(len(order)) + 1)
bh = np.minimum.accumulate(ranked[::-1])[::-1]
summary.loc[summary.index[order], "permanova_q"] = np.minimum(bh, 1.0)

summary.to_csv(os.path.join(heredir, "pcoa_permanova_summary.csv"), index=False)
print(summary[["label", "n", "permanova_F", "permanova_R2",
               "permanova_p", "permanova_q", "permdisp_p"]].to_string(index=False))
