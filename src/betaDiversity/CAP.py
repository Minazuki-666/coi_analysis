# %%
"""Canonical analysis of principal coordinates (Anderson & Willis 2003).

The supervised counterpart of PCoA.py: the same Bray-Curtis principal
coordinates are computed first, then a discriminant analysis is run on the
leading m of them, so the plotted axis is the direction that best separates
sec from non-sec rather than the direction of most total variation.

A supervised axis always looks separated, so nothing here is read off the
ordination alone. m is chosen by leave-one-out allocation success, and both
the canonical correlation and that allocation success are tested against
label permutations - with m re-selected inside every permutation, so the null
carries the same selection advantage as the observed model.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

sns.set_theme(style="white")
sns.set_context("talk", font_scale=0.9)

N_PERM = 9999      # canonical correlation test (cheap, closed form per permutation)
N_PERM_CV = 999    # allocation-success test (re-runs the whole m selection)
SEED = 0
PALETTE = {"sec": "#d62728", "non-sec": "#1f77b4"}
GROUPS = ["sec", "non-sec"]   # fixed order so the legend matches across timings
INK = "#222222"
MUTED = "#7a7a7a"
N_TAXA = 12        # taxa listed on the contribution panel

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
def pcoa_axes(D):
  """Principal coordinates of a distance matrix (positive eigenvalues only)."""
  n = D.shape[0]
  A = -0.5 * D ** 2
  J = np.eye(n) - np.ones((n, n)) / n
  G = J @ A @ J
  eigval, eigvec = np.linalg.eigh((G + G.T) / 2)
  order = np.argsort(eigval)[::-1]
  eigval, eigvec = eigval[order], eigvec[:, order]
  pos = eigval > 1e-10
  return eigvec[:, pos] * np.sqrt(eigval[pos]), eigval[pos] / eigval[pos].sum()


def loo_accuracy(Q, z):
  """Leave-one-out allocation success of a two-group LDA on the columns of Q.

  Equal priors, so the 4:1 imbalance between sec and non-sec cannot be gamed
  by calling everything sec; the summary metric is therefore the mean of the
  two per-group rates. Group means and the pooled scatter are downdated
  rather than recomputed, which keeps the permutation test affordable.
  """
  n, m = Q.shape
  hit = np.zeros(2)
  # per-group running totals, so leaving one sample out is a rank-one downdate
  totals = [(Q[z == g].sum(axis=0), Q[z == g].T @ Q[z == g], int((z == g).sum()))
            for g in (0, 1)]
  ridge = 1e-8 * np.eye(m)
  for i in range(n):
    gi = z[i]
    means, scatter = [], np.zeros((m, m))
    for g in (0, 1):
      s, c, k = totals[g]
      if g == gi:
        s, c, k = s - Q[i], c - np.outer(Q[i], Q[i]), k - 1
      mu = s / k
      means.append(mu)
      scatter += c - k * np.outer(mu, mu)
    W = scatter / (n - 1 - 2) + ridge
    d = [np.dot(Q[i] - mu, np.linalg.solve(W, Q[i] - mu)) for mu in means]
    hit[gi] += int(np.argmin(d) == gi)
  per_group = hit / np.array([(z == 0).sum(), (z == 1).sum()])
  return per_group.mean(), per_group


def select_m(Q, z, m_max):
  """Pick the number of principal coordinates by LOO allocation success.

  Ties go to the smaller m: extra axes that do not classify better only add
  variance, and Anderson & Willis warn against letting m grow towards n.
  """
  curve = np.array([loo_accuracy(Q[:, :m], z)[0] for m in range(1, m_max + 1)])
  return int(np.argmax(curve)) + 1, curve


def canonical(Q, z):
  """Squared canonical correlation and the discriminant scores.

  With two groups the canonical correlation between the principal coordinates
  and the group indicator is the multiple correlation of that indicator on the
  coordinates, and the canonical axis is the corresponding fitted direction -
  identical to the LDA discriminant up to scale.
  """
  y = z - z.mean()
  b, *_ = np.linalg.lstsq(Q, y, rcond=None)
  scores = Q @ b
  delta2 = float((scores @ y) ** 2 / ((scores @ scores) * (y @ y)))
  return scores, delta2


def spearman_with(X, scores):
  """Spearman correlation of every taxon with the canonical axis.

  Anderson & Willis overlay species on a CAP plot by rank correlation, which
  suits the many-zero, non-linear genus abundances better than Pearson.
  """
  R = np.apply_along_axis(rankdata, 0, X)
  s = rankdata(scores)
  R = R - R.mean(axis=0)
  s = s - s.mean()
  denom = np.sqrt((R ** 2).sum(axis=0) * (s ** 2).sum())
  with np.errstate(invalid="ignore", divide="ignore"):
    return np.where(denom > 0, (R.T @ s) / denom, 0.0)
# %%
def run_timing(timing):
  label, slug = TIMING_LABELS[timing]
  mask = timing_all == timing
  X, y = X_all[mask], labels_all[mask]
  n = int(mask.sum())
  # z = 1 for sec, 0 for non-sec
  z = (y == "sec").astype(int)

  D = squareform(pdist(X, metric="braycurtis"))
  Q, explained = pcoa_axes(D)
  # the pooled within-group covariance is estimated from the smaller group, so
  # m stays well below it; 15 axes is the ceiling regardless
  m_max = int(min(15, min(z.sum(), (1 - z).sum()) - 3, Q.shape[1]))
  m, curve = select_m(Q, z, m_max)
  acc, per_group = loo_accuracy(Q[:, :m], z)

  Qm = Q[:, :m]
  scores, delta2 = canonical(Qm, z)
  # orient the axis so sec sits on the positive side in every timing
  if scores[z == 1].mean() < scores[z == 0].mean():
    scores = -scores

  rng = np.random.default_rng(SEED)
  # canonical correlation under permutation, with m held at its observed value
  P = Qm @ np.linalg.pinv(Qm.T @ Qm) @ Qm.T
  perm_delta = np.empty(N_PERM)
  for i in range(N_PERM):
    yp = z[rng.permutation(n)].astype(float)
    yp -= yp.mean()
    perm_delta[i] = (yp @ P @ yp) / (yp @ yp)
  p_delta = (np.sum(perm_delta >= delta2) + 1) / (N_PERM + 1)

  # allocation success under permutation, re-selecting m each time so the null
  # gets the same freedom to fit as the observed model
  rng = np.random.default_rng(SEED + 1)
  perm_acc = np.empty(N_PERM_CV)
  for i in range(N_PERM_CV):
    zp = z[rng.permutation(n)]
    mp, _ = select_m(Q, zp, m_max)
    perm_acc[i] = loo_accuracy(Q[:, :mp], zp)[0]
  p_acc = (np.sum(perm_acc >= acc) + 1) / (N_PERM_CV + 1)

  rho = spearman_with(X, scores)
  top = np.argsort(np.abs(rho))[::-1][:N_TAXA]
  contrib = pd.DataFrame({
    "taxon": [short_name(taxa[i]) for i in top],
    "lineage": taxa.to_numpy()[top],
    "spearman_CAP1": rho[top],
    "mean_sec": X[z == 1][:, top].mean(axis=0),
    "mean_non_sec": X[z == 0][:, top].mean(axis=0),
  })
  contrib["higher_in"] = np.where(
    contrib["mean_sec"] >= contrib["mean_non_sec"], "sec", "non-sec")

  plot_cap(label, slug, scores, z, delta2, p_delta, acc, per_group, p_acc,
           m, explained[:m].sum(), contrib)
  plot_diagnostics(label, slug, curve, m, acc, perm_acc, p_acc)

  pd.DataFrame({"ID": df.loc[mask, "ID"].to_numpy(), "sec-type": y,
                "CAP1": scores}).to_csv(
    os.path.join(heredir, f"cap_scores_{slug}.csv"), index=False)
  contrib.to_csv(os.path.join(heredir, f"cap_taxa_{slug}.csv"), index=False)

  print(f"[{label}] n={n} ({z.sum()} sec / {(1 - z).sum()} non-sec)")
  print(f"  m = {m} of {m_max} candidate axes "
        f"({explained[:m].sum() * 100:.1f}% of the variation)")
  print(f"  canonical  delta2 = {delta2:.4f}  p = {p_delta:.4f}")
  print(f"  LOO allocation  balanced = {acc * 100:.1f}%  "
        f"(sec {per_group[1] * 100:.1f}%, non-sec {per_group[0] * 100:.1f}%)  "
        f"p = {p_acc:.4f}")
  return {
    "timing": timing, "label": label, "n": n,
    "n_sec": int(z.sum()), "n_non_sec": int((1 - z).sum()),
    "m": m, "m_variation": explained[:m].sum(),
    "delta2": delta2, "delta2_p": p_delta,
    "loo_balanced_accuracy": acc, "loo_sec": per_group[1],
    "loo_non_sec": per_group[0], "loo_p": p_acc,
    "perm_acc_mean": perm_acc.mean(),
  }
# %%
def plot_cap(label, slug, scores, z, delta2, p_delta, acc, per_group, p_acc,
             m, var_m, contrib):
  fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})

  rng = np.random.default_rng(SEED)
  ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
  for k, cls in enumerate(GROUPS):
    sel = z == (1 if cls == "sec" else 0)
    jitter = rng.uniform(-0.16, 0.16, sel.sum())
    ax.scatter(scores[sel], np.full(sel.sum(), k) + jitter, s=48,
               color=PALETTE[cls], edgecolor="white", linewidth=0.8,
               alpha=0.8, zorder=3, label=f"{cls} (n={sel.sum()})")
    ax.plot([scores[sel].mean()] * 2, [k - 0.3, k + 0.3], color=PALETTE[cls],
            linewidth=2.5, zorder=4)
  ax.set_yticks([0, 1])
  ax.set_yticklabels(GROUPS)
  ax.set_ylim(-0.6, 1.6)
  ax.set_xlabel("CAP1 (canonical axis)")
  ax.set_title(f"{label} — CAP on Bray-Curtis")
  ax.text(0.02, -0.20,
          f"m = {m} axes ({var_m * 100:.0f}% of variation)   "
          f"$\\delta^2$ = {delta2:.3f}, p = {p_delta:.4f}\n"
          f"LOO allocation {acc * 100:.0f}% balanced "
          f"(sec {per_group[1] * 100:.0f}%, non-sec {per_group[0] * 100:.0f}%), "
          f"p = {p_acc:.3f}   |   thick bar = group mean",
          transform=ax.transAxes, fontsize=10, va="top", color=MUTED,
          linespacing=1.6)
  ax.legend(loc="upper right", frameon=False, fontsize=11)
  sns.despine(ax=ax, left=True)

  order = contrib.iloc[::-1]
  pos = np.arange(len(order))
  # colour by the side of the axis, which the bar length already shows; the
  # group with the larger mean abundance can differ (a rank correlation is not
  # a mean difference) and is kept in the exported table instead
  colors = [PALETTE["sec"] if r > 0 else PALETTE["non-sec"]
            for r in order["spearman_CAP1"]]
  bx.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
  bx.barh(pos, order["spearman_CAP1"], height=0.62, color=colors, zorder=3)
  bx.set_yticks(pos)
  bx.set_yticklabels(order["taxon"], fontsize=10)
  bx.set_xlabel("Spearman correlation with CAP1")
  bx.set_title("Genera contributing to the axis")
  bx.text(0.5, -0.20,
          "positive = the sec side of CAP1",
          transform=bx.transAxes, fontsize=10, va="top", ha="center",
          color=MUTED)
  sns.despine(ax=bx, left=True)
  bx.tick_params(axis="y", length=0)

  fig.tight_layout()
  fig.savefig(os.path.join(figdir, f"cap_bray_{slug}.svg"))
  plt.show()


def plot_diagnostics(label, slug, curve, m, acc, perm_acc, p_acc):
  fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 5.2))

  ms = np.arange(1, len(curve) + 1)
  ax.plot(ms, curve * 100, color=MUTED, linewidth=2, marker="o",
          markersize=5, zorder=3)
  ax.scatter([m], [acc * 100], s=110, color=INK, zorder=4)
  ax.annotate(f"m = {m}", xy=(m, acc * 100), xytext=(9, -3),
              textcoords="offset points", fontsize=10, color=INK, va="center")
  ax.set_xticks(ms[::2] if len(ms) > 8 else ms)
  ax.axhline(50, color="lightgray", linewidth=1.2, linestyle="--", zorder=1)
  ax.text(ms[-1], 50, " chance", va="center", fontsize=9, color=MUTED)
  ax.set_xlabel("principal coordinates used (m)")
  ax.set_ylabel("LOO balanced accuracy (%)")
  ax.set_title(f"{label} — choice of m")
  sns.despine(ax=ax)

  bx.hist(perm_acc * 100, bins=30, color="#c9c9c9", zorder=2)
  bx.axvline(acc * 100, color=INK, linewidth=2.5, zorder=4)
  bx.annotate(f"observed {acc * 100:.0f}%\np = {p_acc:.3f}",
              xy=(acc * 100, bx.get_ylim()[1] * 0.92),
              xytext=(8, 0), textcoords="offset points", fontsize=10,
              color=INK, va="top")
  bx.set_xlabel("LOO balanced accuracy under permuted labels (%)")
  bx.set_ylabel("permutations")
  bx.set_title(f"null distribution ({len(perm_acc)} permutations)")
  sns.despine(ax=bx)

  fig.tight_layout()
  fig.savefig(os.path.join(figdir, f"cap_diagnostics_{slug}.svg"))
  plt.show()
# %%
summary = pd.DataFrame([run_timing(t) for t in TIMING_LABELS])
# three timings tested for the same effect
for col in ["delta2_p", "loo_p"]:
  order = np.argsort(summary[col].to_numpy())
  ranked = summary[col].to_numpy()[order] * len(order) / (np.arange(len(order)) + 1)
  bh = np.minimum.accumulate(ranked[::-1])[::-1]
  summary.loc[summary.index[order], col.removesuffix("_p") + "_q"] = np.minimum(bh, 1.0)

summary.to_csv(os.path.join(heredir, "cap_summary.csv"), index=False)
print(summary[["label", "n", "m", "delta2", "delta2_p", "delta2_q",
               "loo_balanced_accuracy", "loo_p", "loo_q"]].to_string(index=False))
