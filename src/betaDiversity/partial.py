# %%
"""Does the sec/non-sec beta-diversity signal survive adjustment for feeding type?

PCoA.py, dbRDA.py and CAP.py all compare sec with non-sec while ignoring how the
infant was fed, and feeding type is the strongest known driver of the infant gut
community. The CAP allocation success at 4-5 mos. (69%, p = 0.034 before
correction) is therefore ambiguous: it could be sec-type, or feeding type
showing through an unbalanced design.

This script separates the two. Only samples with a known feeding type are used,
and every model is run twice on that same subset:

  unadjusted   sec-type alone                          <- the earlier analysis
  partial      sec-type after conditioning on feeding  <- the adjustment

Running both on the identical subset matters: 62 of 263 samples have no feeding
type, so an adjusted model on fewer samples would differ from the earlier result
for two reasons at once.

Two adjusted models are fitted, matching the two earlier scripts:

  partial db-RDA   distance ~ sec-type + Condition(feeding), i.e. partial
                   PERMANOVA, plus a variation partitioning of the two variables
  partial CAP      the discriminant runs on principal coordinates taken from the
                   feeding-conditioned Gower matrix

Permutations are restricted to within feeding-type strata, which holds the
covariate fixed exactly rather than relying on a residual approximation. The
db-RDA and CAP cores are duplicated from dbRDA.py and CAP.py so that each script
here stays standalone, as elsewhere in this repo.
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

N_PERM = 9999      # db-RDA pseudo-F (closed form per permutation)
N_PERM_CV = 999    # CAP allocation success (re-runs the whole m selection)
SEED = 0
PALETTE = {"sec": "#d62728", "non-sec": "#1f77b4"}
GROUPS = ["sec", "non-sec"]
# the variation partitioning panel colours *variables*, not groups
VAR_COLORS = {"sec-type | feeding": "#d62728", "shared": "#9a9a9a",
              "feeding | sec-type": "#1f77b4"}
INK = "#222222"
MUTED = "#7a7a7a"
N_TAXA = 12

JP_TO_EN = {"母乳": "breastMilk", "人工": "formula", "混合": "both"}
NUTRITION_TYPES = ["both", "breastMilk", "formula"]   # "N/A" is dropped

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


def read_csv_jp(path):
  """The Japanese-labelled file has been seen as both cp932 and utf-8."""
  for enc in ["utf-8", "cp932"]:
    try:
      return pd.read_csv(path, header=0, encoding=enc)
    except UnicodeDecodeError:
      continue
  raise UnicodeDecodeError(f"cannot decode {path} as utf-8 or cp932")


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

# join the infant feeding type on family (= cohortID[:6]); the file carries one
# type per cohort with no month, so the same type applies to every timing of a
# family - at 4-5 mos. that is the weakest assumption in this script
nutri = read_csv_jp(os.path.join(datadir, "formulaOrBreastMilk_development.csv"))
nutri["family"] = nutri["cohortID"].astype(str).str[:6].astype(int)
nutri["type"] = nutri["type"].map(JP_TO_EN)
feeding_all = (df[["family"]].merge(nutri[["family", "type"]], on="family", how="left")
               ["type"].to_numpy(dtype=object))
known = np.isin(feeding_all, NUTRITION_TYPES)
print(f"feeding type joined: {known.sum()} / {len(df)}")
# %%
def gower(D):
  """Gower's centred matrix G = -0.5 J D^2 J, the response of a db-RDA."""
  n = D.shape[0]
  A = -0.5 * D ** 2
  J = np.eye(n) - np.ones((n, n)) / n
  G = J @ A @ J
  return (G + G.T) / 2


def dummies(labels, drop_first=True):
  """Centred indicator columns of a factor (k-1 columns for k levels)."""
  levels = np.unique(labels)
  Z = np.column_stack([(labels == g).astype(float)
                       for g in (levels[1:] if drop_first else levels)])
  return Z - Z.mean(axis=0)


def hat(Z):
  return Z @ np.linalg.pinv(Z.T @ Z) @ Z.T


def constrained_trace(G, Z):
  """Inertia of G explained by the columns of Z."""
  H = hat(Z)
  return float(np.trace(H @ G @ H))


def r2_adj(r2, n, k):
  """Ezekiel's adjustment, as used by vegan::varpart."""
  return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def strata_permutations(strata, n_perm, seed):
  """Index permutations that shuffle only within each stratum.

  Restricted this way, every permuted data set has exactly the observed
  feeding-type composition, so the covariate is held fixed by construction
  rather than by residualisation.
  """
  rng = np.random.default_rng(seed)
  blocks = [np.flatnonzero(strata == s) for s in np.unique(strata)]
  for _ in range(n_perm):
    idx = np.arange(len(strata))
    for b in blocks:
      idx[b] = rng.permutation(b)
    yield idx


def partial_test(G, z, C, n_perm=N_PERM, strata=None, seed=SEED):
  """Pseudo-F for one constraint, optionally conditioned on the columns of C.

  With C given this is a partial db-RDA (equivalently a partial PERMANOVA):
  both the response and the constraint are residualised on C first, and the
  denominator loses the conditioning degrees of freedom.
  """
  n = len(z)
  q = 0 if C is None else C.shape[1]
  if C is None:
    Gc, resid = G, np.eye(n)
  else:
    Hc = hat(C)
    resid = np.eye(n) - Hc
    Gc = resid @ G @ resid
  total_c = float(np.trace(Gc))

  def pseudo_f(zv):
    u = resid @ (zv - zv.mean())
    ss = float(u @ Gc @ u) / float(u @ u)   # Z is one column, so H = uu'/u'u
    return (ss / 1) / ((total_c - ss) / (n - q - 2)), ss

  f_obs, ss_obs = pseudo_f(z.astype(float))
  perms = (strata_permutations(strata, n_perm, seed) if strata is not None
           else (np.random.default_rng(seed + i).permutation(n)
                 for i in range(n_perm)))
  count = 0
  for idx in perms:
    count += pseudo_f(z[idx].astype(float))[0] >= f_obs
  return {"F": f_obs, "p": (count + 1) / (n_perm + 1),
          "R2": ss_obs / float(np.trace(G)), "df2": n - q - 2}


def pcoa_axes(G):
  """Principal coordinates of a Gower matrix (positive eigenvalues only)."""
  eigval, eigvec = np.linalg.eigh((G + G.T) / 2)
  order = np.argsort(eigval)[::-1]
  eigval, eigvec = eigval[order], eigvec[:, order]
  pos = eigval > 1e-10
  return eigvec[:, pos] * np.sqrt(eigval[pos]), eigval[pos] / eigval[pos].sum()


def loo_accuracy(Q, z):
  """Leave-one-out allocation success of a two-group LDA on the columns of Q.

  Equal priors, so the imbalance between sec and non-sec cannot be gamed by
  calling everything sec; the summary metric is the mean of the two per-group
  rates. Group means and the pooled scatter are downdated rather than
  recomputed, which keeps the permutation test affordable.
  """
  n, m = Q.shape
  hit = np.zeros(2)
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
  """Pick the number of principal coordinates by LOO allocation success."""
  curve = np.array([loo_accuracy(Q[:, :m], z)[0] for m in range(1, m_max + 1)])
  return int(np.argmax(curve)) + 1, curve


def cap_axis(Q, z):
  """CAP1 scores: the LDA discriminant on the given principal coordinates."""
  y = z - z.mean()
  b, *_ = np.linalg.lstsq(Q, y, rcond=None)
  scores = Q @ b
  return scores if scores[z == 1].mean() >= scores[z == 0].mean() else -scores


def cap_model(Q, z, m_max, strata=None, seed=SEED):
  """m selection, LOO allocation success and its permutation test.

  m is re-selected inside every permutation, so the null distribution carries
  the same selection advantage as the observed model; without that the test is
  anticonservative, and the null here sits well above 50%.
  """
  m, curve = select_m(Q, z, m_max)
  acc, per_group = loo_accuracy(Q[:, :m], z)
  perms = (strata_permutations(strata, N_PERM_CV, seed) if strata is not None
           else (np.random.default_rng(seed + i).permutation(len(z))
                 for i in range(N_PERM_CV)))
  perm_acc = np.empty(N_PERM_CV)
  for i, idx in enumerate(perms):
    zp = z[idx]
    mp, _ = select_m(Q, zp, m_max)
    perm_acc[i] = loo_accuracy(Q[:, :mp], zp)[0]
  return {"m": m, "curve": curve, "acc": acc, "per_group": per_group,
          "perm_acc": perm_acc,
          "p": (np.sum(perm_acc >= acc) + 1) / (N_PERM_CV + 1)}


def spearman_with(X, scores):
  """Spearman correlation of every taxon with the canonical axis."""
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
  mask = (timing_all == timing) & known
  X, y, feed = X_all[mask], labels_all[mask], feeding_all[mask]
  n = int(mask.sum())
  z = (y == "sec").astype(int)

  D = squareform(pdist(X, metric="braycurtis"))
  G = gower(D)
  Zs, Zf = dummies(y), dummies(feed)

  # --- db-RDA: sec-type alone, feeding alone, and each given the other -------
  unadj = partial_test(G, z, None)
  adj = partial_test(G, z, Zf, strata=feed)
  # feeding given sec-type: permuting feeding within sec-type strata
  Hf = hat(Zf)
  feed_alone = constrained_trace(G, Zf) / np.trace(G)
  both = constrained_trace(G, np.column_stack([Zs, Zf])) / np.trace(G)
  sec_alone = unadj["R2"]
  # Ezekiel-adjusted components, as in vegan::varpart
  a_sec = r2_adj(both, n, 3) - r2_adj(feed_alone, n, 2)
  c_feed = r2_adj(both, n, 3) - r2_adj(sec_alone, n, 1)
  b_shared = r2_adj(sec_alone, n, 1) + r2_adj(feed_alone, n, 2) - r2_adj(both, n, 3)

  # --- CAP: unadjusted and feeding-conditioned, on the same samples ----------
  m_max = int(min(15, min(z.sum(), (1 - z).sum()) - 3))
  Q_un, expl_un = pcoa_axes(G)
  R = np.eye(n) - Hf
  Q_adj, expl_adj = pcoa_axes(R @ G @ R)
  m_max = int(min(m_max, Q_un.shape[1], Q_adj.shape[1]))
  cap_un = cap_model(Q_un, z, m_max)
  cap_adj = cap_model(Q_adj, z, m_max, strata=feed, seed=SEED + 1)

  scores = cap_axis(Q_adj[:, :cap_adj["m"]], z)
  rho = spearman_with(X, scores)
  top = np.argsort(np.abs(rho))[::-1][:N_TAXA]
  contrib = pd.DataFrame({
    "taxon": [short_name(taxa[i]) for i in top],
    "lineage": taxa.to_numpy()[top],
    "spearman_CAP1_adjusted": rho[top],
    "mean_sec": X[z == 1][:, top].mean(axis=0),
    "mean_non_sec": X[z == 0][:, top].mean(axis=0),
  })
  contrib.to_csv(os.path.join(heredir, f"partial_taxa_{slug}.csv"), index=False)
  pd.DataFrame({"ID": df.loc[mask, "ID"].to_numpy(), "sec-type": y,
                "feeding": feed, "CAP1_adjusted": scores}).to_csv(
    os.path.join(heredir, f"partial_scores_{slug}.csv"), index=False)

  plot_cap(label, slug, scores, z, cap_un, cap_adj, adj, contrib,
           expl_adj[:cap_adj["m"]].sum())

  print(f"\n[{label}] n={n} ({z.sum()} sec / {(1 - z).sum()} non-sec), "
        f"feeding: " + ", ".join(f"{t} {int((feed == t).sum())}"
                                 for t in NUTRITION_TYPES))
  print(f"  db-RDA  unadjusted  F = {unadj['F']:.3f}  R2 = {unadj['R2']:.4f}  "
        f"p = {unadj['p']:.4f}")
  print(f"  db-RDA  partial     F = {adj['F']:.3f}  R2 = {adj['R2']:.4f}  "
        f"p = {adj['p']:.4f}  (feeding conditioned out)")
  print(f"  varpart adj.R2  sec|feed {a_sec * 100:+.2f}%  "
        f"shared {b_shared * 100:+.2f}%  feed|sec {c_feed * 100:+.2f}%")
  print(f"  CAP     unadjusted  m = {cap_un['m']}  "
        f"LOO {cap_un['acc'] * 100:.1f}%  p = {cap_un['p']:.4f}")
  print(f"  CAP     partial     m = {cap_adj['m']}  "
        f"LOO {cap_adj['acc'] * 100:.1f}% "
        f"(sec {cap_adj['per_group'][1] * 100:.0f}%, "
        f"non-sec {cap_adj['per_group'][0] * 100:.0f}%)  "
        f"p = {cap_adj['p']:.4f}")
  return {
    "timing": timing, "label": label, "n": n,
    "n_sec": int(z.sum()), "n_non_sec": int((1 - z).sum()),
    "dbrda_F": unadj["F"], "dbrda_R2": unadj["R2"], "dbrda_p": unadj["p"],
    "dbrda_partial_F": adj["F"], "dbrda_partial_R2": adj["R2"],
    "dbrda_partial_p": adj["p"],
    "varpart_sec_given_feeding": a_sec, "varpart_shared": b_shared,
    "varpart_feeding_given_sec": c_feed,
    "cap_m": cap_un["m"], "cap_acc": cap_un["acc"], "cap_p": cap_un["p"],
    "cap_partial_m": cap_adj["m"], "cap_partial_acc": cap_adj["acc"],
    "cap_partial_sec": cap_adj["per_group"][1],
    "cap_partial_non_sec": cap_adj["per_group"][0],
    "cap_partial_p": cap_adj["p"],
    "cap_partial_null_mean": cap_adj["perm_acc"].mean(),
  }
# %%
def plot_cap(label, slug, scores, z, cap_un, cap_adj, adj, contrib, var_m):
  fig, (ax, bx, cx) = plt.subplots(
    1, 3, figsize=(17.5, 5.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})

  rng = np.random.default_rng(SEED)
  ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
  for k, cls in enumerate(GROUPS):
    sel = z == (1 if cls == "sec" else 0)
    jitter = rng.uniform(-0.16, 0.16, sel.sum())
    ax.scatter(scores[sel], np.full(sel.sum(), k) + jitter, s=46,
               color=PALETTE[cls], edgecolor="white", linewidth=0.8,
               alpha=0.8, zorder=3, label=f"{cls} (n={sel.sum()})")
    ax.plot([scores[sel].mean()] * 2, [k - 0.3, k + 0.3], color=PALETTE[cls],
            linewidth=2.5, zorder=4)
  ax.set_yticks([0, 1])
  ax.set_yticklabels(GROUPS)
  ax.set_ylim(-0.6, 1.6)
  ax.set_xlabel("CAP1, feeding type conditioned out")
  ax.set_title(f"{label} — partial CAP")
  ax.text(0.0, -0.22,
          f"m = {cap_adj['m']} axes ({var_m * 100:.0f}% of variation)\n"
          f"partial db-RDA F = {adj['F']:.2f}, p = {adj['p']:.4f}",
          transform=ax.transAxes, fontsize=10, va="top", color=MUTED,
          linespacing=1.6)
  ax.legend(loc="upper right", frameon=False, fontsize=10)
  sns.despine(ax=ax, left=True)

  bx.hist(cap_adj["perm_acc"] * 100, bins=30, color="#c9c9c9", zorder=2)
  bx.axvline(cap_adj["acc"] * 100, color=INK, linewidth=2.5, zorder=4)
  bx.annotate(f"adjusted {cap_adj['acc'] * 100:.0f}%\np = {cap_adj['p']:.3f}",
              xy=(cap_adj["acc"] * 100, bx.get_ylim()[1] * 0.95),
              xytext=(8, 0), textcoords="offset points", fontsize=10,
              color=INK, va="top")
  bx.axvline(cap_un["acc"] * 100, color=MUTED, linewidth=2,
             linestyle="--", zorder=3)
  bx.annotate(f"unadjusted {cap_un['acc'] * 100:.0f}%",
              xy=(cap_un["acc"] * 100, bx.get_ylim()[1] * 0.45),
              xytext=(-8, 0), textcoords="offset points", fontsize=10,
              color=MUTED, va="top", ha="right")
  bx.set_xlabel("LOO balanced accuracy (%)")
  bx.set_ylabel("permutations")
  bx.set_title(f"null within feeding strata ({N_PERM_CV} perm.)")
  sns.despine(ax=bx)

  order = contrib.iloc[::-1]
  pos = np.arange(len(order))
  colors = [PALETTE["sec"] if r > 0 else PALETTE["non-sec"]
            for r in order["spearman_CAP1_adjusted"]]
  cx.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
  cx.barh(pos, order["spearman_CAP1_adjusted"], height=0.62, color=colors,
          zorder=3)
  cx.set_yticks(pos)
  cx.set_yticklabels(order["taxon"], fontsize=9.5)
  cx.set_xlabel("Spearman correlation with CAP1")
  cx.set_title("Genera contributing to the axis")
  cx.text(0.5, -0.22, "positive = the sec side", transform=cx.transAxes,
          fontsize=10, va="top", ha="center", color=MUTED)
  sns.despine(ax=cx, left=True)
  cx.tick_params(axis="y", length=0)

  fig.tight_layout()
  fig.savefig(os.path.join(figdir, f"partial_cap_{slug}.svg"))
  plt.show()


def plot_varpart(summary):
  fig, ax = plt.subplots(figsize=(9.5, 5.4))
  components = list(VAR_COLORS)
  cols = ["varpart_sec_given_feeding", "varpart_shared",
          "varpart_feeding_given_sec"]
  height = 0.24
  base = np.arange(len(summary))[::-1]
  ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
  for k, (name, col) in enumerate(zip(components, cols)):
    vals = summary[col].to_numpy() * 100
    ypos = base + (1 - k) * height
    ax.barh(ypos, vals, height=height * 0.88, color=VAR_COLORS[name],
            zorder=3, label=name)
    for yv, v in zip(ypos, vals):
      ax.text(v + (0.06 if v >= 0 else -0.06), yv, f"{v:+.1f}",
              va="center", ha="left" if v >= 0 else "right", fontsize=9,
              color=MUTED)
  ax.set_yticks(base)
  ax.set_yticklabels([f"{r.label}\n(n={r.n})" for r in summary.itertuples()],
                     fontsize=11)
  ax.set_xlabel("adjusted $R^2$ of the Bray-Curtis variation (%)")
  ax.set_title("Variation partitioning: sec-type vs feeding type")
  # room for the value labels at the ends of the bars
  lo, hi = ax.get_xlim()
  ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.12 * (hi - lo))
  ax.text(0.0, -0.26,
          "colours are variables, not groups;  a negative share means the two "
          "variables explain less together than apart",
          transform=ax.transAxes, fontsize=9.5, va="top", color=MUTED)
  ax.legend(loc="upper right", frameon=False, fontsize=10)
  sns.despine(ax=ax, left=True)
  ax.tick_params(axis="y", length=0)
  fig.tight_layout()
  fig.savefig(os.path.join(figdir, "partial_varpart.svg"))
  plt.show()
# %%
summary = pd.DataFrame([run_timing(t) for t in TIMING_LABELS])
# three timings tested for the same effect
for col in ["dbrda_partial_p", "cap_partial_p"]:
  order = np.argsort(summary[col].to_numpy())
  ranked = summary[col].to_numpy()[order] * len(order) / (np.arange(len(order)) + 1)
  bh = np.minimum.accumulate(ranked[::-1])[::-1]
  summary.loc[summary.index[order], col.removesuffix("_p") + "_q"] = np.minimum(bh, 1.0)

plot_varpart(summary)
summary.to_csv(os.path.join(heredir, "partial_summary.csv"), index=False)
print()
print(summary[["label", "n", "dbrda_p", "dbrda_partial_p", "dbrda_partial_q",
               "cap_acc", "cap_p", "cap_partial_acc", "cap_partial_p",
               "cap_partial_q"]].to_string(index=False))
