# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import f as f_dist

sns.set_theme(style="white")
sns.set_context("talk", font_scale=0.9)

N_PERM = 1000
N_SPLITS = 7
MAX_ORTHO = 3
SEED = 0

# formulaOrBreastMilk_development.csv carries one feeding type per cohort (no month),
# so the type is applied to every timing of that family.
JP_TO_EN = {"母乳": "breastMilk", "人工": "formula", "混合": "both"}
NUTRITION_TYPES = ["both", "breastMilk", "formula"]   # "N/A" is dropped
SLINE_CMAP = sns.color_palette("mako_r", as_cmap=True)
# the S-line colour is |corr|, so fix the scale to its full range and keep
# the colouring comparable across subsets
SLINE_VMAX = 1.0
N_ANNOTATE = 10   # S-line bins labelled, ranked by |p(corr)[1]|

# how each subset is labelled in the figures, and the slug used in file names
SUBSET_LABELS = {
  "timing3": ("Newborn", "newborn"),
  "timing4": ("1 mo.", "1mo"),
  "timing5": ("4-5 mos.", "4-5mos"),
  "matched_timing3": ("Newborn (feeding-type matched)", "newborn_matched"),
}
# %%
heredir = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.dirname(os.path.dirname(heredir))
datadir = os.path.join(basedir, "data")
figdir = os.path.join(heredir, "fig")
os.makedirs(figdir, exist_ok=True)
# %%
df = pd.read_csv(os.path.join(datadir, "feces_BinningResults.csv"), header=0)
df = df[df["age"] == "c"].reset_index(drop=True)

ppm = df.columns[6:].astype(float).to_numpy()
X_all = df.iloc[:, 6:].to_numpy(dtype=float)
labels_all = df["sec-type"].to_numpy()
timing_all = df["timing"].to_numpy()
# %%
def read_csv_jp(path):
  """The Japanese-labelled file has been seen as both cp932 and utf-8."""
  for enc in ["utf-8", "cp932"]:
    try:
      return pd.read_csv(path, header=0, encoding=enc)
    except UnicodeDecodeError:
      continue
  raise UnicodeDecodeError(f"cannot decode {path} as utf-8 or cp932")


# join the infant feeding type on family (= cohortID[:6])
nutri = read_csv_jp(os.path.join(datadir, "formulaOrBreastMilk_development.csv"))
nutri["family"] = nutri["cohortID"].astype(str).str[:6].astype(int)
nutri["type"] = nutri["type"].map(JP_TO_EN)
nutrition_all = (df[["family"]].merge(nutri[["family", "type"]], on="family", how="left")
                 ["type"].to_numpy(dtype=object))

print(f"feeding type joined: {np.isin(nutrition_all, NUTRITION_TYPES).sum()} / {len(df)}")
# %%
def pareto_scale(X, center=None, scale=None):
  """Pareto scaling: mean-center, divide by sqrt(SD)."""
  if center is None:
    center = X.mean(axis=0)
  if scale is None:
    scale = np.sqrt(X.std(axis=0, ddof=1))
    scale[scale == 0] = 1.0
  return (X - center) / scale, center, scale


def opls_fit(X, y, n_ortho):
  """OPLS with a single y (Trygg & Wold 2002). X, y must be centered/scaled."""
  Xres = X.copy()
  w = X.T @ y
  w /= np.linalg.norm(w)

  Wo, Po, To = [], [], []
  for _ in range(n_ortho):
    t = Xres @ w
    p = Xres.T @ t / (t @ t)
    wo = p - (w @ p) * w          # w is unit-norm
    wo /= np.linalg.norm(wo)
    to = Xres @ wo
    po = Xres.T @ to / (to @ to)
    Xres = Xres - np.outer(to, po)
    Wo.append(wo); Po.append(po); To.append(to)

  tp = Xres @ w
  pp = Xres.T @ tp / (tp @ tp)
  c = (y @ tp) / (tp @ tp)
  return {
    "w": w, "t_pred": tp, "p_pred": pp, "c": c,
    "W_ortho": np.array(Wo).T if Wo else np.zeros((X.shape[1], 0)),
    "P_ortho": np.array(Po).T if Po else np.zeros((X.shape[1], 0)),
    "T_ortho": np.array(To).T if To else np.zeros((X.shape[0], 0)),
    "X_pred": Xres,
  }


def opls_transform(model, Xnew):
  """Remove orthogonal variation from new samples and return predictive scores."""
  Xres = Xnew.copy()
  for i in range(model["W_ortho"].shape[1]):
    wo = model["W_ortho"][:, i]
    po = model["P_ortho"][:, i]
    to = Xres @ wo
    Xres = Xres - np.outer(to, po)
  return Xres @ model["w"]


def make_folds(n, n_splits=N_SPLITS, seed=SEED):
  idx = np.random.default_rng(seed).permutation(n)
  return idx, np.array_split(idx, n_splits)


def r2y_q2(X, y, n_ortho, folds):
  """R2Y on the full fit and Q2 from k-fold CV (scaling refit inside each fold)."""
  Xs, _, _ = pareto_scale(X)
  yc = y - y.mean()
  m = opls_fit(Xs, yc, n_ortho)
  tss = np.sum(yc ** 2)
  r2y = 1 - np.sum((yc - m["t_pred"] * m["c"]) ** 2) / tss

  y_hat = np.zeros_like(y)
  all_idx = np.arange(len(y))
  for fold in folds:
    tr = np.setdiff1d(all_idx, fold)
    Xtr, center, scale = pareto_scale(X[tr])
    ymean = y[tr].mean()
    mt = opls_fit(Xtr, y[tr] - ymean, n_ortho)
    y_hat[fold] = opls_transform(mt, (X[fold] - center) / scale) * mt["c"] + ymean
  q2 = 1 - np.sum((y - y_hat) ** 2) / tss
  return r2y, q2
# %%
def match_nutrition(restrict, seed=SEED):
  """k:1 frequency matching on feeding type.

  Every non-sec sample with a known feeding type is kept; within each type
  category k sec samples per non-sec sample are drawn at random, so the feeding
  type distribution is identical in the two classes by construction.
  """
  eligible = np.isin(nutrition_all, NUTRITION_TYPES) & restrict
  n_non = {t: np.sum(eligible & (nutrition_all == t) & (labels_all == "non-sec"))
           for t in NUTRITION_TYPES}
  n_sec = {t: np.sum(eligible & (nutrition_all == t) & (labels_all == "sec"))
           for t in NUTRITION_TYPES}
  k = int(min(n_sec[t] // n_non[t] for t in NUTRITION_TYPES if n_non[t] > 0))

  rng = np.random.default_rng(seed)
  mask = np.zeros(len(df), dtype=bool)
  for t in NUTRITION_TYPES:
    if n_non[t] == 0:
      continue   # category absent in non-sec cannot be matched
    non_idx = np.flatnonzero(eligible & (nutrition_all == t) & (labels_all == "non-sec"))
    sec_idx = np.flatnonzero(eligible & (nutrition_all == t) & (labels_all == "sec"))
    mask[non_idx] = True
    mask[rng.choice(sec_idx, size=k * len(non_idx), replace=False)] = True

  print(f"\n--- feeding type matching (k={k}, seed={seed}) ---")
  print(pd.crosstab(pd.Series(nutrition_all[mask], name="type"),
                    pd.Series(labels_all[mask], name="sec-type"), margins=True))
  print("proportion within class (%):")
  print((pd.crosstab(pd.Series(nutrition_all[mask], name="type"),
                     pd.Series(labels_all[mask], name="sec-type"),
                     normalize="columns") * 100).round(1))
  return mask
# %%
def gradient_line(x, y, c, cmap, vmax, linewidth=2.5, n_interp=24):
  """A polyline whose colour varies smoothly along its length.

  A LineCollection gives each segment one flat colour, which reads as a staircase
  on a 245-bin spectrum. Resampling the path inside every original segment and
  colouring each sub-segment by its midpoint makes the transition continuous.
  """
  t = np.arange(len(x))
  t_fine = np.linspace(0, len(x) - 1, (len(x) - 1) * n_interp + 1)
  xf, yf, cf = (np.interp(t_fine, t, v) for v in (x, y, c))

  points = np.array([xf, yf]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, vmax),
                      linewidth=linewidth, capstyle="round", joinstyle="round")
  lc.set_array((cf[:-1] + cf[1:]) / 2)
  return lc


def plot_s_line(p_ctr, p_corr, label, path, annotate):
  """S-line: p(ctr)[1] against chemical shift, coloured by |p(corr)[1]|."""
  order = np.argsort(-ppm)   # high -> low ppm
  x, y, c = ppm[order], p_ctr[order], np.abs(p_corr[order])

  lc = gradient_line(x, y, c, SLINE_CMAP, SLINE_VMAX)
  fig, ax = plt.subplots(figsize=(12, 5))
  ax.add_collection(lc)
  ax.set_xlim(x.max(), x.min())
  pad = (0.20 if annotate else 0.05) * (y.max() - y.min())
  ax.set_ylim(y.min() - pad, y.max() + pad)
  ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)

  title = f"OPLS-DA S-line — {label} (positive = higher in sec)"
  if annotate:
    top = np.argsort(-np.abs(p_corr))[:N_ANNOTATE]
    ax.scatter(ppm[top], p_ctr[top], s=14, facecolor="none",
               edgecolor="#c0392b", linewidth=1.0, zorder=3)
    # bins ranked by correlation cluster together in ppm, so stack labels that
    # would otherwise collide into rows instead of letting them overprint
    last_ppm, level = None, 0
    for i in sorted(top, key=lambda j: -ppm[j]):
      level = level + 1 if last_ppm is not None and last_ppm - ppm[i] < 0.5 else 0
      last_ppm = ppm[i]
      up = p_ctr[i] >= 0
      ax.annotate(f"{ppm[i]:.2f}", xy=(ppm[i], p_ctr[i]),
                  xytext=(0, (9 + 11 * level) * (1 if up else -1)),
                  textcoords="offset points", ha="center",
                  va="bottom" if up else "top",
                  fontsize=9, color="#c0392b", zorder=4)
    title += f"\nlabelled: top {N_ANNOTATE} bins by |p(corr)[1]|"

  ax.set_xlabel("chemical shift [ppm]")
  ax.set_ylabel("p(ctr)[1]")
  ax.set_title(title, fontsize=12)
  fig.colorbar(lc, ax=ax, label="|p(corr)[1]|")
  fig.tight_layout()
  fig.savefig(path)
  plt.show()


def run_subset(name, mask):
  """Full OPLS-DA workflow for one subset: model selection, plots, permutation test."""
  X, labels = X_all[mask], labels_all[mask]
  y = (labels == "sec").astype(float)   # sec = 1, non-sec = 0
  n = len(y)
  label, slug = SUBSET_LABELS[name]
  print(f"\n===== {name} [{label}]: "
        f"n={n} (sec={int(y.sum())}, non-sec={int(n - y.sum())}) =====")

  _, folds = make_folds(n)

  # --- component selection by CV Q2 -----------------------------------------
  scan = pd.DataFrame(
    [dict(zip(["n_ortho", "R2Y", "Q2"], (k, *r2y_q2(X, y, k, folds))))
     for k in range(1, MAX_ORTHO + 1)]
  )
  print(scan.round(3).to_string(index=False))
  n_ortho = int(scan.loc[scan["Q2"].idxmax(), "n_ortho"])
  r2y_obs, q2_obs = scan.loc[scan["n_ortho"] == n_ortho, ["R2Y", "Q2"]].iloc[0]
  print(f"selected n_ortho = {n_ortho}")

  # --- final model ----------------------------------------------------------
  Xs, _, x_scale = pareto_scale(X)
  yc = y - y.mean()
  model = opls_fit(Xs, yc, n_ortho)
  tp, to1 = model["t_pred"], model["T_ortho"][:, 0]
  ss_x = np.sum(Xs ** 2)
  r2x = (np.sum(np.outer(tp, model["p_pred"]) ** 2)
         + np.sum((model["T_ortho"] @ model["P_ortho"].T) ** 2)) / ss_x

  # --- permutation test -----------------------------------------------------
  rng = np.random.default_rng(SEED)
  perm = np.empty((N_PERM, 3))   # |corr(y_perm, y)|, R2Y, Q2
  for i in range(N_PERM):
    yp = rng.permutation(y)
    perm[i, 0] = abs(np.corrcoef(yp, y)[0, 1])
    perm[i, 1:] = r2y_q2(X, yp, n_ortho, folds)
  p_r2y = (1 + np.sum(perm[:, 1] >= r2y_obs)) / (1 + N_PERM)
  p_q2 = (1 + np.sum(perm[:, 2] >= q2_obs)) / (1 + N_PERM)
  print(f"R2Y={r2y_obs:.3f} (p={p_r2y:.3f})  Q2={q2_obs:.3f} (p={p_q2:.3f})")

  # --- score plot -----------------------------------------------------------
  fig, ax = plt.subplots(figsize=(7, 6))
  palette = {"sec": "#d62728", "non-sec": "#1f77b4"}
  for cls in ["sec", "non-sec"]:
    sel = labels == cls
    ax.scatter(tp[sel], to1[sel], s=40, alpha=0.7,
               color=palette[cls], edgecolor="white", linewidth=0.5,
               label=f"{cls} (n={sel.sum()})")
  f_crit = f_dist.ppf(0.95, 2, n - 2)
  radius = np.sqrt(2 * (n - 1) / (n - 2) * f_crit)
  ax.add_patch(Ellipse(
    (0, 0), 2 * radius * tp.std(ddof=1), 2 * radius * to1.std(ddof=1),
    fill=False, edgecolor="gray", linestyle="--", linewidth=1.2,
  ))
  ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.set_xlabel("$t_{pred}$ [1]")
  ax.set_ylabel("$t_{ortho}$ [1]")
  ax.set_title(
    f"OPLS-DA score plot — {label}\n"
    f"1+{n_ortho}, pareto | R2X={r2x:.3f}  R2Y={r2y_obs:.3f}  "
    f"Q2={q2_obs:.3f}  $p_{{Q2}}$={p_q2:.3f}",
    fontsize=11,
  )
  ax.legend(frameon=False, fontsize=12)
  fig.tight_layout()
  fig.savefig(os.path.join(figdir, f"score_plot_{slug}.svg"))
  plt.show()

  # --- S-line ---------------------------------------------------------------
  # SIMCA notation. p(ctr)[1] is the predictive loading expressed in centred
  # (unscaled) units: the model loading lives in scaled space, so multiplying by
  # the column scaling factor back-transforms it, which is what "ctr" denotes.
  # p(corr)[1] is the score-variable correlation, invariant to that scaling.
  p_ctr = model["p_pred"] * x_scale
  p_corr = ((Xs.T @ tp) / (n - 1)) / (Xs.std(axis=0, ddof=1) * tp.std(ddof=1))
  plot_s_line(p_ctr, p_corr, label, os.path.join(figdir, f"s_line_{slug}.svg"),
              annotate=False)
  plot_s_line(p_ctr, p_corr, label,
              os.path.join(figdir, f"s_line_{slug}_annotated.svg"), annotate=True)

  # --- permutation plot -----------------------------------------------------
  fig, axes = plt.subplots(1, 2, figsize=(13, 5))
  ax = axes[0]
  ax.scatter(perm[:, 0], perm[:, 1], s=12, alpha=0.4, color="#1f77b4", label="R2Y (perm)")
  ax.scatter(perm[:, 0], perm[:, 2], s=12, alpha=0.4, color="#2ca02c", label="Q2 (perm)")
  ax.scatter([1], [r2y_obs], s=90, color="#1f77b4", edgecolor="black", zorder=5)
  ax.scatter([1], [q2_obs], s=90, color="#2ca02c", edgecolor="black", zorder=5)
  intercepts = {}
  for j, key, obs, color in [(1, "R2Y", r2y_obs, "#1f77b4"), (2, "Q2", q2_obs, "#2ca02c")]:
    slope, intercept = np.polyfit(np.append(perm[:, 0], 1), np.append(perm[:, j], obs), 1)
    intercepts[key] = intercept
    xs = np.array([0, 1])
    ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.2, linestyle="--")
  ax.text(0.03, 0.03,
          f"intercepts: R2Y={intercepts['R2Y']:.3f}, Q2={intercepts['Q2']:.3f}",
          transform=ax.transAxes, fontsize=11)
  ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.set_xlabel("|corr(permuted y, y)|")
  ax.set_ylabel("R2Y / Q2")
  ax.set_title(f"Permutation plot — {label}\n({N_PERM} permutations)", fontsize=11)
  ax.legend(frameon=False, fontsize=11)

  ax = axes[1]
  ax.hist(perm[:, 2], bins=40, color="#2ca02c", alpha=0.6, label="Q2 null")
  ax.axvline(q2_obs, color="black", linewidth=2,
             label=f"observed Q2 = {q2_obs:.3f}\np = {p_q2:.3f}")
  ax.set_xlabel("Q2")
  ax.set_ylabel("count")
  ax.set_title(f"Q2 null distribution\n{label}", fontsize=11)
  ax.legend(frameon=False, fontsize=11)
  fig.tight_layout()
  fig.savefig(os.path.join(figdir, f"permutation_{slug}.svg"))
  plt.show()

  # --- loadings table -------------------------------------------------------
  df_loading = pd.DataFrame({"ppm": ppm, "p_ctr": p_ctr, "p_corr": p_corr})
  df_loading = df_loading.reindex(
    df_loading["p_ctr"].abs().sort_values(ascending=False).index)
  df_loading.to_csv(os.path.join(heredir, f"feces_opls_sline_{slug}.csv"), index=False)
  print(df_loading.head(10).round(4).to_string(index=False))

  return {"subset": name, "label": label, "n": n, "n_sec": int(y.sum()), "n_nonsec": int(n - y.sum()),
          "n_ortho": n_ortho, "R2X": r2x, "R2Y": r2y_obs, "p(R2Y)": p_r2y,
          "Q2": q2_obs, "p(Q2)": p_q2,
          "int(R2Y)": intercepts["R2Y"], "int(Q2)": intercepts["Q2"]}
# %%
# no feeding-type adjustment
subsets = [(f"timing{t}", timing_all == t) for t in [3, 4, 5]]
# feeding-type adjusted (timing3 only: the other timings have too few non-sec
# samples to match without collapsing the model down to n < 50)
subsets += [("matched_timing3", match_nutrition(timing_all == 3))]

results = pd.DataFrame([run_subset(name, mask) for name, mask in subsets])
# %%
print("\n===== summary =====")
print(results.round(3).to_string(index=False))
results.to_csv(os.path.join(heredir, "feces_opls_summary.csv"), index=False)
