# %%
"""Does adjusting for infant feeding type change the sec/non-sec OPLS-DA result?

The main analysis (src/opls/feces.py) reports one unadjusted model (Newborn,
n=106) and one feeding-type matched model (n=56). Those two differ in three ways
at once -- matching, sample size, and which samples carry feeding-type data -- so
their similar Q2 does not by itself show that the adjustment is inconsequential.

This script isolates the matching effect:

  full     n=106  every timing3 sample, unadjusted
   +- joined   n=80  only samples with a feeding type, unadjusted   <- population
      +- matched  n=56  k:1 frequency matched on feeding type       <- matching
      +- random   n=56  same size, drawn ignoring feeding type      <- sample size

matched vs random is the decisive comparison: same population, same n, the only
difference is whether the feeding type distribution was equalised. Both are
random draws, so they are repeated N_DRAWS times and compared as distributions.

The OPLS core is duplicated from src/opls/feces.py so that each analysis
directory stays a standalone script, as elsewhere in this repo.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr

sns.set_theme(style="white")
sns.set_context("talk", font_scale=0.9)

N_SPLITS = 7
N_ORTHO = 3        # fixed across every model here so the comparison is like-for-like
N_DRAWS = 200      # random draws of the matched and the size-matched random sets
N_TOP = 10         # top bins compared between loadings
SEED = 0

JP_TO_EN = {"母乳": "breastMilk", "人工": "formula", "混合": "both"}
NUTRITION_TYPES = ["both", "breastMilk", "formula"]
TIMING = 3         # "Newborn", the timing the main analysis matched on
# %%
heredir = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.dirname(os.path.dirname(heredir))
datadir = os.path.join(basedir, "data")
figdir = os.path.join(heredir, "fig")
os.makedirs(figdir, exist_ok=True)
# %%
def read_csv_jp(path):
  for enc in ["utf-8", "cp932"]:
    try:
      return pd.read_csv(path, header=0, encoding=enc)
    except UnicodeDecodeError:
      continue
  raise UnicodeDecodeError(f"cannot decode {path} as utf-8 or cp932")


df = pd.read_csv(os.path.join(datadir, "feces_BinningResults.csv"), header=0)
df = df[(df["age"] == "c") & (df["timing"] == TIMING)].reset_index(drop=True)

nutri = read_csv_jp(os.path.join(datadir, "formulaOrBreastMilk_development.csv"))
nutri["family"] = nutri["cohortID"].astype(str).str[:6].astype(int)
nutri["type"] = nutri["type"].map(JP_TO_EN)

ppm = df.columns[6:].astype(float).to_numpy()
X_all = df.iloc[:, 6:].to_numpy(dtype=float)
labels_all = df["sec-type"].to_numpy()
y_all = (labels_all == "sec").astype(float)
nutrition_all = (df[["family"]].merge(nutri[["family", "type"]], on="family", how="left")
                 ["type"].to_numpy(dtype=object))
has_type = np.isin(nutrition_all, NUTRITION_TYPES)

print(f"timing{TIMING}: n={len(df)} "
      f"(sec={int(y_all.sum())}, non-sec={int(len(df) - y_all.sum())})")
print(f"with a usable feeding type: {has_type.sum()}")
# %%
# ---- OPLS core (duplicated from src/opls/feces.py) --------------------------
def pareto_scale(X, center=None, scale=None):
  if center is None:
    center = X.mean(axis=0)
  if scale is None:
    scale = np.sqrt(X.std(axis=0, ddof=1))
    scale[scale == 0] = 1.0
  return (X - center) / scale, center, scale


def opls_fit(X, y, n_ortho):
  Xres = X.copy()
  w = X.T @ y
  w /= np.linalg.norm(w)

  Wo, Po, To = [], [], []
  for _ in range(n_ortho):
    t = Xres @ w
    p = Xres.T @ t / (t @ t)
    wo = p - (w @ p) * w
    wo /= np.linalg.norm(wo)
    to = Xres @ wo
    po = Xres.T @ to / (to @ to)
    Xres = Xres - np.outer(to, po)
    Wo.append(wo); Po.append(po); To.append(to)

  tp = Xres @ w
  pp = Xres.T @ tp / (tp @ tp)
  c = (y @ tp) / (tp @ tp)
  return {"w": w, "t_pred": tp, "p_pred": pp, "c": c,
          "W_ortho": np.array(Wo).T, "P_ortho": np.array(Po).T,
          "T_ortho": np.array(To).T}


def opls_transform(model, Xnew):
  Xres = Xnew.copy()
  for i in range(model["W_ortho"].shape[1]):
    to = Xres @ model["W_ortho"][:, i]
    Xres = Xres - np.outer(to, model["P_ortho"][:, i])
  return Xres @ model["w"]


def make_folds(n, n_splits=N_SPLITS, seed=SEED):
  idx = np.random.default_rng(seed).permutation(n)
  return np.array_split(idx, n_splits)


def r2y_q2(X, y, n_ortho=N_ORTHO):
  folds = make_folds(len(y))
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
  return r2y, 1 - np.sum((y - y_hat) ** 2) / tss


def loadings(X, y, n_ortho=N_ORTHO):
  """p(ctr)[1] and p(corr)[1], in SIMCA notation."""
  Xs, _, x_scale = pareto_scale(X)
  m = opls_fit(Xs, y - y.mean(), n_ortho)
  tp = m["t_pred"]
  p_ctr = m["p_pred"] * x_scale
  p_corr = ((Xs.T @ tp) / (len(y) - 1)) / (Xs.std(axis=0, ddof=1) * tp.std(ddof=1))
  return p_ctr, p_corr, tp
# %%
# ---- (c) is feeding type associated with sec-type at all? -------------------
ct = pd.crosstab(pd.Series(nutrition_all[has_type], name="type"),
                 pd.Series(labels_all[has_type], name="sec-type"))
chi2, p_chi2, dof, _ = chi2_contingency(ct)
print("\n===== (c) feeding type x sec-type =====")
print(ct)
print((ct["non-sec"] / ct.sum(axis=1) * 100).round(1).rename("non-sec %"))
print(f"chi2 = {chi2:.3f}, df = {dof}, p = {p_chi2:.3f}")
print("A confounder must be associated with both exposure and outcome; "
      f"{'no' if p_chi2 >= 0.05 else 'AN'} association with sec-type is seen here.")
# %%
# ---- the four rungs of the ladder ------------------------------------------
def matched_mask(seed):
  """k:1 frequency matching on feeding type."""
  n_non = {t: np.sum(has_type & (nutrition_all == t) & (labels_all == "non-sec"))
           for t in NUTRITION_TYPES}
  n_sec = {t: np.sum(has_type & (nutrition_all == t) & (labels_all == "sec"))
           for t in NUTRITION_TYPES}
  k = int(min(n_sec[t] // n_non[t] for t in NUTRITION_TYPES if n_non[t] > 0))

  rng = np.random.default_rng(seed)
  mask = np.zeros(len(df), dtype=bool)
  for t in NUTRITION_TYPES:
    if n_non[t] == 0:
      continue
    non_idx = np.flatnonzero(has_type & (nutrition_all == t) & (labels_all == "non-sec"))
    sec_idx = np.flatnonzero(has_type & (nutrition_all == t) & (labels_all == "sec"))
    mask[non_idx] = True
    mask[rng.choice(sec_idx, size=k * len(non_idx), replace=False)] = True
  return mask, k


def random_mask(seed, n_sec, n_non):
  """Same population and class sizes as the matched set, feeding type ignored."""
  rng = np.random.default_rng(seed)
  mask = np.zeros(len(df), dtype=bool)
  for cls, size in [("sec", n_sec), ("non-sec", n_non)]:
    idx = np.flatnonzero(has_type & (labels_all == cls))
    mask[rng.choice(idx, size=size, replace=False)] = True
  return mask


m0, k_match = matched_mask(SEED)
n_sec_m, n_non_m = int(y_all[m0].sum()), int((1 - y_all[m0]).sum())
print(f"\nmatching: k={k_match} -> sec {n_sec_m} / non-sec {n_non_m}")

ladder = []
for rung, mask in [("full", np.ones(len(df), dtype=bool)),
                   ("joined", has_type),
                   ("matched", m0),
                   ("random", random_mask(SEED, n_sec_m, n_non_m))]:
  r2y, q2 = r2y_q2(X_all[mask], y_all[mask])
  ladder.append({"set": rung, "n": int(mask.sum()),
                 "n_sec": int(y_all[mask].sum()),
                 "n_nonsec": int((1 - y_all[mask]).sum()),
                 "R2Y": r2y, "Q2": q2})
ladder = pd.DataFrame(ladder)
print("\n===== (a) one change at a time (single draw, seed=0, 1+3) =====")
print(ladder.round(3).to_string(index=False))
# %%
# ---- (b) repeat the draw: matched vs random as distributions ----------------
# The reference loadings the draws are compared against: the unadjusted model
# fitted on the same population (joined), so only the draw differs.
p_ctr_ref, p_corr_ref, _ = loadings(X_all[has_type], y_all[has_type])
top_ref = set(np.argsort(-np.abs(p_corr_ref))[:N_TOP])

draws = []
for seed in range(N_DRAWS):
  for kind, mask in [("matched", matched_mask(seed)[0]),
                     ("random", random_mask(seed, n_sec_m, n_non_m))]:
    r2y, q2 = r2y_q2(X_all[mask], y_all[mask])
    p_ctr, p_corr, _ = loadings(X_all[mask], y_all[mask])
    draws.append({
      "seed": seed, "draw": kind, "R2Y": r2y, "Q2": q2,
      # (d) does the draw reproduce the unadjusted model's loadings?
      "r_pcorr": spearmanr(p_corr, p_corr_ref).statistic,
      "top_overlap": len(top_ref & set(np.argsort(-np.abs(p_corr))[:N_TOP])),
    })
draws = pd.DataFrame(draws)
draws.to_csv(os.path.join(heredir, "confounding_draws.csv"), index=False)

print(f"\n===== (b) {N_DRAWS} draws each =====")
print(draws.groupby("draw")[["R2Y", "Q2", "r_pcorr", "top_overlap"]]
      .agg(["mean", "std"]).round(3))

stats = []
for metric in ["Q2", "R2Y", "r_pcorr", "top_overlap"]:
  a = draws.loc[draws["draw"] == "matched", metric].to_numpy()
  b = draws.loc[draws["draw"] == "random", metric].to_numpy()
  pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
  stats.append({"metric": metric, "matched": a.mean(), "random": b.mean(),
                "difference": a.mean() - b.mean(), "pooled_SD": pooled,
                "cohen_d": (a.mean() - b.mean()) / pooled if pooled else np.nan,
                "mannwhitney_p": mannwhitneyu(a, b).pvalue})
stats = pd.DataFrame(stats)
print(stats.round(3).to_string(index=False))
print("\nNote: the draws resample the same samples, so N_DRAWS is chosen, not "
      "observed -- read cohen_d (difference relative to draw-to-draw spread), "
      "not the p value, which shrinks as N_DRAWS grows.")
# %%
# ---- (e) does the predictive score track feeding type? ----------------------
_, _, tp_joined = loadings(X_all[has_type], y_all[has_type])
nut_joined = nutrition_all[has_type]
groups = [tp_joined[nut_joined == t] for t in NUTRITION_TYPES]
h_stat, p_kruskal = kruskal(*groups)
print("\n===== (e) t_pred by feeding type (joined model) =====")
for t, g in zip(NUTRITION_TYPES, groups):
  print(f"  {t:<11} n={len(g):>3}  median t_pred = {np.median(g):+.3f}")
print(f"Kruskal-Wallis H = {h_stat:.3f}, p = {p_kruskal:.3f} "
      f"({'no' if p_kruskal >= 0.05 else 'A'} shift of t_pred by feeding type)")
# %%
# ---- figures ----------------------------------------------------------------
palette = {"matched": "#d62728", "random": "#7f7f7f"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
sns.histplot(data=draws, x="Q2", hue="draw", bins=30, alpha=0.5,
             palette=palette, ax=ax)
for rung, style in [("full", ":"), ("joined", "--")]:
  q = ladder.loc[ladder["set"] == rung, "Q2"].iloc[0]
  ax.axvline(q, color="black", linestyle=style, linewidth=1.4,
             label=f"{rung} (n={ladder.loc[ladder['set'] == rung, 'n'].iloc[0]}): {q:.3f}")
ax.legend(frameon=False, fontsize=10)
ax.set_title(f"Q2 over {N_DRAWS} draws (n={n_sec_m + n_non_m}, 1+{N_ORTHO})",
             fontsize=12)

ax = axes[1]
sns.histplot(data=draws, x="r_pcorr", hue="draw", bins=30, alpha=0.5,
             palette=palette, ax=ax)
ax.set_xlabel("Spearman r of p(corr)[1] vs the unadjusted model")
ax.set_title("Do the draws reproduce the unadjusted loadings?", fontsize=12)
fig.suptitle("Feeding-type matching vs size-matched random selection", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "matched_vs_random.svg"))
plt.show()
# %%
p_ctr_m, p_corr_m, _ = loadings(X_all[m0], y_all[m0])
lim = np.abs(np.concatenate([p_corr_ref, p_corr_m])).max() * 1.05

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.axhline(0, color="lightgray", linewidth=0.8)
ax.axvline(0, color="lightgray", linewidth=0.8)
ax.plot([-lim, lim], [-lim, lim], color="lightgray", linestyle="--", linewidth=1)
ax.scatter(p_corr_ref, p_corr_m, s=22, alpha=0.6, color="#2c7fb8",
           edgecolor="white", linewidth=0.4)
rho = spearmanr(p_corr_ref, p_corr_m).statistic
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("p(corr)[1] — unadjusted (joined)")
ax.set_ylabel("p(corr)[1] — feeding-type matched")
ax.set_title(f"Loadings, adjusted vs unadjusted\nSpearman r = {rho:.3f} "
             f"(245 bins), top-{N_TOP} overlap = "
             f"{len(top_ref & set(np.argsort(-np.abs(p_corr_m))[:N_TOP]))}/{N_TOP}",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "loadings_adjusted_vs_unadjusted.svg"))
plt.show()
# %%
fig, ax = plt.subplots(figsize=(7, 6))
jitter_rng = np.random.default_rng(SEED)
markers = {"sec": "o", "non-sec": "^"}
nut_palette = dict(zip(NUTRITION_TYPES, sns.color_palette("Set2", 3)))
for cls in ["sec", "non-sec"]:
  for t in NUTRITION_TYPES:
    sel = (labels_all[has_type] == cls) & (nut_joined == t)
    if sel.sum() == 0:
      continue
    ax.scatter(tp_joined[sel],
               NUTRITION_TYPES.index(t) + jitter_rng.normal(0, 0.06, sel.sum()),
               s=45, alpha=0.8, marker=markers[cls],
               color=nut_palette[t], edgecolor="black", linewidth=0.4,
               label=f"{t} / {cls}")
ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
ax.set_yticks(range(len(NUTRITION_TYPES)), NUTRITION_TYPES)
ax.set_xlabel("$t_{pred}$ [1]")
ax.set_title(f"Predictive score by feeding type (joined, n={has_type.sum()})\n"
             f"Kruskal-Wallis p = {p_kruskal:.3f}", fontsize=12)
ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
fig.tight_layout()
fig.savefig(os.path.join(figdir, "tpred_by_feeding_type.svg"))
plt.show()
# %%
summary = {
  "timing": TIMING, "n_full": len(df), "n_joined": int(has_type.sum()),
  "n_matched": n_sec_m + n_non_m, "k_match": k_match, "n_ortho": N_ORTHO,
  "n_draws": N_DRAWS,
  "chi2_feeding_vs_sec": chi2, "p_chi2": p_chi2,
  "kruskal_tpred_by_feeding": h_stat, "p_kruskal": p_kruskal,
  **{f"Q2_{r['set']}": r["Q2"] for _, r in ladder.iterrows()},
  **{f"{s['metric']}_diff_matched_minus_random": s["difference"]
     for _, s in stats.iterrows()},
  **{f"{s['metric']}_cohen_d": s["cohen_d"] for _, s in stats.iterrows()},
}
ladder.to_csv(os.path.join(heredir, "confounding_ladder.csv"), index=False)
stats.to_csv(os.path.join(heredir, "confounding_matched_vs_random.csv"), index=False)
pd.Series(summary).to_csv(os.path.join(heredir, "confounding_summary.csv"),
                          header=False)
print("\n===== summary =====")
print(pd.Series(summary).to_string())
