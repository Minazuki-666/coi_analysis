# %%
"""各時期で sec / non-sec 間に差のあった Genus の存在量 (Y) を、乳児便の NMR bin (X) で説明する
OPLS (回帰) を行い、S-line プロットを描く。

- Y: log10(相対存在量 + 擬似カウント)。擬似カウントは、その時期での非ゼロ最小値の 1/2
- X: feces_BinningResults.csv の乳児便 (age == "c") 245 bin、Pareto スケーリング
- 直交成分数 0-3 を 7-fold CV の Q2 で選ぶ。パーミュテーション検定でも毎回同じ選択をやり直し、
  Q2 の帰無分布を作る (成分数選択の楽観バイアスを含めた検定にするため)
- 12 モデルの p(Q2) に BH-FDR
- 感度分析: sec のみで同じモデル。対象の Genus は分泌型で差があったものなので、
  全サンプルでの結果が分泌型の違いを拾っているだけでないかを確認する
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from scipy.stats import f as f_dist
import seaborn as sns
from statsmodels.stats.multitest import multipletests

sns.set_theme(style="white")
sns.set_context("talk", font_scale=0.9)

N_PERM = 1000
N_SPLITS = 7
MAX_ORTHO = 3
SEED = 0
SLINE_CMAP = sns.color_palette("mako_r", as_cmap=True)
SLINE_VMAX = 1.0  # S-line の色は |p(corr)| なので、全範囲で固定してモデル間で比較できるようにする
N_ANNOTATE = 10   # S-line でラベルを付ける bin の数 (|p(corr)[1]| の上位)

TIMING_LABEL = {3: "Newborn", 4: "1 mo.", 5: "4-5 mos."}
TARGETS = [
  (3, "f__Enterobacteriaceae_A|__ (unassigned)"),
  (3, "g__Lactococcus_A_346120"),
  (3, "g__Phocaeicola_A_858004"),
  (4, "g__Streptococcus"),
  (4, "g__Klebsiella_724518"),
  (4, "f__Enterococcaceae|__ (unassigned)"),
  (5, "f__Enterobacteriaceae_A|__ (unassigned)"),
  (5, "g__Collinsella"),
  (5, "g__Sellimonas"),
  (5, "g__Lactococcus_A_346120"),
  (5, "g__Clostridium_T"),
  (5, "g__Lactobacillus"),
]
# %%
heredir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(os.path.dirname(heredir)), "data")
figdir = os.path.join(heredir, "fig")
os.makedirs(figdir, exist_ok=True)

feces = pd.read_csv(os.path.join(datadir, "feces_BinningResults.csv"), header=0)
feces = feces[feces["age"] == "c"].reset_index(drop=True)
bin_cols = list(feces.columns[6:])
ppm = np.array(bin_cols, dtype=float)
feces["ID"] = feces["id"].astype(str).str[:11]  # Genus csv の ID は feces id の上11桁

genus = pd.read_csv(os.path.join(datadir, "20260901_Genus_RelativeAb.csv"))
genus = genus.dropna(subset=["ID", "timing", "sec-type"]).copy()  # 末尾の空行を除く
taxa = list(genus.columns[6:])


# Genus名。属が表記されていない列は、同定できた最下位の分類群から後ろをそのまま残す (gut/genus_sectype.py と同じ)
def taxon_name(lineage):
  ranks = lineage.split("|")
  unnamed = [r.endswith("__") for r in ranks]
  if not unnamed[-1]:
    return ranks[-1]
  last = max(i for i, u in enumerate(unnamed) if not u)
  return "|".join(ranks[last:]) + " (unassigned)"


genus = genus.rename(columns={c: taxon_name(c) for c in taxa})
names = [g for _, g in TARGETS]
genus[list(set(names))] = genus[list(set(names))].astype(float)

# timing と sec-type は両ファイルで一致することを確認済みなので feces 側を使う
df = feces.merge(genus[["ID"] + list(set(names))], on="ID", how="inner")
print(f"matched samples: {len(df)}")
print(pd.crosstab(df["timing"].map(TIMING_LABEL), df["sec-type"]))
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
  }


def opls_transform(model, Xnew):
  """Remove orthogonal variation from new samples and return predictive scores."""
  Xres = Xnew.copy()
  for i in range(model["W_ortho"].shape[1]):
    to = Xres @ model["W_ortho"][:, i]
    Xres = Xres - np.outer(to, model["P_ortho"][:, i])
  return Xres @ model["w"]


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
    ymean = y[tr].mean()
    if np.ptp(y[tr]) == 0:   # 学習側で Genus が一度も検出されない fold は平均で予測
      y_hat[fold] = ymean
      continue
    Xtr, center, scale = pareto_scale(X[tr])
    mt = opls_fit(Xtr, y[tr] - ymean, n_ortho)
    y_hat[fold] = opls_transform(mt, (X[fold] - center) / scale) * mt["c"] + ymean
  q2 = 1 - np.sum((y - y_hat) ** 2) / tss
  return r2y, q2


def select_model(X, y, folds):
  """直交成分数 0..MAX_ORTHO から Q2 最大のものを選ぶ"""
  scan = [(k, *r2y_q2(X, y, k, folds)) for k in range(MAX_ORTHO + 1)]
  return max(scan, key=lambda s: s[2])   # (n_ortho, R2Y, Q2)
# %%
def gradient_line(x, y, c, cmap, vmax, linewidth=2.5, n_interp=24):
  """色が連続的に変わる折れ線 (src/opls/feces.py と同じ)"""
  t = np.arange(len(x))
  t_fine = np.linspace(0, len(x) - 1, (len(x) - 1) * n_interp + 1)
  xf, yf, cf = (np.interp(t_fine, t, v) for v in (x, y, c))
  points = np.array([xf, yf]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, vmax),
                      linewidth=linewidth, capstyle="round", joinstyle="round")
  lc.set_array((cf[:-1] + cf[1:]) / 2)
  return lc


def plot_s_line(p_ctr, p_corr, title, path):
  """S-line: p(ctr)[1] を化学シフトに対して描き、|p(corr)[1]| で色付けする"""
  order = np.argsort(-ppm)
  x, y, c = ppm[order], p_ctr[order], np.abs(p_corr[order])
  lc = gradient_line(x, y, c, SLINE_CMAP, SLINE_VMAX)
  fig, ax = plt.subplots(figsize=(12, 5))
  ax.add_collection(lc)
  ax.set_xlim(x.max(), x.min())
  pad = 0.20 * (y.max() - y.min())
  ax.set_ylim(y.min() - pad, y.max() + pad)
  ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)

  top = np.argsort(-np.abs(p_corr))[:N_ANNOTATE]
  ax.scatter(ppm[top], p_ctr[top], s=14, facecolor="none",
             edgecolor="#c0392b", linewidth=1.0, zorder=3)
  last_ppm, level = None, 0
  for i in sorted(top, key=lambda j: -ppm[j]):
    level = level + 1 if last_ppm is not None and last_ppm - ppm[i] < 0.5 else 0
    last_ppm = ppm[i]
    up = p_ctr[i] >= 0
    ax.annotate(f"{ppm[i]:.2f}", xy=(ppm[i], p_ctr[i]),
                xytext=(0, (9 + 11 * level) * (1 if up else -1)),
                textcoords="offset points", ha="center",
                va="bottom" if up else "top", fontsize=9, color="#c0392b", zorder=4)

  ax.set_xlabel("chemical shift [ppm]")
  ax.set_ylabel("p(ctr)[1]")
  ax.set_title(title, fontsize=12)
  fig.colorbar(lc, ax=ax, label="|p(corr)[1]|")
  fig.tight_layout()
  fig.savefig(path)
  plt.show()


def slug(s):
  return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")
# %%
def plot_score(ex, r, path):
  """Score plot: t_pred[1] vs t_ortho[1]。直交成分が0のモデルは縦軸を観測値 y (log10存在量) にする。
  色は y、記号は分泌型 (丸 = sec, 三角 = non-sec)。"""
  tp, to1, y, sec = ex["t_pred"], ex["t_ortho"], ex["y"], ex["sec"]
  fig, ax = plt.subplots(figsize=(7.5, 6))
  vmin, vmax = y.min(), y.max()
  for cls, marker in [("sec", "o"), ("non-sec", "^")]:
    sel = sec == cls
    if not sel.any():
      continue
    sc = ax.scatter(tp[sel], (to1 if to1 is not None else y)[sel], c=y[sel], cmap="viridis",
                    vmin=vmin, vmax=vmax, marker=marker, s=50, alpha=0.8,
                    edgecolor="white", linewidth=0.5, label=f"{cls} (n={sel.sum()})")
  if to1 is not None:   # Hotelling T2 95% 楕円
    n = len(y)
    radius = np.sqrt(2 * (n - 1) / (n - 2) * f_dist.ppf(0.95, 2, n - 2))
    ax.add_patch(Ellipse((0, 0), 2 * radius * tp.std(ddof=1), 2 * radius * to1.std(ddof=1),
                         fill=False, edgecolor="gray", linestyle="--", linewidth=1.2))
    ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.set_xlabel("$t_{pred}$ [1]")
  ax.set_ylabel("$t_{ortho}$ [1]" if to1 is not None else "observed y (log10 abundance)")
  ax.set_title(f"OPLS score plot — {r['timing']} {r['genus']} ({r['subset']})\n"
               f"1+{r['n_ortho']} | R2Y={r['R2Y']:.3f}  Q2={r['Q2']:.3f}  "
               f"$p_{{Q2}}$={r['p(Q2)']:.3f}", fontsize=11)
  ax.legend(frameon=False, fontsize=11)
  fig.colorbar(sc, ax=ax, label="y = log10(abundance + pseudocount)")
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)


def plot_permutation(ex, r, path):
  """Permutation plot (左: R2Y/Q2 と y 並べ替え相関の関係, 右: Q2 の帰無分布)"""
  perm = ex["perm"]
  fig, axes = plt.subplots(1, 2, figsize=(13, 5))
  ax = axes[0]
  ax.scatter(perm[:, 0], perm[:, 1], s=12, alpha=0.4, color="#1f77b4", label="R2Y (perm)")
  ax.scatter(perm[:, 0], perm[:, 2], s=12, alpha=0.4, color="#2ca02c", label="Q2 (perm)")
  ax.scatter([1], [r["R2Y"]], s=90, color="#1f77b4", edgecolor="black", zorder=5)
  ax.scatter([1], [r["Q2"]], s=90, color="#2ca02c", edgecolor="black", zorder=5)
  for j, obs, color in [(1, r["R2Y"], "#1f77b4"), (2, r["Q2"], "#2ca02c")]:
    slope, intercept = np.polyfit(np.append(perm[:, 0], 1), np.append(perm[:, j], obs), 1)
    ax.plot([0, 1], [intercept, slope + intercept], color=color, linewidth=1.2, linestyle="--")
  ax.text(0.03, 0.03, f"intercepts: R2Y={r['int(R2Y)']:.3f}, Q2={r['int(Q2)']:.3f}",
          transform=ax.transAxes, fontsize=11)
  ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.set_xlabel("|corr(permuted y, y)|")
  ax.set_ylabel("R2Y / Q2")
  ax.set_title(f"Permutation plot — {r['timing']} {r['genus']} ({r['subset']})\n"
               f"({N_PERM} permutations, n_ortho re-selected each time)", fontsize=11)
  ax.legend(frameon=False, fontsize=11)

  ax = axes[1]
  ax.hist(perm[:, 2], bins=40, color="#2ca02c", alpha=0.6, label="Q2 null")
  ax.axvline(r["Q2"], color="black", linewidth=2,
             label=f"observed Q2 = {r['Q2']:.3f}\np = {r['p(Q2)']:.3f}")
  ax.set_xlabel("Q2")
  ax.set_ylabel("count")
  ax.set_title("Q2 null distribution", fontsize=11)
  ax.legend(frameon=False, fontsize=11)
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)
# %%
def run_model(t, g, sec_only=False):
  d = df[df["timing"] == t]
  if sec_only:
    d = d[d["sec-type"] == "sec"]
  X = d[bin_cols].to_numpy(dtype=float)
  ab = d[g].to_numpy(dtype=float)
  nonzero = ab[ab > 0]
  row = {"timing": TIMING_LABEL[t], "genus": g, "subset": "sec only" if sec_only else "all",
         "n": len(d), "n_sec": int((d["sec-type"] == "sec").sum()),
         "n_detected": len(nonzero), "prevalence": len(nonzero) / len(d)}
  if len(nonzero) < 3:
    print(f"skip {row['timing']} {g} ({row['subset']}): detected in {len(nonzero)} samples")
    return row, None, None
  y = np.log10(ab + nonzero.min() / 2)

  folds = np.array_split(np.random.default_rng(SEED).permutation(len(y)), N_SPLITS)
  n_ortho, r2y, q2 = select_model(X, y, folds)

  rng = np.random.default_rng(SEED)
  perm = np.empty((N_PERM, 3))   # |corr(y_perm, y)|, R2Y, Q2
  for i in range(N_PERM):
    yp = rng.permutation(y)
    perm[i] = [abs(np.corrcoef(yp, y)[0, 1]), *select_model(X, yp, folds)[1:]]
  p_r2y = (1 + np.sum(perm[:, 1] >= r2y)) / (1 + N_PERM)
  p_q2 = (1 + np.sum(perm[:, 2] >= q2)) / (1 + N_PERM)
  # 永続性プロットの回帰直線 (x=1 の観測値を含めて引く) の切片。Q2 の切片が負なら過適合の疑いは小さい
  intercepts = {k: np.polyfit(np.append(perm[:, 0], 1), np.append(perm[:, j], obs), 1)[1]
                for k, j, obs in [("R2Y", 1, r2y), ("Q2", 2, q2)]}

  Xs, _, x_scale = pareto_scale(X)
  model = opls_fit(Xs, y - y.mean(), n_ortho)
  tp = model["t_pred"]
  # SIMCA 表記: p(ctr) は予測ローディングを中心化のみの単位に戻したもの、p(corr) はスコアと変数の相関
  p_ctr = model["p_pred"] * x_scale
  p_corr = ((Xs.T @ tp) / (len(y) - 1)) / (Xs.std(axis=0, ddof=1) * tp.std(ddof=1))
  # 予測ローディングの向きを「Genus が多いほど正」にそろえる
  sign = np.sign(np.corrcoef(tp, y)[0, 1])
  p_ctr, p_corr = sign * p_ctr, sign * p_corr

  row.update({"n_ortho": n_ortho, "R2Y": r2y, "p(R2Y)": p_r2y, "Q2": q2, "p(Q2)": p_q2,
              "int(R2Y)": intercepts["R2Y"], "int(Q2)": intercepts["Q2"]})
  loading = pd.DataFrame({"timing": row["timing"], "genus": g, "subset": row["subset"],
                          "ppm": ppm, "p_ctr": p_ctr, "p_corr": p_corr})
  extra = {"t_pred": sign * tp, "t_ortho": model["T_ortho"][:, 0] if n_ortho >= 1 else None,
           "y": y, "sec": d["sec-type"].to_numpy(), "perm": perm}
  return row, loading, extra
# %%
rows, loadings, extras = [], [], {}
for t, g in TARGETS:
  for sec_only in [False, True]:
    row, loading, extra = run_model(t, g, sec_only)
    rows.append(row)
    extras[(row["timing"], g, row["subset"])] = extra
    if loading is not None:
      loadings.append(loading)
summary = pd.DataFrame(rows)
# 12 モデルの p(Q2) に BH-FDR (全サンプル / sec のみ それぞれで)
summary["q(Q2)"] = np.nan
for subset, s in summary.groupby("subset"):
  ok = s["p(Q2)"].notna()
  summary.loc[s.index[ok], "q(Q2)"] = multipletests(s.loc[ok, "p(Q2)"], method="fdr_bh")[1]
loadings = pd.concat(loadings, ignore_index=True)
print(summary.round(3).to_string(index=False))

summary.to_csv(os.path.join(heredir, "opls_genus_summary.csv"), index=False)
loadings.to_csv(os.path.join(heredir, "opls_genus_loadings.csv"), index=False)
# %% Score plot と Permutation plot (全サンプル / sec のみ の全モデル)
for _, r in summary.iterrows():
  ex = extras[(r["timing"], r["genus"], r["subset"])]
  if ex is None:
    continue
  stem = f"{slug(r['timing'])}_{slug(r['genus'])}" + ("_sec_only" if r["subset"] == "sec only" else "")
  plot_score(ex, r, os.path.join(figdir, f"score_plot_{stem}.svg"))
  plot_permutation(ex, r, os.path.join(figdir, f"permutation_{stem}.svg"))
# %% S-line (全サンプルのモデル)
for _, r in summary[summary["subset"] == "all"].iterrows():
  ld = loadings[(loadings["timing"] == r["timing"]) & (loadings["genus"] == r["genus"])
                & (loadings["subset"] == "all")]
  title = (f"OPLS S-line — {r['timing']} {r['genus']} (positive = higher with more of the genus)\n"
           f"n={r['n']}, detected={r['n_detected']}, 1+{r['n_ortho']} | R2Y={r['R2Y']:.3f}  "
           f"Q2={r['Q2']:.3f}  p={r['p(Q2)']:.3f}  q={r['q(Q2)']:.3f}")
  path = os.path.join(figdir, f"s_line_{slug(r['timing'])}_{slug(r['genus'])}.svg")
  plot_s_line(ld["p_ctr"].to_numpy(), ld["p_corr"].to_numpy(), title, path)
