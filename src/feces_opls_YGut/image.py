# %%
"""乳児便の NMR bin (X) で Bifidobacterium / Bacteroides の存在量 (Y, CLR 変換後) を説明する OPLS (回帰)。

- X: feces_BinningResults.csv の乳児便 (age == "c") 245 bin、Pareto スケーリング
- Y: 20260901_Genus_RelativeAb.csv の相対存在量を全 638 分類群で CLR 変換したもの。
  ゼロが 97% を占めるので、全データの非ゼロ最小値の 1/2 を全要素に加えて再正規化してから CLR をとる
  (CLR はサンプルごとの変換なので、時期で分割しても Y の値は変わらない)
- 対象: 全タイミング (Newborn + 1 mo. + 4-5 mos.) と、各タイミングに分割した計 4 サブセット
- 直交成分数 0-3 を 7-fold CV の Q2 で選ぶ。パーミュテーション検定でも毎回同じ選択をやり直し、
  Q2 の帰無分布を作る (成分数選択の楽観バイアスを含めた検定にするため)
- CV の fold は family 単位で切る (全タイミングでは同じ児が最大 3 回入るため、
  同一児が学習側と検証側に分かれて Q2 が過大になるのを防ぐ)
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
from adjustText import adjust_text
from statsmodels.stats.multitest import multipletests

sns.set_theme(style="white")
sns.set_context("talk", font_scale=0.9)

N_PERM = 1000
N_SPLITS = 7
MAX_ORTHO = 3
SEED = 0
SLINE_CMAP = sns.color_palette("mako_r", as_cmap=True)
SLINE_VMAX = 1.0  # S-line の色は |p(corr)| なので、全範囲で固定してモデル間で比較できるようにする
N_ANNOTATE = 15   # S-line でラベルを付ける bin の数 (|p(corr)[1]| の上位)

TIMING_LABEL = {3: "Newborn", 4: "1 mo.", 5: "4-5 mos."}
# Bacteroides は GTDB で Bacteroides_H / _G / _F に分かれるが、_G と _F は全サンプルで 0 なので _H のみ
TARGETS = {
  "Bifidobacterium": "g__Bifidobacterium_388775",
  "Bacteroides": "g__Bacteroides_H",
}
SUBSETS = [("All timings", None), ("Newborn", 3), ("1 mo.", 4), ("4-5 mos.", 5)]
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


def clr(rel):
  """CLR。ゼロは非ゼロ最小値の 1/2 を全要素に加えて処理し、再正規化してから log 比をとる"""
  pseudo = rel[rel > 0].min() / 2
  comp = rel + pseudo
  comp = comp / comp.sum(axis=1, keepdims=True)
  logc = np.log(comp)
  return logc - logc.mean(axis=1, keepdims=True), pseudo


clr_all, pseudo = clr(genus[taxa].to_numpy(dtype=float))
print(f"CLR: {len(taxa)} taxa, pseudocount = {pseudo:.3g}")

clr_df = genus[["ID"]].copy()
for name, key in TARGETS.items():
  col = [c for c in taxa if c.endswith("|" + key)]
  assert len(col) == 1, (key, col)
  clr_df[name] = clr_all[:, taxa.index(col[0])]
  clr_df[name + "_rel"] = genus[col[0]].to_numpy(dtype=float)

# timing と sec-type は両ファイルで一致することを確認済みなので feces 側を使う
df = feces.merge(clr_df, on="ID", how="inner")
df = df[df["timing"].isin(TIMING_LABEL)].reset_index(drop=True)
print(f"matched samples: {len(df)} / {len(genus)}")
print(pd.crosstab(df["timing"].map(TIMING_LABEL), df["sec-type"]))
print("families per timing (重複なしなら n と一致):",
      df.groupby("timing")["family"].nunique().to_dict())
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


def make_folds(groups, n_splits=N_SPLITS, seed=SEED):
  """group (= family) 単位の k-fold。同じ児のサンプルは必ず同じ fold に入る"""
  uniq = np.random.default_rng(seed).permutation(np.unique(groups))
  fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
  fid = np.array([fold_of[g] for g in groups])
  return [np.flatnonzero(fid == k) for k in range(n_splits)]


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


def perm_line(corr, vals, obs):
  """permutation plot の回帰直線。観測値 (x=1, obs) を必ず通るように、
  y = obs + slope * (x - 1) の傾きを最小二乗で求め、(slope, x=0 での切片) を返す"""
  dx = corr - 1
  slope = np.sum(dx * (vals - obs)) / np.sum(dx ** 2)
  return slope, obs - slope


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


def plot_s_line(p_ctr, p_corr, title, path, annotate=True):
  """S-line: p(ctr)[1] を化学シフトに対して描き、|p(corr)[1]| で色付けする。
  annotate=False なら上位 bin の丸印と ppm ラベルを付けない版になる"""
  order = np.argsort(-ppm)
  x, y, c = ppm[order], p_ctr[order], np.abs(p_corr[order])
  lc = gradient_line(x, y, c, SLINE_CMAP, SLINE_VMAX)
  fig, ax = plt.subplots(figsize=(12, 5))
  ax.add_collection(lc)
  ax.set_xlim(x.max(), x.min())
  pad = (0.20 if annotate else 0.05) * (y.max() - y.min())
  ax.set_ylim(y.min() - pad, y.max() + pad)
  ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)

  if annotate:
    top = np.argsort(-np.abs(p_corr))[:N_ANNOTATE]
    ax.scatter(ppm[top], p_ctr[top], s=14, facecolor="none",
               edgecolor="#c0392b", linewidth=1.0, zorder=3)
    # |p(corr)| 上位の bin は ppm 上で固まるので、ラベルは adjustText で重ならないように配置する
    texts = [ax.text(ppm[i], p_ctr[i], f"{ppm[i]:.2f}", ha="center", va="center",
                     fontsize=9, color="#c0392b", zorder=4) for i in top]
    adjust_text(texts, ax=ax, x=ppm[top], y=p_ctr[top], expand=(1.3, 1.6),
                arrowprops=dict(arrowstyle="-", color="#c0392b", linewidth=0.6))

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
MARKERS = ["o", "^", "s"]


def plot_score(ex, r, path):
  """Score plot: t_pred[1] vs t_ortho[1]。直交成分が 0 のモデルは縦軸を観測値 y にする。
  色は y、記号は全タイミングなら採取時期、各タイミングなら分泌型 (sec / non-sec)。"""
  tp, to1, y, grp = ex["t_pred"], ex["t_ortho"], ex["y"], ex["group"]
  fig, ax = plt.subplots(figsize=(7.5, 6))
  for marker, cls in zip(MARKERS, sorted(set(grp), key=list(ex["group_order"]).index)):
    sel = grp == cls
    sc = ax.scatter(tp[sel], (to1 if to1 is not None else y)[sel], c=y[sel], cmap="viridis",
                    vmin=y.min(), vmax=y.max(), marker=marker, s=50, alpha=0.8,
                    edgecolor="white", linewidth=0.5, label=f"{cls} (n={sel.sum()})")
  if to1 is not None:   # Hotelling T2 95% 楕円
    n = len(y)
    radius = np.sqrt(2 * (n - 1) / (n - 2) * f_dist.ppf(0.95, 2, n - 2))
    ax.add_patch(Ellipse((0, 0), 2 * radius * tp.std(ddof=1), 2 * radius * to1.std(ddof=1),
                         fill=False, edgecolor="gray", linestyle="--", linewidth=1.2))
    ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.set_xlabel("$t_{pred}$ [1]")
  ax.set_ylabel("$t_{ortho}$ [1]" if to1 is not None else f"observed y (CLR {r['genus']})")
  ax.set_title(f"OPLS score plot — {r['subset']}, {r['genus']}\n"
               f"1+{r['n_ortho']}, pareto | R2Y={r['R2Y']:.3f}  Q2={r['Q2']:.3f}  "
               f"$p_{{Q2}}$={r['p(Q2)']:.3f}", fontsize=11)
  ax.legend(frameon=False, fontsize=11)
  fig.colorbar(sc, ax=ax, label=f"y = CLR({r['genus']})")
  fig.tight_layout()
  fig.savefig(path)
  plt.show()


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
    slope, intercept = perm_line(perm[:, 0], perm[:, j], obs)
    ax.plot([0, 1], [intercept, obs], color=color, linewidth=1.2, linestyle="--")
  ax.text(0.03, 0.03, f"intercepts: R2Y={r['int(R2Y)']:.3f}, Q2={r['int(Q2)']:.3f}",
          transform=ax.transAxes, fontsize=11)
  ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
  ax.set_xlabel("|corr(permuted y, y)|")
  ax.set_ylabel("R2Y / Q2")
  ax.set_title(f"Permutation plot — {r['subset']}, {r['genus']}\n"
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
  plt.show()
# %%
def run_model(subset, timing, genus_name):
  d = df if timing is None else df[df["timing"] == timing]
  X = d[bin_cols].to_numpy(dtype=float)
  y = d[genus_name].to_numpy(dtype=float)
  detected = d[genus_name + "_rel"].to_numpy() > 0
  folds = make_folds(d["family"].to_numpy())

  n_ortho, r2y, q2 = select_model(X, y, folds)

  rng = np.random.default_rng(SEED)
  perm = np.empty((N_PERM, 3))   # |corr(y_perm, y)|, R2Y, Q2
  for i in range(N_PERM):
    yp = rng.permutation(y)
    perm[i] = [abs(np.corrcoef(yp, y)[0, 1]), *select_model(X, yp, folds)[1:]]
  p_r2y = (1 + np.sum(perm[:, 1] >= r2y)) / (1 + N_PERM)
  p_q2 = (1 + np.sum(perm[:, 2] >= q2)) / (1 + N_PERM)
  # 観測値 (x=1) を通る回帰直線の x=0 での切片。Q2 の切片が負なら過適合の疑いは小さい
  intercepts = {k: perm_line(perm[:, 0], perm[:, j], obs)[1]
                for k, j, obs in [("R2Y", 1, r2y), ("Q2", 2, q2)]}

  Xs, _, x_scale = pareto_scale(X)
  model = opls_fit(Xs, y - y.mean(), n_ortho)
  tp = model["t_pred"]
  ss_x = np.sum(Xs ** 2)
  r2x = (np.sum(np.outer(tp, model["p_pred"]) ** 2)
         + np.sum((model["T_ortho"] @ model["P_ortho"].T) ** 2)) / ss_x
  # SIMCA 表記: p(ctr) は予測ローディングを中心化のみの単位に戻したもの、p(corr) はスコアと変数の相関
  p_ctr = model["p_pred"] * x_scale
  p_corr = ((Xs.T @ tp) / (len(y) - 1)) / (Xs.std(axis=0, ddof=1) * tp.std(ddof=1))
  # 予測ローディングの向きを「Genus が多いほど正」にそろえる
  sign = np.sign(np.corrcoef(tp, y)[0, 1])
  p_ctr, p_corr = sign * p_ctr, sign * p_corr

  row = {"subset": subset, "genus": genus_name, "n": len(d), "n_detected": int(detected.sum()),
         "prevalence": detected.mean(), "n_ortho": n_ortho, "R2X": r2x,
         "R2Y": r2y, "p(R2Y)": p_r2y, "Q2": q2, "p(Q2)": p_q2,
         "int(R2Y)": intercepts["R2Y"], "int(Q2)": intercepts["Q2"]}
  loading = pd.DataFrame({"subset": subset, "genus": genus_name,
                          "ppm": ppm, "p_ctr": p_ctr, "p_corr": p_corr})
  if timing is None:
    group, order = d["timing"].map(TIMING_LABEL).to_numpy(), list(TIMING_LABEL.values())
  else:
    group, order = d["sec-type"].to_numpy(), ["sec", "non-sec"]
  extra = {"t_pred": sign * tp, "t_ortho": model["T_ortho"][:, 0] if n_ortho >= 1 else None,
           "y": y, "group": group, "group_order": order, "perm": perm}
  return row, loading, extra
# %%
rows, loadings, extras = [], [], {}
for subset, timing in SUBSETS:
  for genus_name in TARGETS:
    row, loading, extra = run_model(subset, timing, genus_name)
    rows.append(row)
    loadings.append(loading)
    extras[(subset, genus_name)] = extra
    print(f"{subset:12s} {genus_name:16s} n={row['n']:3d} 1+{row['n_ortho']} "
          f"R2Y={row['R2Y']:.3f} Q2={row['Q2']:.3f} p(Q2)={row['p(Q2)']:.3f}")
summary = pd.DataFrame(rows)
summary["q(Q2)"] = multipletests(summary["p(Q2)"], method="fdr_bh")[1]   # 8 モデルで BH-FDR
loadings = pd.concat(loadings, ignore_index=True)
print(summary.round(3).to_string(index=False))

summary.to_csv(os.path.join(heredir, "opls_summary.csv"), index=False)
loadings.to_csv(os.path.join(heredir, "opls_loadings.csv"), index=False)
# %% Score plot / S-line / Permutation plot
for _, r in summary.iterrows():
  ex = extras[(r["subset"], r["genus"])]
  stem = f"{slug(r['subset'])}_{r['genus']}"
  plot_score(ex, r, os.path.join(figdir, f"score_plot_{stem}.svg"))
  ld = loadings[(loadings["subset"] == r["subset"]) & (loadings["genus"] == r["genus"])]
  title = (f"OPLS S-line — {r['subset']}, {r['genus']} (positive = higher CLR {r['genus']})\n"
           f"n={r['n']}, detected={r['n_detected']}, 1+{r['n_ortho']} | R2Y={r['R2Y']:.3f}  "
           f"Q2={r['Q2']:.3f}  p={r['p(Q2)']:.3f}  q={r['q(Q2)']:.3f}")
  plot_s_line(ld["p_ctr"].to_numpy(), ld["p_corr"].to_numpy(), title,
              os.path.join(figdir, f"s_line_{stem}.svg"))
  plot_s_line(ld["p_ctr"].to_numpy(), ld["p_corr"].to_numpy(), title,
              os.path.join(figdir, f"s_line_{stem}_nomarker.svg"), annotate=False)
  plot_permutation(ex, r, os.path.join(figdir, f"permutation_{stem}.svg"))

# %%
