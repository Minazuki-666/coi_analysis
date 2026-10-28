# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.collections import LineCollection

sns.set_style("whitegrid")
N_ANNOTATE = 15  # S-line でラベルを付ける bin の数 (|p(corr)[1]| の上位)
# %%
df_score = pd.read_csv("./score.txt", sep="\t")
df_sline = pd.read_csv("./sline.txt", sep="\t")
df_sline["abs(M1.p(corr)[1])"] = df_sline["M1.p(corr)[1]"].abs()
# 末尾2行(空行とIntercept行)を除く
df_perm = pd.read_csv("./permutation.txt", sep="\t").iloc[:-2].astype("float")
y_col = "M1.YVar(Bifidobacterium_CLR)"
# %% Score plot (色: Bifidobacterium の CLR、記号: 分泌型)
fig, ax = plt.subplots(figsize=(7.5, 6))
markers = {"sec": "o", "non-sec": "^"}
labels = {"sec": "Secretor", "non-sec": "Non-secretor"}
for cls, marker in markers.items():
  d = df_score[df_score["$ClassID"] == cls]
  sc = ax.scatter(
    d["M1.t[1]"],
    d["M1.to[1]"],
    c=d[y_col],
    cmap="viridis",
    vmin=df_score[y_col].min(),
    vmax=df_score[y_col].max(),
    marker=marker,
    s=50,
    edgecolor="white",
    linewidth=0.5,
    label=labels[cls],
  )
ax.legend(frameon=False)
ax.set_xlabel("Predictive component")
ax.set_ylabel("Orthogonal component")
fig.colorbar(sc, ax=ax, label="CLR(Bifidobacterium)")
fig.savefig("score.svg", bbox_inches="tight")
plt.close(fig)

# %% S-line (annotate=True なら |p(corr)[1]| 上位 N_ANNOTATE 個にケミカルシフトを表記)
def plot_s_line(annotate):
  fig, ax = plt.subplots(figsize=(10, 5))
  d = df_sline.sort_values("Primary ID")
  x = d["Primary ID"].to_numpy()
  y = d["M1.p(ctr)[1]"].to_numpy()
  c = d["abs(M1.p(corr)[1])"].to_numpy()
  norm = plt.Normalize(0, 1)
  # 線は点どうしを直線でつなぐ。色は各ビン(Primary ID ±0.02 ppm)の中心で
  # そのビンの値になり、隣のビンの中心との間は滑らかなグラデーションにする
  n_sub = 20  # 点の間を分割する数(大きいほど滑らか)
  xs = np.concatenate(
    [np.linspace(x[i], x[i + 1], n_sub, endpoint=False) for i in range(len(x) - 1)]
    + [x[-1:]]
  )
  ys = np.interp(xs, x, y)
  cs = np.interp(xs, x, c)
  points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  lc = LineCollection(segments, cmap="mako_r", norm=norm, linewidth=2)
  lc.set_array((cs[:-1] + cs[1:]) / 2)
  ax.add_collection(lc)
  ax.set_xlim(x.max(), x.min())  # NMRスペクトルと同様に高ppmを左に
  ax.set_ylim(y.min() * 1.1, y.max() * 1.1)
  ax.set_xlabel("¹H Chemical shift (ppm)")
  ax.set_ylabel("p(ctr)")
  fig.colorbar(lc, ax=ax, label="Absolute correlation with predictive component")
  if annotate:
    labeled = d.nlargest(N_ANNOTATE, "abs(M1.p(corr)[1])")
    texts = [
      ax.text(r["Primary ID"], r["M1.p(ctr)[1]"], f"{r['Primary ID']:.2f}", fontsize=9)
      for _, r in labeled.iterrows()
    ]
    adjust_text(
      texts,
      ax=ax,
      expand=(1.5, 1.5),
      arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    )
  fig.savefig(f"sline{'_labeled' if annotate else ''}.svg", bbox_inches="tight")
  plt.close(fig)


plot_s_line(annotate=False)
plot_s_line(annotate=True)


# %% Permutation plot
def fit_through_point(x, y, x0, y0):
  """点 (x0, y0) を必ず通る最小二乗直線。傾きと切片(x=0)を返す。"""
  dx = np.asarray(x) - x0
  dy = np.asarray(y) - y0
  slope = np.sum(dx * dy) / np.sum(dx**2)
  intercept = y0 - slope * x0
  return slope, intercept


# 元モデル(Correlation == 1.0)と並べ替えモデルを分離
is_orig = np.isclose(df_perm["Correlation"], 1.0)
perm = df_perm[~is_orig]
orig = df_perm[is_orig]

fig, ax = plt.subplots()
x_line = np.array([0.0, 1.0])
mako = sns.color_palette("mako", 2)
for col, color in zip(["R2", "Q2"], mako):
  y0 = orig[col].iloc[0]
  ax.scatter(perm["Correlation"], perm[col], s=12, color=color, alpha=0.6)
  ax.scatter([1.0], [y0], s=60, color=color, edgecolor="k", zorder=3)
  slope, icpt = fit_through_point(perm["Correlation"], perm[col], 1.0, y0)
  ax.plot(
    x_line, icpt + slope * x_line, color=color, lw=1.5,
    label=f"{col} (intercept = {icpt:.3f})",
  )
ax.set_xlabel("Correlation between original and permuted Y")
ax.set_ylabel("R²Y / Q²")
ax.legend()
plt.tight_layout()
fig.savefig("permutation.svg", bbox_inches="tight")
plt.close(fig)
# %%
