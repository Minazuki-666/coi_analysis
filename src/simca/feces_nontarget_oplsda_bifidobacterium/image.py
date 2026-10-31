# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.collections import LineCollection

sns.set_style("whitegrid")
N_ANNOTATE = 20  # S-line でラベルを付ける bin の数 (|p(corr)[1]| の上位)
AGES = ["newborn", "1mo", "45mo"]  # score_bifidobacterium_{age}.txt / sline_bifidobacterium_{age}.txt
markers = {"sec": "o", "non-sec": "^"}
labels = {"sec": "Secretor", "non-sec": "Non-secretor"}


def plot_score(df_score, age):
  # SIMCA のモデル番号 (M4, M5 ...) は age ごとに違うので、列は接尾辞で探す
  t_col = next(c for c in df_score.columns if c.endswith(".t[1]"))
  to_col = next(c for c in df_score.columns if c.endswith(".to[1]"))
  y_col = next(c for c in df_score.columns if ".YVar(" in c)
  fig, ax = plt.subplots(figsize=(7.5, 6))
  for cls, marker in markers.items():
    d = df_score[df_score["$ClassID"] == cls]
    sc = ax.scatter(
      d[t_col],
      d[to_col],
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
  fig.savefig(f"score_{age}.svg", bbox_inches="tight")
  plt.close(fig)


# S-line (annotate=True なら |p(corr)[1]| 上位 N_ANNOTATE 個にケミカルシフトを表記)
def plot_s_line(df_sline, ctr_col, age, annotate):
  fig, ax = plt.subplots(figsize=(10, 5))
  d = df_sline.sort_values("Primary ID")
  x = d["Primary ID"].to_numpy()
  y = d[ctr_col].to_numpy()
  c = d["abs_corr"].to_numpy()
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
    labeled = d.nlargest(N_ANNOTATE, "abs_corr")
    texts = [
      ax.text(r["Primary ID"], r[ctr_col], f"{r['Primary ID']:.2f}", fontsize=9)
      for _, r in labeled.iterrows()
    ]
    adjust_text(
      texts,
      ax=ax,
      expand=(1.5, 1.5),
      arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    )
  fig.savefig(f"sline_{age}{'_labeled' if annotate else ''}.svg", bbox_inches="tight")
  plt.close(fig)





# %%
for age in AGES:
  df_score = pd.read_csv(f"./score_bifidobacterium_{age}.txt", sep="\t")
  plot_score(df_score, age)
  df_sline = pd.read_csv(f"./sline_bifidobacterium_{age}.txt", sep="\t")
  ctr_col = next(c for c in df_sline.columns if c.endswith(".p(ctr)[1]"))
  corr_col = next(c for c in df_sline.columns if c.endswith(".p(corr)[1]"))
  df_sline["abs_corr"] = df_sline[corr_col].abs()
  plot_s_line(df_sline, ctr_col, age, annotate=False)
  plot_s_line(df_sline, ctr_col, age, annotate=True)
# %%
