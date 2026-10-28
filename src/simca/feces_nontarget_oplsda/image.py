# %%
import os, glob, re
from matplotlib.cm import viridis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.collections import LineCollection

sns.set_style("whitegrid")
# %%
# 時期ごとの入力。sline は列名が異なるため (ppm, p_ctr, p_corr) に揃える
DATASETS = {
  "1mo": dict(
    score="./score_1mo.txt",
    sline=lambda: pd.read_csv("./sline_1mo.txt", sep="\t").rename(
      columns={
        "Primary ID": "ppm",
        "M12.p(ctr)[1]": "p_ctr",
        "M12.p(corr)[1]": "p_corr",
      }
    ),
  ),
  "newborn": dict(
    score="./score_newborn.txt",
    sline=lambda: pd.read_csv("../../opls/feces_opls_sline_newborn.csv"),
  ),
}


def plot_score(df_score):
  fig, ax = plt.subplots()
  sns.scatterplot(
    data=df_score,
    x="M12.t[1]",
    y="M12.to[1]",
    hue="secretor",
    palette="mako",
    hue_order=[True, False],
    ax=ax,
  )
  secretor_labels = {"TRUE": "Secretor", "FALSE": "Non-secretor"}
  for text in ax.get_legend().get_texts():
    text.set_text(secretor_labels.get(text.get_text().upper(), text.get_text()))
  ax.get_legend().set_title(None)
  ax.set_xlabel("Predictive component")
  ax.set_ylabel("Orthogonal component")
  return fig


def plot_sline(df_sline, label_peaks=False):
  fig, ax = plt.subplots(figsize=(10, 5))
  d = df_sline.assign(abs_corr=df_sline["p_corr"].abs()).sort_values("ppm")
  x = d["ppm"].to_numpy()
  y = d["p_ctr"].to_numpy()
  c = d["abs_corr"].to_numpy()
  norm = plt.Normalize(0, 1)
  # 線は点どうしを直線でつなぐ。色は各ビン(ppm ±0.02)の中心で
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
  if label_peaks:
    # |p(corr)[1]|>0.3 のビンのケミカルシフトを表記
    labeled = d[d["abs_corr"] > 0.3]
    texts = [
      ax.text(r["ppm"], r["p_ctr"], f"{r['ppm']:.2f}", fontsize=9)
      for _, r in labeled.iterrows()
    ]
    adjust_text(
      texts,
      ax=ax,
      expand=(1.5, 1.5),
      arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    )
  return fig


# %%
for name, cfg in DATASETS.items():
  df_score = pd.read_csv(cfg["score"], sep="\t")
  df_sline = cfg["sline"]()
  figs = {
    f"score_{name}.svg": plot_score(df_score),
    f"sline_{name}.svg": plot_sline(df_sline),
    f"sline_labeled_{name}.svg": plot_sline(df_sline, label_peaks=True),
  }
  for fname, fig in figs.items():
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
# %%
