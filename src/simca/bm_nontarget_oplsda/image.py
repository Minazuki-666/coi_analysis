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
df_score = pd.read_csv("./score.txt", sep="\t")
df_sline = pd.read_csv("./sline.txt", sep="\t")
df_sline["abs(M6.p(corr)[1])"] = df_sline["M6.p(corr)[1]"].abs()
# %%
fig, ax = plt.subplots()
sns.scatterplot(
  data=df_score,
  x="M6.t[1]",
  y="M6.to[1]",
  hue="secretor",
  palette="mako",
  hue_order=[True, False]
)
secretor_labels = {"TRUE": "Secretor", "FALSE": "Non-secretor"}
for text in ax.get_legend().get_texts():
  text.set_text(secretor_labels.get(text.get_text().upper(), text.get_text()))
ax.get_legend().set_title(None)
ax.set_xlabel("Predictive component")
ax.set_ylabel("Orthogonal component")
fig.savefig("score.svg", bbox_inches="tight")
plt.close(fig)

# %%
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))
d = df_sline.sort_values("Primary ID")
x = d["Primary ID"].to_numpy()
y = d["M6.p(ctr)[1]"].to_numpy()
c = d["abs(M6.p(corr)[1])"].to_numpy()
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
fig.savefig("sline.svg", bbox_inches="tight")
plt.close(fig)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
d = df_sline.sort_values("Primary ID")
x = d["Primary ID"].to_numpy()
y = d["M6.p(ctr)[1]"].to_numpy()
c = d["abs(M6.p(corr)[1])"].to_numpy()
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
# |p(corr)[1]|>0.3 のビンのケミカルシフトを表記
labeled = d[d["abs(M6.p(corr)[1])"] > 0.3]
texts = [
  ax.text(r["Primary ID"], r["M6.p(ctr)[1]"], f"{r['Primary ID']:.2f}", fontsize=9)
  for _, r in labeled.iterrows()
]
adjust_text(
  texts,
  ax=ax,
  expand=(1.5, 1.5),
  arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
)
fig.savefig("sline_labeled.svg", bbox_inches="tight")
plt.close(fig)
# %%
