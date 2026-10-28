# %%
import os, glob, re
from matplotlib.cm import viridis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

sns.set_style("whitegrid")
# %%
df_score = pd.read_csv("./score.txt", sep="\t")
df_loading = pd.read_csv("./loading.txt", sep="\t")
df_loading["abs(M2.p(corr)[1])"] = df_loading["M2.p(corr)[1]"].abs()
# %%
fig, ax = plt.subplots()
sns.scatterplot(
  data=df_score,
  x="M2.t[1]",
  y="M2.to[1]",
  hue="M2.YVar(timing)",
  palette="crest",
)
timing_labels = {3: "Newborn", 4: "1 mo.", 5: "4-5 mos."}
for text in ax.get_legend().get_texts():
  text.set_text(timing_labels.get(int(float(text.get_text())), text.get_text()))
ax.get_legend().set_title(None)
ax.set_xlabel("Predictive component (64.3%)")
ax.set_ylabel("Orthogonal component (20.0%)")
fig.savefig("score.svg", bbox_inches="tight")
plt.close(fig)

# %%
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
  data=df_loading,
  x="M2.pq[1]",
  y="M2.poso[1]",
  hue="M2.p(corr)[1]",
  hue_norm=(-1, 1),
  palette="coolwarm",
  legend=False,
  ax=ax,
)
name_col = df_loading.columns[0]  # "Primary ID" (代謝物名)
texts = []
for _, row in df_loading[df_loading["abs(M2.p(corr)[1])"] > 0.5].iterrows():
  texts.append(
    ax.text(
      row["M2.pq[1]"],
      row["M2.poso[1]"],
      row[name_col],
      fontsize=9,
      horizontalalignment="right" if row["M2.p(corr)[1]"] > 0 else "left",
    )
  )
adjust_text(
  texts,
  ax=ax,
  expand=(1.9, 1.9),
  arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
  zorder=5
)
norm = plt.Normalize(-1, 1)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
ax.set_xlabel("Predictive component (64.3%)")
ax.set_ylabel("Orthogonal component (20.0%)")
fig.colorbar(sm, ax=ax, label="Correlation with predictive component")
fig.savefig("loading.svg", bbox_inches="tight")
plt.close(fig)
# %%
