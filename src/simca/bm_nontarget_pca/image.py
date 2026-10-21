# %%
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
import seaborn as sns

sns.set_style("whitegrid")
# %%
df = pd.read_csv("./data.csv")
df
# %%
fig, ax = plt.subplots()
sns.scatterplot(
  data=df[df["sec"]==True],
  x="M1.t[1]",
  y="M1.t[2]",
  style="sec",
  hue="lactationStage",
  palette="viridis",
  alpha=0.7,
  ax=ax,
)
sns.scatterplot(
  data=df[df["sec"]==False],
  x="M1.t[1]",
  y="M1.t[2]",
  style="sec",
  markers="d",
  hue="lactationStage",
  palette="viridis_r",
  edgecolor="black",
  alpha=0.7,
  zorder=5,
  ax=ax,
)
ax.set_xlabel("PC1 (77.3%)")
ax.set_ylabel("PC2 (11.1%)")
# 2回目のlactationStage(見出し+3項目)と重複するsec見出しを凡例から除く
handles, labels = ax.get_legend_handles_labels()
drop = set(range(6, 11))
legend_titles = {
  "lactationStage": "Lactation stage",
  "sec": "Status",
  "True": "Secretor",
  "False": "Non-secretor",
}
ax.legend(
  [h for i, h in enumerate(handles) if i not in drop],
  [legend_titles.get(l, l) for i, l in enumerate(labels) if i not in drop],
)
plt.show()
# %%
