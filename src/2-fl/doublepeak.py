# %%
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme("notebook")
sns.set_style("whitegrid")
# sns.set_palette("muted")
# %%
df = pd.read_csv("../../data/hbm_profile.csv")
conc_cols = list(df.columns[df.columns.get_loc("2'-fucosyllactose"):])  # 濃度の列
DILUTION = 600 / 270  # hbm_profile.csv の濃度は希釈後の値なので、希釈前 (母乳中) の濃度に戻す
df[conc_cols] = df[conc_cols] * DILUTION

# %%
fig, ax = plt.subplots()
sns.histplot(
  data=df,
  x="2'-fucosyllactose",
  hue="timing",
  multiple="stack",
  log_scale=10,
  bins=50,
  palette="crest_r",
  ax=ax,
)
# conc_leftside = 0.55 * DILUTION  # 0.55 / 1.6 mM は希釈後の濃度で決めた境界
# conc_rightside = 1.6 * DILUTION
# ax.vlines(
#   conc_leftside, 
#   0, 43,
#   colors="gray",
#   linestyles="--",
# )
# ax.text(
#   conc_leftside,
#   40,
#   f"{conc_leftside:.2f} mM → ",
#   horizontalalignment = "right"
# )
# ax.vlines(
#   conc_rightside, 
#   0, 43,
#   colors="gray",
#   linestyles="--",
# )
# ax.text(
#   conc_rightside,
#   40,
#   f" ← {conc_rightside:.2f} mM"
# )
# ax.set_ylim(0,43)
timing_labels = {"3": "Newborn", "4": "1 mo.", "5": "4-5 mos."}
for text in ax.get_legend().get_texts():
  text.set_text(timing_labels.get(text.get_text(), text.get_text()))
ax.get_legend().set_title(None)
ax.set_xlabel("2'-FL concentration (mM)")
ax.set_ylabel("Number of samples")
plt.savefig("./doublepeak.png", dpi=300, bbox_inches="tight")
# %%