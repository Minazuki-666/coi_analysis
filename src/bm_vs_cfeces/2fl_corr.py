# %%
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
# %%
df_bm = pd.read_csv("../data/hbm_profile.csv")
df_bm["key"] = df_bm["family"].astype(str) + "_" + df_bm["timing"].astype(str)
df_bm = df_bm.reindex(columns=["key", "timing", "sec", "2'-fucosyllactose", "3-fucosyllactose",])
df_bm
#%%
df_cfeces = pd.read_csv("../data/feces_BinningResults.csv", header=0)
df_cfeces = df_cfeces[df_cfeces["age"] == "c"]
df_cfeces["key"] = df_cfeces["family"].astype(str) + "_" + df_cfeces["timing"].astype(str)
df_cfeces = df_cfeces.reindex(columns=df_cfeces.columns[6:][::-1])
df_cfeces
# %%
df_merge = pd.merge(
  df_bm, df_cfeces, on="key"
)
df_merge
#%%
df_corr = pd.DataFrame(columns=["bm", "cfeces", "timing", "corr", "pval"])
for timing in [3, 4, 5]:
  for i in ["2'-fucosyllactose", "3-fucosyllactose"]:
    for j in ["1.18", "1.22", "4.62", "5.3"]:
      df_merge_temp = df_merge[df_merge["timing"] == timing]
      corr, pval = spearmanr(df_merge_temp[i], df_merge_temp[j])
      df_corr_temp = pd.DataFrame([{"bm": i, "cfeces": j, "timing": timing, "corr": corr, "pval": pval}])
      df_corr = pd.concat([df_corr, df_corr_temp], ignore_index=True)
df_corr
# %%
def star(pval):
  if pval < 0.001:
    return "p < 0.001"
  elif pval < 0.05:
    return f"p = {pval:.3f}"
  else:
    return ""
#%%
for timing in [3, 4, 5]:
  df_corr_temp = df_corr[df_corr["timing"] == timing]
  df_corr_temp_pivot = df_corr_temp.pivot(
    columns="bm", index="cfeces", values="corr",
  )
  df_corr_pval_temp_pivot = df_corr_temp.pivot(
    columns="bm", index="cfeces", values="pval",
  ).astype(float)
  df_corr_pval_temp_pivot = df_corr_pval_temp_pivot.map(star).astype(str)

  fig = plt.figure(figsize=(4, 2))
  ax = fig.add_subplot(111)
  sns.heatmap(
    data=df_corr_temp_pivot.astype(float),
    annot=df_corr_pval_temp_pivot.values,
    cmap="coolwarm", 
    fmt="",
    center=0,
    vmin=-1, 
    vmax=1,
    ax=ax,
  )
  converted_timing = {3: "Newborn", 4: "1 mo.", 5: "4-5 mos."}
  ax.set_title(f"Spearman Correlation (timing={converted_timing[timing]})")
  ax.set_xlabel("Breast milk")
  ax.set_ylabel("Children feces")
  plt.show()

# %%
