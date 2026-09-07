# %%
import os, glob, re, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.weightstats import ttest_ind

sns.set_theme(style="white")
sns.set_context("talk", font_scale=1.2)
# %%
heredir = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.dirname(heredir).split(os.sep)
basedir = os.sep.join(basedir[:-1])
datadir = os.path.join(basedir, "data")
datadir
#%%
df = pd.read_csv(os.path.join(datadir, "hbm_profile.csv"))
df["timing"] = df["timing"].replace({3: 0, 4: 1, 5: 2})
df["sec"] = (df["sec"] == "sec")
df
#%%
for metabolite in df.columns[5:]:
  tukey_results = pairwise_tukeyhsd(
      endog=df[metabolite],
      groups=df["timing"],
      alpha=0.05
  )
  tukey_results = tukey_results.summary_frame()
  tukey_results["p-adj"] = tukey_results["p-adj"].round(3)

  fig, ax = plt.subplots(figsize=(6, 6))
  sns.stripplot(
    data=df, 
    x="timing",
    y=metabolite, 
    color="black",
  alpha=0.4,
  ax=ax
  )
  sns.boxplot(
    data=df,
    x="timing",
    y=metabolite,
    showfliers=False,
    color="lightgray",
    fill=False,
    ax=ax
  )
  height_original = ax.get_ylim()[1]
  height_max = height_original

  for i, j in itertools.combinations(df["timing"].unique(), 2):
    if i > j:
      i, j = j, i
    height_now = height_original * 1.1 ** (abs(i-j)-1)
    if height_now > height_max:
      height_max = height_now
    tukey_results_selected = tukey_results[
      (tukey_results["group_t"].isin([i, j])) & 
      (tukey_results["group_c"].isin([i, j]))
    ]
    pval = tukey_results_selected["p-adj"].values[0]
    if pval < 0.001:
      pval_str = "p < 0.001"
    else:
      pval_str = f"p = {pval:.3f}"
    if tukey_results_selected["reject"].values[0]:
      ax.text(
        x=(i+j)/2,
        y=height_now,
        s=pval_str,
        fontsize=16,
        ha="center",
        va="bottom"
      )
      ax.hlines(
        y=height_now,
        xmin=i + 0.1,
        xmax=j - 0.1,
        color="black",
        linewidth=1
      )

  ax.set_ylim(top=height_max * 1.1)
  ax.set_xticks([0, 1, 2])
  ax.set_xticklabels(["Newborn", "1 mo.", "4-5 mos."])
  ax.set_xlabel("")
  ax.set_ylabel("Concentration (mM)")
  ax.set_title(metabolite)
  os.makedirs(os.path.join(heredir, "fig", "timingStats"), exist_ok=True)
  plt.savefig(
    os.path.join(heredir, "fig", "timingStats", f"{metabolite}.svg"),
    bbox_inches="tight"
  )

# %%
for metabolite in df.columns[5:]:
  fig, ax = plt.subplots(figsize=(6, 6))
  sns.stripplot(
    data=df, 
    x="timing",
    y=metabolite,
    hue="sec",
    dodge=True,
    palette="deep",
  alpha=0.4,
  ax=ax
  )
  sns.boxplot(
    data=df,
    x="timing",
    y=metabolite,
    hue="sec",
    dodge=True,
    showfliers=False,
    palette="deep",
    fill=False,
    legend=False,
    ax=ax
  )
  height_original = ax.get_ylim()[1]

  for i in df["timing"].unique():
    ttest_results = ttest_ind(
      df[(df["timing"] == i) & (df["sec"] == True)][metabolite],
      df[(df["timing"] == i) & (df["sec"] == False)][metabolite],
      usevar="unequal"
    )
    pval = ttest_results[1]
    if pval < 0.001:
      pval_str = "p < 0.001"
    else:
      pval_str = f"p = {pval:.3f}"

    if pval < 0.05:
      ax.text(
        x=i,
        y=height_original,
        s=pval_str,
        fontsize=16,
        ha="center",
        va="bottom"
      )
      ax.hlines(
        y=height_original,
        xmin=i - 0.2,
        xmax=i + 0.2,
        color="black",
        linewidth=1
      )

  ax.set_ylim(top=height_original * 1.1)
  ax.set_xticks([0, 1, 2])
  ax.set_xticklabels(["Newborn", "1 mo.", "4-5 mos."])
  ax.set_xlabel("")
  ax.set_ylabel("Concentration (mM)")
  ax.set_title(metabolite)
  ax.legend(["Se+", "Se-"], title="")
  os.makedirs(os.path.join(heredir, "fig", "secStats"), exist_ok=True)
  plt.savefig(
    os.path.join(heredir, "fig", "secStats", f"{metabolite}.svg"),
    bbox_inches="tight"
  )

# %%
