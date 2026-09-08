# %%
import os, glob, re, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttest_ind

sns.set_theme(style="white")
sns.set_context("talk", font_scale=1.2)

FDR_ALPHA = 0.05

def fdr_bh(pvals):
  """Benjamini-Hochberg FDR correction (NaN-safe)."""
  pvals = np.asarray(pvals, dtype=float)
  qvals = np.full(pvals.shape, np.nan)
  valid = ~np.isnan(pvals)
  if valid.any():
    qvals[valid] = multipletests(pvals[valid], alpha=FDR_ALPHA, method="fdr_bh")[1]
  return qvals

def pval_label(pval, symbol="q"):
  if pval < 0.001:
    return f"{symbol} < 0.001"
  return f"{symbol} = {pval:.3f}"
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

metabolites = list(df.columns[5:])
#%%
# Tukey HSD adjusts within a metabolite, i.e. the 3 timing pairs drawn in one
# figure are the family (no correction across metabolites).
timing_stats = []
for metabolite in metabolites:
  tukey_results = pairwise_tukeyhsd(
      endog=df[metabolite],
      groups=df["timing"],
      alpha=0.05,
      use_var="unequal",
  ).summary_frame()
  tukey_results.insert(0, "metabolite", metabolite)
  timing_stats.append(tukey_results)

timing_stats = pd.concat(timing_stats, ignore_index=True)
timing_stats.to_csv(
  os.path.join(heredir, "timingStats_adjusted.csv"), index=False
)
timing_stats
#%%
for metabolite in metabolites:
  tukey_results = timing_stats[timing_stats["metabolite"] == metabolite]

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
    if tukey_results_selected["reject"].values[0]:
      ax.text(
        x=(i+j)/2,
        y=height_now,
        s=pval_label(pval, symbol="p"),
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
# Welch t-test for Se+ vs Se- at every timing; BH-FDR is applied within each
# metabolite, i.e. the 3 timings drawn in one figure are the family.
sec_stats = []
for metabolite in metabolites:
  for i in sorted(df["timing"].unique()):
    sec_pos = df[(df["timing"] == i) & (df["sec"] == True)][metabolite].dropna()
    sec_neg = df[(df["timing"] == i) & (df["sec"] == False)][metabolite].dropna()
    if len(sec_pos) < 2 or len(sec_neg) < 2:
      pval = np.nan
    else:
      pval = ttest_ind(sec_pos, sec_neg, usevar="unequal")[1]
    sec_stats.append({
      "metabolite": metabolite,
      "timing": i,
      "n_sec_pos": len(sec_pos),
      "n_sec_neg": len(sec_neg),
      "p-value": pval,
    })

sec_stats = pd.DataFrame(sec_stats)
sec_stats["q-value"] = (
  sec_stats.groupby("metabolite")["p-value"].transform(fdr_bh)
)
sec_stats["reject_fdr"] = sec_stats["q-value"] < FDR_ALPHA
sec_stats.to_csv(os.path.join(heredir, "secStats_fdr.csv"), index=False)
sec_stats
# %%
for metabolite in metabolites:
  sec_results = sec_stats[sec_stats["metabolite"] == metabolite]

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
    sec_results_selected = sec_results[sec_results["timing"] == i]
    qval = sec_results_selected["q-value"].values[0]

    if sec_results_selected["reject_fdr"].values[0]:
      ax.text(
        x=i,
        y=height_original,
        s=pval_label(qval),
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
