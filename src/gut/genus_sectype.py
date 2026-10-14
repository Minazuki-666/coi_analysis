# %%
"""timingごとに sec / non-sec 間で Genus 相対存在量を比較する。"""
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests

PREV_MIN = 0.001  # 各timing内で出現率10%未満のGenusは検定から除外
N_PERM = 9999
rng = np.random.default_rng(0)

heredir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(os.path.dirname(heredir)), "data")
df = pd.read_csv(os.path.join(datadir, "20260901_Genus_RelativeAb.csv"))
df = df.dropna(subset=["ID", "timing", "sec-type"]).copy()
df["timing"] = df["timing"].astype(int)
taxa = df.columns[6:]
short = {c: c.split("|")[-1] if c.split("|")[-1] != "__" else c.split("|")[-2] + "(unclassified)" for c in taxa}

def permanova(X, g, n_perm=N_PERM):
  D = squareform(pdist(X, "braycurtis")) ** 2
  n = len(g)
  def F(lab):
    ssw = 0.0
    for k in np.unique(lab):
      idx = lab == k
      ssw += D[np.ix_(idx, idx)].sum() / (2 * idx.sum())
    sst = D.sum() / (2 * n)
    a = len(np.unique(lab))
    return ((sst - ssw) / (a - 1)) / (ssw / (n - a)), 1 - ssw / sst
  f0, r2 = F(g)
  p = (1 + sum(F(rng.permutation(g))[0] >= f0 for _ in range(n_perm))) / (n_perm + 1)
  return f0, r2, p

# %%
rows, perm_rows = [], []
for t, d in df.groupby("timing"):
  sec = d["sec-type"].values == "sec"
  X = d[taxa].astype(float)
  f, r2, p = permanova(X.values, sec.astype(int))
  perm_rows.append(dict(timing=t, n_sec=sec.sum(), n_non=(~sec).sum(), pseudoF=f, R2=r2, p=p))
  keep = [c for c in taxa if (X[c] > 0).mean() >= PREV_MIN]
  res = []
  for c in keep:
    a, b = X.loc[sec, c], X.loc[~sec, c]
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    res.append(dict(timing=t, genus=short[c], mean_sec=a.mean(), mean_non=b.mean(),
                    median_sec=a.median(), median_non=b.median(),
                    log2FC=np.log2((a.mean() + 1e-6) / (b.mean() + 1e-6)), p=p))
  r = pd.DataFrame(res)
  r["q"] = multipletests(r["p"], method="fdr_bh")[1]
  rows.append(r)
out = pd.concat(rows).sort_values(["timing", "p"])
perm = pd.DataFrame(perm_rows)
out.to_csv(os.path.join(heredir, "genus_sectype_by_timing.csv"), index=False)
perm.to_csv(os.path.join(heredir, "genus_sectype_permanova.csv"), index=False)
print(perm.to_string(index=False))
for t, r in out.groupby("timing"):
  print(f"\n== timing {t}: tested {len(r)} genera, p<0.05: {(r.p<0.05).sum()}, q<0.05: {(r.q<0.05).sum()}")
  print(r.head(8).round(4).to_string(index=False))
