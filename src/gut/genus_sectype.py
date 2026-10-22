# %%
"""timingごとに sec / non-sec 間で Genus レベルの多様性と各Genusの存在量を比較する。

- alpha多様性 (Observed features): Mann-Whitney U検定
- beta多様性 (Bray-Curtis): PCoA + PERMANOVA
- 各Genus: Mann-Whitney U検定 (多重検定補正なし)。p<0.05のGenusはstripplotを描画
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform

sns.set_style("whitegrid")
N_PERM = 9999
PREV_MIN = 0.10  # 検定するGenusは、そのtiming内で出現率10%以上のものに限る
ALPHA = 0.05
rng = np.random.default_rng(0)

heredir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(os.path.dirname(heredir)), "data")
df = pd.read_csv(os.path.join(datadir, "20260901_Genus_RelativeAb.csv"))
df = df.dropna(subset=["ID", "timing", "sec-type"]).copy()  # 末尾の空行を除く
df["timing"] = df["timing"].astype(int)

# feces_BinningResults.csv の id の上11桁と一致するIDのサンプルだけを解析に使う
feces_ids = pd.read_csv(os.path.join(datadir, "feces_BinningResults.csv"), usecols=["id"])["id"]
feces_id11 = set(feces_ids.astype(str).str[:11])
in_feces = df["ID"].isin(feces_id11)
n_before = df.groupby("timing").size()
df = df[in_feces].copy()
taxa = list(df.columns[6:])
df[taxa] = df[taxa].astype(float)

timing_labels = {3: "Newborn", 4: "1 mo.", 5: "4-5 mos."}
group_labels = {"sec": "Secretor", "non-sec": "Non-secretor"}
group_order = ["sec", "non-sec"]
palette = dict(zip(group_order, sns.color_palette("mako", 3)[:2]))

# Genus名。属が表記されていない列は、同定できた最下位の分類群から後ろをそのまま残し
# "f__Enterobacteriaceae_A|__ (unassigned)" のように表記する
def taxon_name(lineage):
  ranks = lineage.split("|")
  unnamed = [r.endswith("__") for r in ranks]  # "__", "g__", "f__" など名前が空の階級
  if not unnamed[-1]:
    return ranks[-1]
  last = max(i for i, u in enumerate(unnamed) if not u)
  return "|".join(ranks[last:]) + " (unassigned)"

names = {c: taxon_name(c) for c in taxa}
assert len(set(names.values())) == len(taxa), "Genus名が重複しています"
n_unassigned = sum(n.endswith("(unassigned)") for n in names.values())
print(f"samples: {len(df)}, taxa: {len(taxa)}, unassigned: {n_unassigned}")
# 各timingの使用サンプル数
n_used = pd.crosstab(df["timing"].map(timing_labels), df["sec-type"])
n_used["total"] = n_used.sum(axis=1)
n_used["Genus csv total"] = n_before.rename(index=timing_labels)
n_used["excluded"] = n_used["Genus csv total"] - n_used["total"]
n_used = n_used.loc[list(timing_labels.values())]
print(f"feces_BinningResults.csv と一致: {in_feces.sum()} / {len(in_feces)} samples")
print(n_used.to_string())
n_used.to_csv(os.path.join(heredir, "genus_sample_counts.csv"))

# %%
# alpha多様性
# 検出されたfeature(Genus)数。relative abundance > 0 を検出とみなす
alpha = df[["ID", "timing", "sec-type"]].copy()
alpha["Observed features"] = (df[taxa].to_numpy() > 0).sum(axis=1)
metrics = ["Observed features"]

alpha_rows = []
for t, d in alpha.groupby("timing"):
  for m in metrics:
    a = d.loc[d["sec-type"] == "sec", m]
    b = d.loc[d["sec-type"] == "non-sec", m]
    alpha_rows.append(dict(
      timing=timing_labels[t], metric=m, n_sec=len(a), n_non=len(b),
      median_sec=a.median(), median_non=b.median(),
      p=mannwhitneyu(a, b, alternative="two-sided").pvalue,
    ))
alpha_stats = pd.DataFrame(alpha_rows)
print(alpha_stats.round(4).to_string(index=False))

def p_text(p):
  return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"

# %%
# beta多様性 (Bray-Curtis): 時期ごとにPERMANOVAとPCoAを計算
def permanova(D2, g, n_perm=N_PERM):
  n = len(g)
  sst = D2.sum() / (2 * n)
  def F(lab):
    ssw = 0.0
    for k in np.unique(lab):
      idx = lab == k
      ssw += D2[np.ix_(idx, idx)].sum() / (2 * idx.sum())
    a = len(np.unique(lab))
    return ((sst - ssw) / (a - 1)) / (ssw / (n - a)), 1 - ssw / sst
  f0, r2 = F(g)
  p = (1 + sum(F(rng.permutation(g))[0] >= f0 for _ in range(n_perm))) / (n_perm + 1)
  return f0, r2, p

def pcoa(D):
  n = len(D)
  J = np.eye(n) - np.ones((n, n)) / n
  B = -0.5 * J @ (D ** 2) @ J
  w, v = np.linalg.eigh(B)
  order = np.argsort(w)[::-1]
  w, v = w[order], v[:, order]
  pos = w > 0
  coords = v[:, :2] * np.sqrt(w[:2])
  explained = w[:2] / w[pos].sum()
  return coords, explained

def confidence_ellipse(xy, ax, color, level=0.95):
  """2次元の点群の共分散から、正規分布を仮定した level (既定95%) の楕円を描く。"""
  mean = xy.mean(axis=0)
  vals, vecs = np.linalg.eigh(np.cov(xy, rowvar=False))
  vals, vecs = vals[::-1], vecs[:, ::-1]
  angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
  width, height = 2 * np.sqrt(vals * chi2.ppf(level, 2))
  ax.add_patch(Ellipse(mean, width, height, angle=angle, facecolor=color,
                       edgecolor=color, alpha=0.04, lw=0))
  ax.add_patch(Ellipse(mean, width, height, angle=angle, facecolor="none",
                       edgecolor=color, lw=1.5))

beta_rows, pcoa_res = [], {}
for t, d in df.groupby("timing"):
  D = squareform(pdist(d[taxa].to_numpy(), "braycurtis"))
  g = (d["sec-type"] == "sec").to_numpy().astype(int)
  f, r2, p = permanova(D ** 2, g)
  beta_rows.append(dict(timing=timing_labels[t], n_sec=g.sum(), n_non=(1 - g).sum(),
                        pseudoF=f, R2=r2, p=p))
  pcoa_res[t] = pcoa(D)
beta_stats = pd.DataFrame(beta_rows)
print(beta_stats.round(4).to_string(index=False))

# %%
# 時期ごとに alpha多様性 (3指標) と beta多様性 (PCoA) を1枚の図にまとめる
alpha["Group"] = alpha["sec-type"].map(group_labels)
fig_names = {3: "newborn", 4: "1mo", 5: "4-5mos"}
for t, d in df.groupby("timing"):
  fig, axes = plt.subplots(1, 2, figsize=(7,4), gridspec_kw=dict(width_ratios=[0.6, 1.7]))
  a = alpha[alpha["timing"] == t]
  for ax, m in zip(axes[:1], metrics):
    sns.boxplot(
      data=a, x="Group", y=m, order=list(group_labels.values()),
      hue="Group", hue_order=list(group_labels.values()), legend=False,
      palette={group_labels[g]: palette[g] for g in group_order},
      showfliers=False, width=0.6, boxprops=dict(alpha=0.6), ax=ax,
    )
    sns.stripplot(
      data=a, x="Group", y=m, order=list(group_labels.values()),
      hue="Group", hue_order=list(group_labels.values()), legend=False,
      palette={group_labels[g]: palette[g] for g in group_order},
      size=3, alpha=0.6, edgecolor="none", ax=ax,
    )
    p = alpha_stats[(alpha_stats["timing"] == timing_labels[t]) & (alpha_stats["metric"] == m)]["p"].iloc[0]
    ymax, span = a[m].max(), a[m].max() - a[m].min()
    ax.text(0.5, ymax + 0.12 * span, p_text(p), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(top=ymax + 0.25 * span)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelbottom=False)  # 群名は共通の凡例で示す
    ax.set_title(m, fontsize=10)
    ax.set_ylabel("")
  # beta
  ax = axes[1]
  xy, ev = pcoa_res[t]
  b = beta_stats[beta_stats["timing"] == timing_labels[t]].iloc[0]
  for grp in group_order:
    m = (d["sec-type"] == grp).to_numpy()
    ax.scatter(xy[m, 0], xy[m, 1], s=18, alpha=0.7, color=palette[grp],
               label=group_labels[grp], edgecolor="none")
    confidence_ellipse(xy[m], ax, palette[grp])
  ax.set_title(f"Bray-Curtis PCoA\nPERMANOVA R$^2$ = {b['R2']:.3f}, {p_text(b['p'])}", fontsize=10)
  ax.set_xlabel(f"PCoA1 ({ev[0] * 100:.1f}%)")
  ax.set_ylabel(f"PCoA2 ({ev[1] * 100:.1f}%)")
  fig.suptitle(timing_labels[t], fontsize=12, y=1.02)
  fig.tight_layout()
  # alpha・betaで共通の凡例を図の下中央に置く
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
             bbox_to_anchor=(0.5, 0.02), fontsize=10)
  fig.savefig(os.path.join(heredir, f"genus_diversity_{fig_names[t]}.png"), dpi=300, bbox_inches="tight")
  plt.show()

# %%
# 各Genusの sec vs non-sec 比較 (Mann-Whitney U検定、timingごと、多重検定補正なし)
res = []
for t, d in df.groupby("timing"):
  sec = (d["sec-type"] == "sec").to_numpy()
  for c in taxa:
    x = d[c].to_numpy()
    if (x > 0).mean() < PREV_MIN:
      continue
    a, b = x[sec], x[~sec]
    res.append(dict(
      timing=timing_labels[t], genus=names[c], col=c,
      median_sec=np.median(a), median_non=np.median(b),
      mean_sec=a.mean(), mean_non=b.mean(),
      prev_sec=(a > 0).mean(), prev_non=(b > 0).mean(),
      higher_in="sec" if a.mean() > b.mean() else "non-sec",
      p_mannWhitneyU=mannwhitneyu(a, b, alternative="two-sided").pvalue,
    ))
genus_all = pd.DataFrame(res)
genus_all["timing"] = pd.Categorical(genus_all["timing"], list(timing_labels.values()), ordered=True)
genus_all = genus_all.sort_values(["timing", "p_mannWhitneyU"])
genus_sig = genus_all[genus_all["p_mannWhitneyU"] < ALPHA]

alpha_stats.to_csv(os.path.join(heredir, "genus_alpha_sectype.csv"), index=False)
beta_stats.to_csv(os.path.join(heredir, "genus_beta_sectype_permanova.csv"), index=False)
genus_all.drop(columns="col").to_csv(os.path.join(heredir, "genus_sectype_all_tests.csv"), index=False)
genus_sig.drop(columns="col").to_csv(os.path.join(heredir, "genus_sectype_significant.csv"), index=False)

show_cols = ["genus", "median_sec", "median_non", "prev_sec", "prev_non", "higher_in", "p_mannWhitneyU"]
for t, r in genus_all.groupby("timing", observed=True):
  sig = r[r["p_mannWhitneyU"] < ALPHA]
  print(f"\n== {t}: tested {len(r)} genera, p<{ALPHA}: {len(sig)}")
  if len(sig):
    print(sig[show_cols].round(4).to_string(index=False))

# %%
# 有意差のあったGenusのstripplot (timingごとに1枚)
tcode = {v: k for k, v in timing_labels.items()}
for t, r in genus_sig.groupby("timing", observed=True):
  d = df[df["timing"] == tcode[t]].copy()
  d["Group"] = d["sec-type"].map(group_labels)
  ncol = min(len(r), 4)
  nrow = int(np.ceil(len(r) / ncol))
  fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 3.2 * nrow), squeeze=False)
  for ax, (_, row) in zip(axes.flat, r.iterrows()):
    sns.stripplot(
      data=d, x="Group", y=row["col"], order=list(group_labels.values()),
      hue="Group", hue_order=list(group_labels.values()), legend=False,
      palette={group_labels[g]: palette[g] for g in group_order},
      size=3, alpha=0.6, edgecolor="none", ax=ax,
    )
    for i, g in enumerate(group_order):  # 中央値
      med = np.median(d.loc[d["sec-type"] == g, row["col"]])
      ax.hlines(med, i - 0.3, i + 0.3, color="k", lw=1.5)
    ax.set_title(f"{row['genus']}\n{p_text(row['p_mannWhitneyU'])}", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("Relative abundance")
    ax.tick_params(axis="x", labelsize=8)
  for ax in axes.flat[len(r):]:
    ax.axis("off")
  fig.suptitle(f"{t} (Mann-Whitney U, uncorrected)", fontsize=11)
  fig.tight_layout()
  fig.savefig(os.path.join(heredir, f"genus_sectype_significant_{fig_names[tcode[t]]}.png"), dpi=300, bbox_inches="tight")
  plt.show()
# %%
