# %%
"""hbm_profile.csv の各成分濃度について、論文用の表を2つ csv に出力する (どちらも sec / non-sec, 時期を分けずに全サンプルを使う)。

- 時期での比較: 時期ごとの 平均 ± SD と、同一被験者(family)の対応のある t 検定による時期間の個別比較
- 分泌型での比較: sec / non-sec の 平均 ± SD と、Welch の t 検定
どちらも多重検定補正は Benjamini-Hochberg (FDR)。
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests

heredir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(os.path.dirname(heredir)), "data")
df = pd.read_csv(os.path.join(datadir, "hbm_profile.csv"))

TIMING_LABEL = {3: "Newborn", 4: "1 mo.", 5: "4-5 mos"}
SEC_ORDER = ["sec", "non-sec"]
conc_cols = list(df.columns[df.columns.get_loc("2'-fucosyllactose"):])  # 濃度の列


def fmt_mean_sd(x):
    return f"{x.mean():.3g} ± {x.std():.3g}"  # SD は不偏標準偏差 (ddof=1)


def fmt_p(p):
    return "NA" if np.isnan(p) else "<0.001" if p < 0.001 else f"{p:.3f}"


def bh(p):
    """NaN を除いて BH-FDR の q 値を返す"""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


# %% 時期での比較: 全サンプル(sec + non-sec)で、時期ごとの 平均 ± SD と、対応のある t 検定による時期間比較
PAIRS = list(itertools.combinations(TIMING_LABEL, 2))  # (3,4), (3,5), (4,5)
# 同じ family x timing の重複行 (1例) は平均して1行にする
wide = df.groupby(["family", "timing"])[conc_cols].mean().unstack("timing")
rows = []
for t1, t2 in PAIRS:
    both = wide.xs(t1, axis=1, level="timing").dropna().index.intersection(
        wide.xs(t2, axis=1, level="timing").dropna().index)  # 2時期が揃った被験者
    for c in conc_cols:
        res = ttest_rel(wide.loc[both, (c, t1)], wide.loc[both, (c, t2)])
        rows.append({"metabolite": c, "pair": (t1, t2), "n_pairs": len(both), "p": res.pvalue})
tk = pd.DataFrame(rows)
tk["q"] = bh(tk["p"])  # 51成分 x 3比較 = 153 個の p をまとめて BH

table_timing = pd.DataFrame(index=pd.Index(conc_cols, name="metabolite"))
for t, label in TIMING_LABEL.items():
    x = df[df["timing"] == t]
    table_timing[f"{label} (n={len(x)})"] = x[conc_cols].apply(fmt_mean_sd)
for t1, t2 in PAIRS:
    g = tk[tk["pair"] == (t1, t2)].set_index("metabolite")
    table_timing[f"q: {TIMING_LABEL[t1]} vs {TIMING_LABEL[t2]} (n={g['n_pairs'].iloc[0]} pairs)"] = g["q"].map(fmt_p)
table_timing = table_timing.reset_index().rename(columns={"metabolite": "Metabolite"})
table_timing

# %%
table_timing.to_csv(os.path.join(heredir, "hbm_table_compare_timing.csv"), index=False, encoding="utf-8-sig")

# %% 分泌型での比較: 全時期のサンプルをまとめて、sec / non-sec の 平均 ± SD と Welch の t 検定
a, b = df[df["sec"] == "sec"], df[df["sec"] == "non-sec"]
p = np.array([ttest_ind(a[c], b[c], equal_var=False).pvalue for c in conc_cols])  # Welch
table_sec = pd.DataFrame(index=pd.Index(conc_cols, name="metabolite"))
table_sec[f"sec (n={len(a)})"] = a[conc_cols].apply(fmt_mean_sd)
table_sec[f"non-sec (n={len(b)})"] = b[conc_cols].apply(fmt_mean_sd)
table_sec["p"] = pd.Series(p, index=conc_cols).map(fmt_p)
table_sec["q"] = pd.Series(bh(p), index=conc_cols).map(fmt_p)  # 51成分で BH
table_sec = table_sec.reset_index().rename(columns={"metabolite": "Metabolite"})
table_sec

# %%
table_sec.to_csv(os.path.join(heredir, "hbm_table_compare_sec.csv"), index=False, encoding="utf-8-sig")
