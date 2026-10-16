# %%
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
# %%
df = pd.read_csv("data.csv")
HMOS = ["2'-fucosyllactose", "3-fucosyllactose"]
BINS = ["1.18", "1.22"]
LABEL = {"2'-fucosyllactose": "Breast milk 2'-fl", "3-fucosyllactose": "Breast milk 3-fl"}

# %% sec のみ、timing ごと + 全timingまとめ(All) の Spearman 相関
sec = df[df["sec"] == "sec"]
groups = [(str(t), g) for t, g in sec.groupby("timing")] + [("All", sec)]
rows = []
for timing, g in groups:
    for hmo in HMOS:
        for b in BINS:
            rho, p = spearmanr(g[hmo], g[b])
            rows.append({"timing": timing, "hmo": hmo, "bin": b, "n": len(g), "rho": rho, "p": p})
res = pd.DataFrame(rows)

# 多重検定補正: 全16検定をまとめて Benjamini-Hochberg (FDR)
res["q"] = multipletests(res["p"], method="fdr_bh")[1]
res
# %%
res.to_csv("spearman_sec.csv", index=False)
# %% ヒートマップ (timing ごとに別パネル, 各パネル: 行=HMO, 列=bin)
TIMING_LABEL = {"3": "Newborn", "4": "1 mo.", "5": "4-5 mos", "All": "All"}
timings = ["3", "4", "5", "All"]
n_by_timing = res.groupby("timing")["n"].first()
res["row"] = res["hmo"].map(LABEL)
res["col"] = "Feces " + res["bin"]
COLS = [f"Feces {b}" for b in ["1.22", "1.18"]]


def fmt_q(q):
    # q < 0.05 のセルのみ q 値を表記 (q < 0.001 は "q<0.001")
    return "q<0.001" if q < 0.001 else f"q={q:.2g}" if q < 0.05 else ""


fig, axes = plt.subplots(
    1, len(timings) + 1, figsize=(11, 2.8),
    gridspec_kw={"width_ratios": [1] * len(timings) + [0.05], "wspace": 0.2},
)
cax = axes[-1]
for i, (ax, t) in enumerate(zip(axes[:-1], timings)):
    d = res[res["timing"] == t]
    rho_mat = d.pivot(index="row", columns="col", values="rho").loc[[LABEL[h] for h in HMOS], COLS]
    q_mat = d.pivot(index="row", columns="col", values="q").loc[[LABEL[h] for h in HMOS], COLS]
    annot = rho_mat.round(2).astype(str) + "\n" + q_mat.map(fmt_q)
    sns.heatmap(
        rho_mat, annot=annot, fmt="", cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
        linewidths=0.5, linecolor="white", ax=ax,
        cbar=(i == 0), cbar_ax=cax if i == 0 else None, cbar_kws={"label": "Spearman ρ"},
    )
    ax.set_title(f"{TIMING_LABEL[t]} (n={n_by_timing[t]})")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    if i == 0:
        plt.setp(ax.get_yticklabels(), rotation=0)
    else:
        ax.set_yticks([])
fig.suptitle("Spearman correlation (sec only), FDR (BH) q shown if q < 0.05", y=1.08)
fig.savefig("spearman_sec_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
