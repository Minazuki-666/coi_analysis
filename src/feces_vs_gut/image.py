"""
便 NMR ビン値 × 腸内細菌 (CLR) のスピアマン相関を泌乳期ごとに計算し、ヒートマップ状に描画

- 対象: data/feces_BinningResults_genus.csv の age == "c" (児) のみ
- 属: Bifidobacterium_CLR, Bacteroides_H_CLR
- 時期 (timing): 3=Newborn, 4=1 mo., 5=4-5 mos. ごとに各ビンとのスピアマン相関を計算
- 多重検定: 属 × 時期ごとに全ビンで BH-FDR 補正 (q)
- 描画: q < 0.05 のビンのみ ρ で着色 (それ以外は白)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "feces_BinningResults_genus.csv"
OUT_DIR = HERE / "fig"
OUT_DIR.mkdir(exist_ok=True)

TIMING_LABEL = {3: "Newborn", 4: "1 mo.", 5: "4–5 mos."}
GENERA = [("Bifidobacterium_CLR", "Bifidobacterium"), ("Bacteroides_H_CLR", "Bacteroides")]
Q_THRESHOLD = 0.05
BIN_WIDTH = 0.04

df = pd.read_csv(DATA)
df = df[df["age"] == "c"]
bin_cols = [c for c in df.columns[6:] if not c.endswith("_CLR")]  # 先頭 6 列はメタ情報、続く *_CLR 列は属の CLR
ppm = np.array([float(c) for c in bin_cols])

# ---- 相関計算 ----
rows = []
for clr_col, genus in GENERA:
    for timing, tlabel in TIMING_LABEL.items():
        sub = df[df["timing"] == timing].dropna(subset=[clr_col])
        rho = np.full(len(bin_cols), np.nan)
        p = np.full(len(bin_cols), np.nan)
        for i, b in enumerate(bin_cols):
            x = sub[b]
            if x.nunique() < 2:
                continue
            rho[i], p[i] = spearmanr(sub[clr_col], x)
        ok = ~np.isnan(p)
        q = np.full(len(bin_cols), np.nan)
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
        rows.append(dict(genus=genus, timing=timing, label=tlabel, n=len(sub), rho=rho, p=p, q=q))

res = pd.DataFrame(rows)

# CSV 保存 (long 形式)
long = pd.concat(
    [pd.DataFrame({"genus": r.genus, "timing": r.timing, "n": r.n, "bin_ppm": ppm,
                   "rho": r.rho, "p": r.p, "q": r.q}) for r in res.itertuples()]
)
long.to_csv(HERE / "genus_bin_spearman.csv", index=False)

# ---- 描画 ----
fig, ax = plt.subplots(figsize=(6.6, 3.5))
cmap = plt.get_cmap("RdBu_r")
vmax = 0.6
norm = Normalize(-vmax, vmax)

n_rows = len(res)
for ri, r in enumerate(res.itertuples()):
    y0 = n_rows - ri - 1  # 上から順に並べる
    sig = np.nan_to_num(r.q, nan=1.0) < Q_THRESHOLD
    for x, rho in zip(ppm[sig], r.rho[sig]):
        ax.add_patch(Rectangle((x - BIN_WIDTH / 2, y0), BIN_WIDTH, 1,
                                   facecolor=cmap(norm(rho)), edgecolor="none", linewidth=0))

ax.set_xlim(ppm.max() + 0.1, ppm.min() - 0.1)  # ppm は右が小さい
ax.set_ylim(0, n_rows)
ax.set_yticks(np.arange(n_rows) + 0.5)
ax.set_yticklabels([f"{r.label} ({int((np.nan_to_num(r.q, nan=1.0) < Q_THRESHOLD).sum())})"
                    for r in res.itertuples()][::-1])
ax.set_xlabel("¹H chemical shift (ppm)")
ax.axhline(len(TIMING_LABEL), color="black", linewidth=1)
for s in ax.spines.values():
    s.set_linewidth(1)

# 属名 (左側に縦書き, 斜体)
for gi, (_, genus) in enumerate(GENERA):
    yc = n_rows - (gi * len(TIMING_LABEL) + len(TIMING_LABEL) / 2)
    ax.text(-0.34, yc / n_rows, genus, transform=ax.transAxes, rotation=90,
            ha="center", va="center", style="italic", fontsize=9)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
cb.set_label("Spearman ρ (q < 0.05)")

fig.savefig(OUT_DIR / "genus_bin_spearman.svg", bbox_inches="tight")
fig.savefig(OUT_DIR / "genus_bin_spearman.png", dpi=200, bbox_inches="tight")
print(res[["genus", "label", "n"]].assign(n_sig=[int((np.nan_to_num(r.q, nan=1.0) < Q_THRESHOLD).sum()) for r in res.itertuples()]))
