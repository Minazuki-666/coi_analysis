# %%
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
# %%
def load_permutation(path):
    # 末尾2行(空行とIntercept行)を除く
    df = pd.read_csv(path, sep="\t").iloc[:-2].astype("float")
    return df.sort_values("Correlation")

df_0 = load_permutation("./feces_newborn_permutation.txt")
df_1 = load_permutation("./feces_1mo_permutation.txt")
df_4 = load_permutation("./feces_45mos_permutation.txt")
df_0
# %%
def fit_through_point(x, y, x0, y0):
    """点 (x0, y0) を必ず通る最小二乗直線。傾きと切片(x=0)を返す。"""
    dx = np.asarray(x) - x0
    dy = np.asarray(y) - y0
    slope = np.sum(dx * dy) / np.sum(dx ** 2)
    intercept = y0 - slope * x0
    return slope, intercept

def plot_permutation(df, title=None):
    # 元モデル(Correlation == 1.0)と並べ替えモデルを分離
    is_orig = np.isclose(df["Correlation"], 1.0)
    df_perm = df[~is_orig]
    df_orig = df[is_orig]
    orig_vals = {"R2": df_orig["R2"].iloc[0], "Q2": df_orig["Q2"].iloc[0]}

    fig, ax = plt.subplots()
    x_line = np.array([0.0, 1.0])
    mako = sns.color_palette("mako", 2)
    colors = {"R2": mako[0], "Q2": mako[1]}

    for col in ["R2", "Q2"]:
        c = colors[col]
        y0 = orig_vals[col]
        labelColDict = {"R2": "R²Y", "Q2": "Q²"}
        ax.scatter(df_perm["Correlation"], df_perm[col], s=12, color=c, alpha=0.6)
        ax.scatter([1.0], [y0], s=60, color=c, edgecolor="k", zorder=3)

        slope, icpt = fit_through_point(df_perm["Correlation"], df_perm[col], 1.0, y0)
        ax.plot(x_line, icpt + slope * x_line, color=c, lw=1.5,
                label=f"{labelColDict[col]} (intercept = {icpt:.3f})")

    ax.set_title(title)
    ax.set_xlabel("Correlation between original and permuted Y")
    ax.set_ylabel("R²Y / Q²")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"permutation_{re.sub(r'\W+', '_', title).strip('_')}.svg", bbox_inches="tight")
    plt.close(fig)
    return fig

for df, title in [(df_0, "Newborn"), (df_1, "1 mo."), (df_4, "4-5 mos.")]:
    plot_permutation(df, title)
# %%
