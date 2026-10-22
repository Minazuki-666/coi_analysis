# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, false_discovery_control

sns.set_theme(style="whitegrid")

BINS = ("1.18", "1.22")
PAIRS = [
  ("2'-FL", "1.18"),
  ("2'-FL", "1.22"),
  ("3-FL", "1.18"),
  ("3-FL", "1.22"),
  ("1.18", "1.22"),
]
FOREST_GROUPS = [
  ("Own molecule", [("3-FL", "1.18"), ("2'-FL", "1.22")]),
  ("Other molecule", [("2'-FL", "1.18"), ("3-FL", "1.22")]),
  ("Between bins", [("1.18", "1.22")]),
]
TIMING_NAMES = {3: "Newborn", 4: "1 mo.", 5: "4-5 mos."}
# %%
def make_key(df):
  return df["family"].astype(str) + "_" + df["timing"].astype(str)

df_bm = pd.read_csv("../../data/hbm_profile.csv")
df_bm = df_bm[df_bm["sec"] == "sec"]
df_bm["key"] = make_key(df_bm)
df_bm = df_bm.reindex(columns=["key", "timing", "sec", "2'-fucosyllactose", "3-fucosyllactose"])
df_bm
#%%
df_cfeces = pd.read_csv("../../data/feces_BinningResults.csv", header=0)
df_cfeces = df_cfeces[df_cfeces["age"] == "c"]
df_cfeces["key"] = make_key(df_cfeces)
df_cfeces = df_cfeces.reindex(columns=df_cfeces.columns[6:][::-1])
df_cfeces
# %%
df_merge = pd.merge(df_bm, df_cfeces, on="key").rename(
  columns={
    "2'-fucosyllactose": "2'-FL",
    "3-fucosyllactose": "3-FL",
  },
)
df_merge
#%%
def calc_corr(df, pairs):
  """timing ごとの Spearman 相関 (BH補正q値, Fisher z変換による95%CI付き)"""
  results = []
  for timing, df_t in df.groupby("timing"):
    for x, y in pairs:
      d = df_t[[x, y]].dropna()
      rho, p = spearmanr(d[x], d[y])
      results.append({"timing": timing, "x": x, "y": y, "n": len(d), "rho": rho, "p": p})
  df_corr = pd.DataFrame(results)
  df_corr["q"] = df_corr.groupby("timing")["p"].transform(
    lambda p: false_discovery_control(p, method="bh")
  )
  # Bonett-Wright の分散 1.06/(n-3)
  z = np.arctanh(df_corr["rho"])
  se = np.sqrt(1.06 / (df_corr["n"] - 3))
  df_corr["lo"] = np.tanh(z - 1.96 * se)
  df_corr["hi"] = np.tanh(z + 1.96 * se)
  return df_corr

df_corr = calc_corr(df_merge, PAIRS)
df_corr
#%%
def get_corr(df, x, y, timing):
  return df[(df["timing"] == timing) & (df["x"] == x) & (df["y"] == y)].iloc[0]

def plot_scatter(x, y, title, timing=4):
  r = get_corr(df_corr, x, y, timing)
  fig, ax = plt.subplots()
  sns.scatterplot(
    data=df_merge[df_merge["timing"] == timing],
    x=x,
    y=y,
    color="gray",
    ax=ax
  )
  ax.text(
    0.05, 0.95, f"ρ = {r['rho']:.2f}\nq = {r['q']:.3f}",
    transform=ax.transAxes, va="top",
  )
  ax.set_xlabel(f"Breast milk {x} (mM)")
  ax.set_ylabel(f"Infant feces bin({y}) (mM)")
  ax.set_title(title)
  plt.show()
#%%
plot_scatter("2'-FL", "1.22", "At 1 mo. (n=64)")
plot_scatter("3-FL", "1.18", "At 1 mo. (n=64)")
#%%
def pair_label(x, y):
  if x in BINS:
    return f"Feces {x} bin\nvs Feces {y} bin"
  return f"Breast milk {x}\nvs Feces {y} bin"

def plot_forest(df, groups=FOREST_GROUPS, q_th=0.05):
  timings = sorted(df["timing"].unique())
  colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(timings)))
  offsets = np.linspace(0.25, -0.25, len(timings))

  fig, ax = plt.subplots(figsize=(6.4, 7))
  yticks, ylabels, y = [], [], 0
  for gname, gpairs in groups:
    ax.text(0.01, y + 0.2, gname, style="italic", color="dimgray",
            transform=ax.get_yaxis_transform(), va="bottom")
    y -= 1.0
    marker = "D" if gname == "Between bins" else "o"
    for x, yy in gpairs:
      for t, c, o in zip(timings, colors, offsets):
        r = get_corr(df, x, yy, t)
        ax.plot([r["lo"], r["hi"]], [y + o] * 2, color=c, lw=1.8)
        ax.plot(r["rho"], y + o, marker, color=c,
                mfc=c if r["q"] < q_th else "white", mew=1.5, ms=6)
      yticks.append(y)
      ylabels.append(pair_label(x, yy))
      y -= 1.3
    y -= 0.6
  ax.axvline(0, color="gray", lw=1)
  ax.set_ylim(min(yticks) - 0.8, 1.2)
  ax.set_yticks(yticks)
  ax.set_yticklabels(ylabels)
  ax.set_xlabel(f"Spearman ρ (95% CI; filled: q < {q_th})")
  ax.set_title("Correlation between breast milk FL and infant feces bins")
  ax.spines[["top", "right"]].set_visible(False)
  ax.legend(
    handles=[
      plt.Line2D([], [], color=c, lw=1.8, label=TIMING_NAMES.get(t, f"Timing {t}"))
      for t, c in zip(timings, colors)
    ],
  )
  plt.tight_layout()
  plt.show()
#%%
plot_forest(df_corr)

# %%
