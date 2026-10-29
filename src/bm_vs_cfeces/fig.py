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

SEC_TYPES = ["sec", "non-sec"]
df_corrs = {sec: calc_corr(df_merge[df_merge["sec"] == sec], PAIRS) for sec in SEC_TYPES}
df_corr_all = pd.concat(df_corrs, names=["sec"]).reset_index("sec")
def source_label(v):
  return f"Infant feces bin {v}" if v in BINS else f"Breast milk {v}"

df_corr_all.assign(
  timing=df_corr_all["timing"].map(TIMING_NAMES),
  x=df_corr_all["x"].map(source_label),
  y=df_corr_all["y"].map(source_label),
).to_csv("corr.csv", index=False)
df_corrs
#%%
def get_corr(df, x, y, timing):
  return df[(df["timing"] == timing) & (df["x"] == x) & (df["y"] == y)].iloc[0]

def plot_scatter(df, df_corr, x, y, title, fname, timing=4):
  r = get_corr(df_corr, x, y, timing)
  fig, ax = plt.subplots()
  sns.scatterplot(
    data=df[df["timing"] == timing],
    x=x,
    y=y,
    color="gray",
    ax=ax
  )
  ax.text(
    0.05, 0.95, f"ρ = {r['rho']:.2f}\n" + ("q < 0.001" if r["q"] < 0.001 else f"q = {r['q']:.3f}"),
    transform=ax.transAxes, va="top",
  )
  ax.set_xlabel(f"Breast milk {x} (mM)")
  ax.set_ylabel(f"Infant feces bin({y})")
  ax.set_title(title)
  fig.savefig(fname, bbox_inches="tight")
  plt.show()
#%%
for sec in SEC_TYPES:
  df_s = df_merge[df_merge["sec"] == sec]
  n = (df_s["timing"] == 4).sum()
  title = f"{sec}, at 1 mo. (n={n})"
  plot_scatter(df_s, df_corrs[sec], "2'-FL", "1.22", title, f"scatter_2FL_1.22_{sec}.svg")
  plot_scatter(df_s, df_corrs[sec], "3-FL", "1.18", title, f"scatter_3FL_1.18_{sec}.svg")
#%%
def pair_label(x, y):
  if x in BINS:
    return f"Feces {x} bin\nvs Feces {y} bin"
  return f"Breast milk {x}\nvs Feces {y} bin"

def plot_forest(df, sec, groups=FOREST_GROUPS, q_th=0.05):
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
  ax.set_title(f"Correlation between breast milk FL and infant feces bins ({sec})")
  ax.spines[["top", "right"]].set_visible(False)
  ax.legend(
    handles=[
      plt.Line2D([], [], color=c, lw=1.8, label=TIMING_NAMES.get(t, f"Timing {t}"))
      for t, c in zip(timings, colors)
    ],
  )
  plt.tight_layout()
  fig.savefig(f"forest_{sec}.svg", bbox_inches="tight")
  plt.show()
#%%
for sec in SEC_TYPES:
  plot_forest(df_corrs[sec], sec)

#%%
#%%
def calc_ratio_corr(df):
  """timing ごとの log(feces 1.18/1.22) と log(BM 3-FL/2'-FL) の Spearman 相関"""
  df = df.assign(
    log_feces=np.log(df["1.18"] / df["1.22"]),
    log_bm=np.log(df["3-FL"] / df["2'-FL"]),
  )
  results = []
  for timing, df_t in df.groupby("timing"):
    d = df_t[["log_bm", "log_feces"]].dropna()
    rho, p = spearmanr(d["log_bm"], d["log_feces"])
    se = np.sqrt(1.06 / (len(d) - 3))
    z = np.arctanh(rho)
    results.append({
      "timing": TIMING_NAMES.get(timing, timing), "n": len(d), "rho": rho, "p": p,
      "lo": np.tanh(z - 1.96 * se), "hi": np.tanh(z + 1.96 * se),
    })
  return df, pd.DataFrame(results)

ratio_results = {sec: calc_ratio_corr(df_merge[df_merge["sec"] == sec]) for sec in SEC_TYPES}
for sec, (_, df_rc) in ratio_results.items():
  print(sec)
  print(df_rc)
pd.concat({sec: df_rc for sec, (_, df_rc) in ratio_results.items()}, names=["sec"]).reset_index("sec").to_csv("ratio_corr.csv", index=False)
#%%
def plot_ratio_scatter(df, df_corr, sec):
  fig, axes = plt.subplots(1, len(df_corr), figsize=(4 * len(df_corr), 4), sharex=True, sharey=True)
  for ax, (timing, name) in zip(axes, TIMING_NAMES.items()):
    r = df_corr[df_corr["timing"] == name].iloc[0]
    sns.scatterplot(data=df[df["timing"] == timing], x="log_bm", y="log_feces", color="gray", ax=ax)
    ax.text(0.05, 0.95, f"ρ = {r['rho']:.2f}\n" + ("p < 0.001" if r["p"] < 0.001 else f"p = {r['p']:.3f}"),
            transform=ax.transAxes, va="top")
    ax.set_title(f"{sec}, {name} (n={r['n']})")
    ax.set_xlabel("Breast milk log(3-FL / 2'-FL)")
    ax.set_ylabel("Infant feces log(1.18 / 1.22)")
  plt.tight_layout()
  fig.savefig(f"ratio_scatter_{sec}.svg", bbox_inches="tight")
  plt.show()

for sec, (df_r, df_rc) in ratio_results.items():
  plot_ratio_scatter(df_r, df_rc, sec)

# %%
