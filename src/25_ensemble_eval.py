"""Head-to-head: gradient-boosted trees vs neural nets vs ensembles, on the same blind forward-test cards (Standard).
Reads output/backtest_predictions.csv (trees, from 07_backtest.py) and output/nn/<model>_<year>.parquet.
Ensembles average within-year percentile ranks (all metrics are rank-based)."""
import sys, pathlib, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "output"
gb = pd.read_csv(OUT / "backtest_predictions.csv"); gb = gb[gb.format == "standard"].rename(columns={"pred": "gbdt"})
models = [m for m in ["mlp", "ft"] if list((OUT / "nn").glob(f"{m}_*.parquet"))]
df = gb[["name", "test_year", "rarity", "target", "gbdt"]].copy()
for m in models:
    parts = [pd.read_parquet(f).assign(test_year=int(f.stem.split("_")[1])) for f in (OUT / "nn").glob(f"{m}_*.parquet")]
    nn_ = pd.concat(parts); nn_ = nn_[nn_.format == "standard"][["name", "test_year", "pred"]].rename(columns={"pred": m})
    df = df.merge(nn_, on=["name", "test_year"], how="left")
years = sorted(set.intersection(*[set(df.dropna(subset=[m]).test_year) for m in models])) if models else sorted(df.test_year.unique())
df = df[df.test_year.isin(years)].copy()
for c in ["gbdt"] + models: df[c + "_r"] = df.groupby("test_year")[c].rank(pct=True)
cands = {"trees (gbdt)": df.gbdt_r}
for m in models: cands[m] = df[m + "_r"]; cands[f"trees + {m}"] = (df.gbdt_r + df[m + "_r"]) / 2
if len(models) == 2: cands["trees + mlp + ft"] = (df.gbdt_r + df.mlp_r + df.ft_r) / 3
rows = []
for nm, s in cands.items():
    d = df.assign(s=s); r = {"model": nm}
    r["spearman"] = np.mean([spearmanr(g.target, g.s).correlation for _, g in d.groupby("test_year")])
    r["spearman_rares"] = np.mean([spearmanr(g.target, g.s).correlation for _, g in d[d.rarity >= 2].groupby("test_year")])
    d["rk"] = d.groupby("test_year").s.rank(pct=True, ascending=False)
    r["top2%_staple"] = (d[d.rk <= .02].target >= .1).mean(); r["top5%_staple"] = (d[d.rk <= .05].target >= .1).mean()
    r["staples_in_top10%"] = ((d.target >= .1) & (d.rk <= .1)).sum() / (d.target >= .1).sum()
    rows.append(r)
R = pd.DataFrame(rows).set_index("model").round(3); print("years:", years, "| cards:", len(df)); print(R.to_string())
R.to_csv(OUT / "nn" / "head_to_head.csv")
