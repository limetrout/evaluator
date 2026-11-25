"""Rolling ("walk-forward") backtest: for each year Y, train on everything known before Y, predict cards that
debuted in year Y, then pool all blind predictions. Every card from 2018 on is scored exactly once, blind."""
import importlib.util, pathlib, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train", ROOT / "src/05_train.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
df = T.df; preds = []
for y in range(2018, 2025):
    lo, hi = f"{y}-01", f"{y + 1}-01"
    train = df[df.window_start < lo].reset_index(drop=True); test = df[(df.window_start >= lo) & (df.window_start < hi)].reset_index(drop=True)
    if len(test) == 0 or (test.format == "standard").sum() < 50: continue
    p, *_ = T.fit_predict(train, test)
    test = test.assign(pred=p, test_year=y); preds.append(test)
    r = T.report(str(y), test, p); print(y, {k: v for k, v in r.items() if k.startswith("standard")}, flush=True)
    b = __import__("lightgbm").train(T.PARAMS, __import__("lightgbm").Dataset(train[["rarity", "mv"] + [f"fmt_{f}" for f in T.FORMATS]], train.y), 300)
    preds[-1]["base"] = b.predict(test[["rarity", "mv"] + [f"fmt_{f}" for f in T.FORMATS]])
allp = pd.concat(preds, ignore_index=True)
allp[["name", "first_set", "format", "test_year", "rarity", "mv", "target", "pred", "base"]].to_csv(ROOT / "output/backtest_predictions.csv", index=False)
rows = []
for fmt in ["standard", "pioneer", "modern"]:
    for nm, col in [("model", "pred"), ("rarity+mv baseline", "base")]:
        r = T.report(nm, allp[allp.format == fmt].reset_index(drop=True), allp[allp.format == fmt][col].values); rows.append(r)
    # per-year ranking within each test year (ranks are only comparable inside a year)
st = allp[allp.format == "standard"].copy()
st["pct_rank"] = st.groupby("test_year").pred.rank(pct=True, ascending=False)
bins = [0, .02, .05, .10, .25, .50, 1]
st["bucket"] = pd.cut(st.pct_rank, bins, labels=["top 2%", "2-5%", "5-10%", "10-25%", "25-50%", "bottom 50%"])
tab = st.groupby("bucket", observed=True).agg(cards=("name", "size"), saw_play=("target", lambda t: (t >= .02).mean()), staple=("target", lambda t: (t >= .10).mean()))
print("\nSTANDARD, all years pooled:", len(st), "cards,", int((st.target >= .02).sum()), "saw play,", int((st.target >= .10).sum()), "staples")
print(tab.to_string())
print("\nshare of all staples found in model's top 10% each year:", round(((st.target >= .1) & (st.pct_rank <= .1)).sum() / (st.target >= .1).sum(), 2))
print("share of played cards the model put in its bottom half:", round(((st.target >= .02) & (st.pct_rank > .5)).sum() / (st.target >= .02).sum(), 2))
for y, g in st.groupby("test_year"): print(y, "spearman", round(spearmanr(g.target, g.pred).correlation, 2), "baseline", round(spearmanr(g.target, g.base).correlation, 2), "cards", len(g))
print(pd.DataFrame(rows).set_index("model").T.to_string())
tab.to_csv(ROOT / "output/backtest_buckets.csv")
