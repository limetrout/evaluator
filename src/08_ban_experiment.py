"""Does giving banned cards extra weight help? Rolling 2018-2023 blind backtest under three settings."""
import importlib.util, pathlib, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train", ROOT / "src/05_train.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
df = T.df
print("banned cards in training data:", df[df.oracle_id.isin(T.BANNED)].oracle_id.nunique(), "| as Standard rows:", ((df.format == "standard") & df.oracle_id.isin(T.BANNED)).sum(), flush=True)
rows = []
for name, w, floor in [("as now", 1.0, None), ("banned x3 weight", 3.0, None), ("banned x3 + score floor 30%", 3.0, 0.30)]:
    T.BAN_WEIGHT, T.BAN_FLOOR = w, floor; preds = []
    for y in range(2018, 2024):
        lo, hi = f"{y}-01", f"{y + 1}-01"
        tr = df[df.window_start < lo].reset_index(drop=True); te = df[(df.window_start >= lo) & (df.window_start < hi) & (df.format == "standard")].reset_index(drop=True)
        p, *_ = T.fit_predict(tr, te); preds.append(te.assign(pred=p, yr=y))
    st = pd.concat(preds, ignore_index=True); st["r"] = st.groupby("yr").pred.rank(pct=True, ascending=False)
    rows.append({"setting": name,
        "ranking agreement (avg of years)": round(np.mean([spearmanr(g.target, g.pred).correlation for _, g in st.groupby("yr")]), 3),
        "rares/mythics agreement": round(np.mean([spearmanr(g.target, g.pred).correlation for _, g in st[st.rarity >= 2].groupby("yr")]), 3),
        "top 2%: became staple": round((st[st.r <= .02].target >= .1).mean(), 3),
        "top 5%: became staple": round((st[st.r <= .05].target >= .1).mean(), 3),
        "staples caught in top 10%": round(((st.target >= .1) & (st.r <= .1)).sum() / (st.target >= .1).sum(), 3),
        "biggest hits (25%+) caught in top 10%": round(((st.target >= .25) & (st.r <= .1)).sum() / (st.target >= .25).sum(), 3)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).set_index("setting").T.to_csv(ROOT / "output/ban_weight_experiment.csv")
print(pd.DataFrame(rows).set_index("setting").T.to_string())
