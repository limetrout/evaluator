"""Score every card in the post-rotation Standard pool (Feb 2027: Foundations + Aetherdrift ... Star Trek + Nauctis).

For each card: model prediction (as if it were brand new, in Standard), the current observed Standard play
rate (last 3 months of MTGO data, where it exists), and the top reasons behind the prediction.
Re-run after downloading fresh Scryfall data to score newly spoiled Nauctis cards.
"""
import json, gzip, pathlib, numpy as np, pandas as pd, lightgbm as lgb
import importlib.util
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT, RAW = ROOT / "data/processed", ROOT / "output", ROOT / "data/raw"
spec = importlib.util.spec_from_file_location("train", ROOT / "src/05_train.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

POOL_SETS = ["fdn", "dft", "tdm", "fin", "eoe", "spm", "tla", "ecl", "tmt", "sos", "msh", "hob", "fra", "trk", "nau"]
in_pool, set_of = set(), {}
for l in gzip.open(RAW / "default_cards.jsonl.gz", "rt"):
    p = json.loads(l)
    if p["set"] in POOL_SETS and p.get("oracle_id") and not p.get("promo"):
        in_pool.add(p["oracle_id"]); set_of.setdefault(p["oracle_id"], p["set"].upper())
feat = pd.read_parquet(PROC / "features.parquet")
pool = feat[feat.oracle_id.isin(in_pool) & ~feat.type_line.str.startswith("Basic")].copy()
pool["format"] = "standard"; pool["window_start"] = "2027-02"; pool["target"] = 0.0; pool["y"] = 0.0; pool["ever_played"] = 0; pool["n_months"] = 0; pool["legacy_card"] = 0
for f in T.FORMATS: pool[f"fmt_{f}"] = int(f == "standard")
pool["window_year"] = 2027 + 2 / 12; pool["release_year"] = pool.release_year.clip(lower=2024)   # treat as "current era" cards
pool["doc"] = pool.type_line.str.lower() + " | " + pool.clean_text
pool = pool.reset_index(drop=True)

pred, text_pred, gbm, _ = T.fit_predict(T.df, pool)          # final model: trained on ALL history
pool["model_score"] = np.expm1(pred).clip(min=0).round(2)    # ~ predicted % of top Standard decks
pool["text_score"] = np.expm1(text_pred).clip(min=0).round(2)
pool["pool_set"] = pool.oracle_id.map(set_of)
pool["percentile"] = (pool.model_score.rank(pct=True) * 100).round(1)

# observed play in current Standard (last 3 months with data)
mr = pd.read_parquet(PROC / "monthly_rates.parquet"); st = mr[mr.format == "standard"]
last = sorted(st.month.unique())[-3:]
obs = st[st.month.isin(last)].groupby("oracle_id").rate.sum() / 3
pool["current_std_play_pct"] = (pool.oracle_id.map(obs).fillna(0) * 100).round(1)

# explanations: per-card feature contributions from the tree model
X = pool[T.FEATS].assign(text_score=text_pred)
contrib = gbm.predict(X, pred_contrib=True)[:, :-1]
names = list(X.columns)
def why(i):
    c = contrib[i]; order = np.argsort(-np.abs(c))[:5]
    return "; ".join(f"{'+' if c[j] > 0 else '-'}{names[j]}" for j in order)
pool["top_reasons"] = [why(i) for i in range(len(pool))]
cols = ["name", "pool_set", "type_line", "mv", "rarity", "model_score", "percentile", "current_std_play_pct", "text_score", "top_reasons"]
res = pool.sort_values("model_score", ascending=False)[cols]
res.to_csv(OUT / "post_rotation_scores.csv", index=False)
print(res.head(60).to_string(index=False))
print("\nNauctis spoilers:\n", res[res.pool_set == "NAU"].to_string(index=False))
