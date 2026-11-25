"""Final synergy models + an extra forward check on 2025-2026 Standard (never used in training).

Trains the per-card ranker and the strong-partner probability model on ALL focal years (2017-2024), saves them,
then: for every card released 2025+ that appears in >= 5 recent Standard decks, rank all recently played Standard
cards and check whether its real recent partners (lift >= 2 in >= 4 decks) land in its top 10.
"""
import sys, pathlib, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"
spec = importlib.util.spec_from_file_location("r14", ROOT / "src/14_synergy_ranker.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
S, M = R.S, R.M
(OUT / "models").mkdir(exist_ok=True)

# ---- train final models on all focal years
tr = []
rng = np.random.default_rng(0)
for Yp in R.YEARS:
    d = R.build_year(Yp); l = R.labels(d)
    keep = (l > 0) | (rng.random(len(d)) < 0.3); tr.append(d[keep].assign(lab=l[keep]))
tr = pd.concat(tr, ignore_index=True).sort_values(["fy", "format", "focal"])
FEATS = [c for c in tr.columns if c not in R.META + ["lab"]]
grp = tr.groupby(["fy", "format", "focal"], sort=False).size().values
rank_params = dict(R.RANK_PARAMS); rank_params["objective"] = "lambdarank"
ranker = lgb.train({**rank_params, "metric": "ndcg", "ndcg_eval_at": [10]}, lgb.Dataset(tr[FEATS], tr.lab, group=grp), 400)
binp = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
prob = lgb.train(binp, lgb.Dataset(tr[FEATS], (tr.lab == 2).astype(int)), 400)
ranker.save_model(str(OUT / "models/synergy_ranker.txt")); prob.save_model(str(OUT / "models/synergy_prob.txt"))
pd.Series(FEATS).to_csv(OUT / "models/synergy_features.csv", index=False)

# ---- feature builder for "now" (map from all data, popularity from the last 12 months of Standard)
REAL = S.card_map("2099-01-01")
dk = S.decks[(S.decks.format == "standard") & (S.decks.date >= "2025-01-01")]
NREC = dk.deck.nunique(); deck_sets = dk.groupby("oracle_id").deck.agg(set)
last12 = S.decks[(S.decks.format == "standard") & (S.decks.date >= (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime("%Y-%m-%d"))]
POP_NOW = last12.groupby("oracle_id").deck.nunique() / max(1, last12.deck.nunique())
def features_now(A, B, need, real=None, popmap=None):
    real = REAL if real is None else real; popmap = POP_NOW if popmap is None else popmap
    cross, pred, _ = S.text_to_map(real, sorted(set(need) - set(real)))
    X = M.pair_features(A, B, np.full(len(A), "standard"), np.full(len(A), 2026))
    F = S.map_feats(A, B, real, cross, pred, False, S.RELD)
    X = pd.concat([X, F], axis=1); X["cand_pop_prev"] = pd.Series(B).map(popmap).fillna(0).values
    return X[FEATS]
if __name__ == "__main__":
    # ---- forward check 2025-2026
    NONLAND = set(M.feat.index[M.feat.type_land == 0])           # same as training: non-land cards only
    played = [o for o, s in deck_sets.items() if len(s) >= 5 and o in NONLAND]
    rel = M.cards.first_release.reindex(played)
    new = [o for o in played if isinstance(rel.get(o), str) and rel[o] >= "2025-01-01"]
    print(f"recent Standard decks: {NREC} | played cards: {len(played)} | new (2025+) focal cards: {len(new)}", flush=True)
    def truth(a, b):
        A_, B_ = deck_sets.get(a, set()), deck_sets.get(b, set()); n = len(A_ & B_); E = len(A_) * len(B_) / NREC
        return n >= 4 and (n + 1) / (E + 1) >= 2
    A = np.repeat(new, len(played)); B = np.tile(played, len(new)); k = A != B; A, B = A[k], B[k]
    # HONEST forward check: card map from decks BEFORE 2025 only, popularity from 2024 -> 2025+ cards are placed
    # purely from their text, exactly like a future card would be.
    REAL_PRE = S.card_map("2025-01-01")
    d24 = S.decks[(S.decks.format == "standard") & (S.decks.date >= "2024-01-01") & (S.decks.date < "2025-01-01")]
    POP_PRE = d24.groupby("oracle_id").deck.nunique() / max(1, d24.deck.nunique())
    X = features_now(A, B, played, REAL_PRE, POP_PRE)
    df = pd.DataFrame({"a": A, "b": B, "rank": ranker.predict(X), "prob": prob.predict(X), "map": X.map_cos.fillna(-1).values,
                       "pop": X.cand_pop_prev.values})
    df["t"] = [truth(a, b) for a, b in zip(df.a, df.b)]
    out = {}
    for col in ["rank", "prob", "map", "pop"]:
        rec = []
        for a, g in df.groupby("a"):
            t = g.t.sum()
            if t == 0: continue
            rec.append(g.nlargest(10, col).t.sum() / min(10, t))
        out[col] = (round(float(np.mean(rec)), 3), len(rec))
    print("2025-26 forward check, real partners found in top 10 (recall, cards graded):", out)
    pd.Series({k: v[0] for k, v in out.items()}).to_csv(OUT / "synergy_forward_check_2025_26.csv")
    # ---- spot checks
    name = M.cards.name
    for focal in ["Badgermole Cub", "It'll Quench Ya!", "Mossborn Hydra", "Momo, Friendly Flier"]:
        o = M.cards.index[name == focal]
        if len(o) == 0 or o[0] not in set(df.a): print("skip", focal); continue
        g = df[df.a == o[0]].nlargest(10, "rank")
        print(f"\n{focal} - model's top 10 partners (real recent co-play in brackets):")
        for r in g.itertuples():
            A_, B_ = deck_sets.get(r.a, set()), deck_sets.get(r.b, set()); n = len(A_ & B_)
            print(f"   {name[r.b]:40s} together in {n:3d} decks{'  <- real partner' if r.t else ''}")
