"""SYNERGY MODEL v3: a ranker trained on the real question - "given card X, which cards go with it?"

Training examples: for every card released in a past year Y', in each format, EVERY card it could have been paired
with that year (not only pairs that already co-occurred). Label: 2 = strong partner (lift >= 2 in >= 5 decks),
1 = mild partner (lift >= 1.5 in >= 3 decks), 0 = otherwise. LightGBM learning-to-rank, grouped by focal card.
Features: v1 pair features (+ Claude role tags), card-map closeness (map from data before Y'), and each candidate's
popularity in that format the year BEFORE Y' (known at the time).
Forward test: for year Y the model only uses focal years < Y with labels truncated at the start of Y.
Baselines on the same per-card test: popularity only, card map only, v1 pair model.
"""
import os, sys, pathlib, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"
spec = importlib.util.spec_from_file_location("s12", ROOT / "src/12_synergy_v2.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
M = S.M
FMTS = ["standard", "pioneer", "modern"]
YEARS = range(2017, 2025)

deck_year = S.decks.assign(y=S.decks.date.str[:4].astype(int))
ndecks = deck_year.groupby(["format", "y"]).deck.nunique()
pop = (deck_year.groupby(["format", "y", "oracle_id"]).deck.nunique() / ndecks).rename("pop")

def build_year(Yp):
    """All (focal new card, candidate) rows for focal year Yp, with features (cached)."""
    cache = PROC / f"ranker_rows_{Yp}_{'tags' if M.USE_TAGS else 'notags'}.parquet"
    if cache.exists(): return pd.read_parquet(cache)
    real = S.card_map(f"{Yp}-01-01")
    parts = []; need = set()
    cands = {f: S.candidates_for_year(Yp, f) for f in FMTS}
    for f in FMTS: need |= set(cands[f][1])
    cross, pred, _ = S.text_to_map(real, sorted(need - set(real)))
    for f in FMTS:
        new, seen = cands[f]
        if not new: continue
        A = np.repeat(new, len(seen)); B = np.tile(seen, len(new)); keep = A != B; A, B = A[keep], B[keep]
        X = M.pair_features(A, B, np.full(len(A), f), np.full(len(A), Yp))
        F = S.map_feats(A, B, real, cross, pred, False, S.RELD)
        X = pd.concat([X, F], axis=1)
        X["cand_pop_prev"] = pd.Series(B).map(lambda o: pop.get((f, Yp - 1, o), 0.0)).values
        X["focal"] = A; X["cand"] = B; X["format"] = f; X["fy"] = Yp
        parts.append(X)
    df = pd.concat(parts, ignore_index=True); df.to_parquet(cache); return df

def labels(df, cut_q=None):
    rows = M.pq if cut_q is None else M.pq[M.pq.q < cut_q]
    agg = M.aggregate(rows[rows.start // 10 == df.fy.iloc[0]]) if len(df) else None
    lk = {}
    for a, b, n, y, f in zip(agg.a, agg.b, agg.n, agg.y, agg.format):
        lab = 2 if (n >= 5 and y >= np.log(2)) else (1 if (n >= 3 and y >= np.log(1.5)) else 0)
        lk[(f, a, b)] = lab; lk[(f, b, a)] = lab
    return np.array([lk.get(k, 0) for k in zip(df.format, df.focal, df.cand)])

def recall_at10(df, score, lab):
    out = []
    for (f, a), g in df.assign(s=score, l=lab).groupby(["format", "focal"]):
        t = (g.l == 2).sum()
        if t == 0: continue
        top = g.nlargest(10, "s"); out.append((top.l == 2).sum() / min(10, t))
    return float(np.mean(out)), len(out)

META = ["focal", "cand", "format", "fy"]
RANK_PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10], learning_rate=0.05, num_leaves=63,
                   min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=20,
                   verbose=-1, num_threads=4)
if __name__ == "__main__":
    data = {Yp: build_year(Yp) for Yp in YEARS}
    print("rows per year:", {k: len(v) for k, v in data.items()}, flush=True)
    res = []
    rng = np.random.default_rng(0)
    for Y in range(2019, 2025):
        cut = Y * 10 + 1
        tr = []
        for Yp in YEARS:
            if Yp >= Y: continue
            d = data[Yp]; l = labels(d, cut)
            keep = (l > 0) | (rng.random(len(d)) < 0.3)          # keep all partners, 30% of non-partners
            tr.append(d[keep].assign(lab=l[keep]))
        tr = pd.concat(tr, ignore_index=True).sort_values(["fy", "format", "focal"])
        te = data[Y]; lte = labels(te)
        feats = [c for c in te.columns if c not in META]
        grp = tr.groupby(["fy", "format", "focal"], sort=False).size().values
        rk = lgb.train(RANK_PARAMS, lgb.Dataset(tr[feats], tr.lab, group=grp), 400)
        # v1-style regression on co-occurring pairs (as before), for comparison
        trainp = M.aggregate(M.pq[M.pq.q < cut]); trainp = trainp[trainp.start < cut]
        Xv1 = M.build(trainp); v1 = lgb.train(M.PARAMS, lgb.Dataset(Xv1, trainp.y, weight=np.sqrt(trainp.support.values)), 600)
        v1cols = list(Xv1.columns)
        sc = {"v3_ranker": rk.predict(te[feats]), "v1_pair_model": v1.predict(te[v1cols]),
              "card_map_only": te.map_cos.fillna(-1).values, "popularity_only": te.cand_pop_prev.values}
        r = {"year": Y, "train_rows": len(tr)}
        for k, v in sc.items():
            rec, n = recall_at10(te, v, lte); r[f"{k}_recall@10"] = round(rec, 3)
        r["graded_cards"] = n
        res.append(r); print(r, flush=True)
    R = pd.DataFrame(res).set_index("year"); R.to_csv(OUT / f"synergy_ranker_backtest_{'tags' if M.USE_TAGS else 'notags'}.csv")
    print(R.T.to_string()); print("mean:", R.mean().round(3).to_dict())
    imp = pd.Series(rk.feature_importance("gain"), index=feats).sort_values(ascending=False); print(imp.head(20))
