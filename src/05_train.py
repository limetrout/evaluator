"""Train + honestly evaluate the card-scoring model.

Rows   : (card, format) pairs from targets.parquet  (Standard, Pioneer, Modern, Legacy, Pauper, Vintage)
y      : log1p(100 * peak early play rate)   i.e. ~ "log of % of top decks that ran it"
Model  : (1) TF-IDF n-grams of rules text -> ridge regression  => "text_score" (stacked, out-of-fold)
         (2) LightGBM gradient-boosted trees on ~330 card features + text_score + format
Eval   : TEMPORAL holdout. Train only on cards whose window started before CUTOFF, predict the cards
         released after it, compare with what actually happened. Baselines for comparison.
"""
import json, pathlib, warnings, numpy as np, pandas as pd, lightgbm as lgb, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"
import sys
CUTOFF = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1][:2] == "20" else "2024-01"
FORMATS = ["standard", "pioneer", "modern", "legacy", "pauper", "vintage"]

feat = pd.read_parquet(PROC / "features.parquet")
tg = pd.read_parquet(PROC / "targets.parquet")
df = tg.merge(feat, on="oracle_id", how="inner").reset_index(drop=True)
# ---- semantic embeddings: direct (PCA) features + analog lookups -----------------------------------------
EMB_OK = (PROC / "embeddings.npy").exists()
if EMB_OK:
    from sklearn.decomposition import PCA
    E = np.load(PROC / "embeddings.npy"); eid = pd.read_csv(PROC / "embeddings_ids.csv").oracle_id
    pos = {o: i for i, o in enumerate(eid)}
    PCS = PCA(24, random_state=0).fit_transform(E)

def prepare(d):
    """Derived columns shared by training (05) and scoring (06)."""
    d = d.copy()
    d["y"] = np.log1p(100 * d.target)
    for f in FORMATS: d[f"fmt_{f}"] = (d.format == f).astype(int)
    d["window_year"] = d.window_start.str[:4].astype(int) + d.window_start.str[5:7].astype(int) / 12
    d["doc"] = d.type_line.str.lower() + " | " + d.clean_text
    if EMB_OK:
        d["emb_idx"] = d.oracle_id.map(pos)
        for j in range(24): d[f"emb_pc{j}"] = PCS[d.emb_idx.values, j]
    d["rel_day"] = pd.to_datetime(d.first_release).map(pd.Timestamp.toordinal)
    return d
df = prepare(df)
ANALOG = ["an_top1_y", "an_top1_sim", "an_top1_mvdiff", "an_w10_y", "an_max10_y", "an_mean20_y", "an_n_strong20", "an_top3_mean_y", "an_best_cheaper_y"]

def analog_feats(query, pool):
    """For every query row: look at the 20 most similar cards (same format) released STRICTLY EARLIER that are in
    `pool` (training rows only), and summarise how much those historical analogs were played."""
    out = pd.DataFrame(np.nan, index=query.index, columns=ANALOG)
    if not EMB_OK: return out
    strong = np.log1p(5.0)
    for fmt in query.format.unique():
        q = query[query.format == fmt]; p = pool[pool.format == fmt]
        if len(p) == 0: continue
        Pe = E[p.emb_idx.values]; pday = p.rel_day.values; py = p.y.values; pmv = p.mv.values; poid = p.emb_idx.values.astype(int)
        for a in range(0, len(q), 2000):
            qq = q.iloc[a:a + 2000]
            S = E[qq.emb_idx.values] @ Pe.T
            S[(pday[None, :] >= qq.rel_day.values[:, None]) | (poid[None, :] == qq.emb_idx.values.astype(int)[:, None])] = -2
            idx = np.argpartition(-S, 20, axis=1)[:, :20]
            ss = np.take_along_axis(S, idx, 1); o = np.argsort(-ss, 1); idx = np.take_along_axis(idx, o, 1); ss = np.take_along_axis(ss, o, 1)
            valid = ss > -1; ys = np.where(valid, py[idx], np.nan)
            w = np.where(valid, np.exp(10 * ss), 0)[:, :10]
            cheaper = np.where(valid & (pmv[idx] >= qq.mv.values[:, None]), ys, np.nan)   # analogs costing the same or MORE
            out.loc[qq.index, "an_top1_y"] = ys[:, 0]; out.loc[qq.index, "an_top1_sim"] = np.where(valid[:, 0], ss[:, 0], np.nan)
            out.loc[qq.index, "an_top1_mvdiff"] = qq.mv.values - np.where(valid[:, 0], pmv[idx[:, 0]], np.nan)
            out.loc[qq.index, "an_w10_y"] = np.nansum(w * np.nan_to_num(ys[:, :10]), 1) / np.maximum(w.sum(1), 1e-9)
            with np.errstate(all="ignore"):
                out.loc[qq.index, "an_max10_y"] = np.nanmax(ys[:, :10], 1); out.loc[qq.index, "an_mean20_y"] = np.nanmean(ys, 1)
                out.loc[qq.index, "an_top3_mean_y"] = np.nanmean(ys[:, :3], 1); out.loc[qq.index, "an_best_cheaper_y"] = np.nanmax(cheaper[:, :10], 1)
            out.loc[qq.index, "an_n_strong20"] = (np.nan_to_num(ys, nan=0) > strong).sum(1)
    return out

NONFEAT = {"oracle_id", "format", "target", "ever_played", "window_start", "n_months", "y", "clean_text", "type_line", "name", "first_release", "first_set", "doc", "emb_idx", "rel_day"}
FEATS = [c for c in df.columns if c not in NONFEAT]

def text_model(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_features=60000, sublinear_tf=True)
    Xtr = vec.fit_transform(train.doc); Xte = vec.transform(test.doc)
    F = lambda d: sp.csr_matrix(d[[f"fmt_{f}" for f in FORMATS]].values * 3.0)
    oof = np.zeros(len(train))
    for a, b in GroupKFold(5).split(Xtr, groups=train.oracle_id):
        m = Ridge(alpha=3.0).fit(sp.hstack([Xtr[a], F(train.iloc[a])]), train.y.iloc[a]); oof[b] = m.predict(sp.hstack([Xtr[b], F(train.iloc[b])]))
    m = Ridge(alpha=3.0).fit(sp.hstack([Xtr, F(train)]), train.y)
    return oof, m.predict(sp.hstack([Xte, F(test)])), (vec, m)

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.6,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
def fit_predict(train, test, rounds=1500):
    oof, te_text, tm = text_model(train, test)
    Xtr = pd.concat([train[FEATS].assign(text_score=oof), analog_feats(train, train)], axis=1)
    Xte = pd.concat([test[FEATS].assign(text_score=te_text), analog_feats(test, train)], axis=1)
    w = np.where(train.format == "standard", 2.0, 1.0)          # Standard is what we care most about
    gbm = lgb.train(PARAMS, lgb.Dataset(Xtr, train.y, weight=w), rounds)
    return gbm.predict(Xte), te_text, gbm, tm, Xte

def report(name, test, pred, k=(25, 50, 100)):
    out = {"model": name}
    for fmt in ["standard", "pioneer", "modern"]:
        m = (test.format == fmt).values
        if m.sum() < 30: continue
        t, p = test.target.values[m], pred[m]
        out[f"{fmt}_spearman"] = round(spearmanr(t, p).correlation, 3)
        order = np.argsort(-p)
        for kk in k: out[f"{fmt}_hit@{kk}"] = round(float((t[order[:kk]] >= 0.02).mean()), 2)   # share of top-k that hit >=2% of decks
        r = test.rarity.values[m] >= 2
        out[f"{fmt}_spearman_rares"] = round(spearmanr(t[r], p[r]).correlation, 3)
        if fmt == "standard":   # per set: of the model's 10 favourite cards in each set, share that hit >=2% of decks
            sets = test.first_set.values[m]; hits = []
            for st in np.unique(sets):
                mm = sets == st
                if mm.sum() >= 40: hits.append((t[mm][np.argsort(-p[mm])[:10]] >= 0.02).mean())
            out["standard_per_set_top10_hit"] = round(float(np.mean(hits)), 2) if hits else np.nan
        staples = t >= 0.10; out[f"{fmt}_staples"] = int(staples.sum())
        out[f"{fmt}_staples_in_top5pct"] = round(float(staples[order[: max(1, int(len(t) * .05))]].sum() / max(1, staples.sum())), 2)
    return out

if __name__ == "__main__":
    train = df[df.window_start < CUTOFF].reset_index(drop=True); test = df[df.window_start >= CUTOFF].reset_index(drop=True)
    print(f"train rows {len(train)}  test rows {len(test)}  (test = cards first eligible {CUTOFF} or later)")
    pred, text_pred, gbm, *_ = fit_predict(train, test)
    rows = [report("FULL MODEL (features + text)", test, pred), report("text-only ridge", test, text_pred)]
    # baselines
    b1 = lgb.train(PARAMS, lgb.Dataset(train[["rarity", "mv"] + [f"fmt_{f}" for f in FORMATS]], train.y), 300)
    rows.append(report("baseline: rarity + mana value", test, b1.predict(test[["rarity", "mv"] + [f"fmt_{f}" for f in FORMATS]])))
    rows.append(report("baseline: rarity only", test, test.rarity.values + np.random.rand(len(test)) * 1e-3))
    ev = pd.DataFrame(rows).set_index("model"); print(ev.T.to_string()); ev.to_csv(OUT / "evaluation_temporal.csv")
    # show what the held-out predictions looked like for Standard
    t = test[test.format == "standard"].assign(pred=pred[test.format == "standard"])
    t["pred_pct"] = np.expm1(t.pred).round(1); t["actual_pct"] = (100 * t.target).round(1)
    t.sort_values("pred", ascending=False)[["name", "first_set", "pred_pct", "actual_pct"]].to_csv(OUT / "holdout_standard_predictions.csv", index=False)
    print(t.sort_values("pred", ascending=False)[["name", "first_set", "pred_pct", "actual_pct"]].head(40).to_string(index=False))
    imp = pd.Series(gbm.feature_importance("gain"), index=gbm.feature_name()).sort_values(ascending=False)
    imp.to_csv(OUT / "feature_importance.csv"); print(imp.head(30))
