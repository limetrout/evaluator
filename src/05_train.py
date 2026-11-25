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
CUTOFF = "2024-01"
FORMATS = ["standard", "pioneer", "modern", "legacy", "pauper", "vintage"]

feat = pd.read_parquet(PROC / "features.parquet")
tg = pd.read_parquet(PROC / "targets.parquet")
df = tg.merge(feat, on="oracle_id", how="inner").reset_index(drop=True)
df["y"] = np.log1p(100 * df.target)
for f in FORMATS: df[f"fmt_{f}"] = (df.format == f).astype(int)
df["window_year"] = df.window_start.str[:4].astype(int) + df.window_start.str[5:7].astype(int) / 12
df["doc"] = df.type_line.str.lower() + " | " + df.clean_text
NONFEAT = {"oracle_id", "format", "target", "ever_played", "window_start", "n_months", "y", "clean_text", "type_line", "name", "first_release", "first_set", "doc"}
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

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.6,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
def fit_predict(train, test, rounds=1500):
    oof, te_text, tm = text_model(train, test)
    Xtr = train[FEATS].assign(text_score=oof); Xte = test[FEATS].assign(text_score=te_text)
    w = np.where(train.format == "standard", 2.0, 1.0)          # Standard is what we care most about
    gbm = lgb.train(PARAMS, lgb.Dataset(Xtr, train.y, weight=w), rounds)
    return gbm.predict(Xte), te_text, gbm, tm

def report(name, test, pred, k=(25, 50, 100)):
    out = {"model": name}
    for fmt in ["standard", "pioneer", "modern"]:
        m = (test.format == fmt).values
        if m.sum() < 30: continue
        t, p = test.target.values[m], pred[m]
        out[f"{fmt}_spearman"] = round(spearmanr(t, p).correlation, 3)
        order = np.argsort(-p)
        for kk in k: out[f"{fmt}_hit@{kk}"] = round(float((t[order[:kk]] >= 0.02).mean()), 2)   # share of top-k that hit >=2% of decks
        staples = t >= 0.10; out[f"{fmt}_staples"] = int(staples.sum())
        out[f"{fmt}_staples_in_top5pct"] = round(float(staples[order[: max(1, int(len(t) * .05))]].sum() / max(1, staples.sum())), 2)
    return out

if __name__ == "__main__":
    train = df[df.window_start < CUTOFF].reset_index(drop=True); test = df[df.window_start >= CUTOFF].reset_index(drop=True)
    print(f"train rows {len(train)}  test rows {len(test)}  (test = cards first eligible {CUTOFF} or later)")
    pred, text_pred, gbm, _ = fit_predict(train, test)
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
