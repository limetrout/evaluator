"""SYNERGY MODEL v2: deck-based card map ("card2vec") + text-to-map model + per-card partner grading.

1. Card map: from winning MAIN decks seen before the cutoff (Standard, Pioneer, Modern, non-land cards), build
   card-card co-occurrence -> positive PMI -> SVD to 64 numbers per card. Cards that live in the same decks end
   up close together (captures whole archetypes / packages, not just pairs).
2. Text-to-map: ridge regression from a card's printed text (language-model embedding) + basic stats (+ Claude
   role tags if available) to its map position. Lets brand-new cards be placed on the map.
3. Pair score: cosine between the two cards' map positions (real position for cards with history before the
   cutoff, predicted position for new cards). Added as features to the v1 pair model.
4. Grading per new card: rank every card it could pair with in its format; recall of its real strong partners
   (lift >= 2 in >= 5 decks) in the top 10.
Strictly forward: everything used for year Y comes from data before year Y.
"""
import sys, pathlib, importlib.util, numpy as np, pandas as pd, lightgbm as lgb, scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"
spec = importlib.util.spec_from_file_location("pm", ROOT / "src/10_pair_model.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
DIM = 64

decks = pd.read_parquet(PROC / "decks.parquet", columns=["deck", "format", "date", "oracle_id", "main"])
decks = decks[decks.format.isin(["standard", "pioneer", "modern"]) & (decks.main > 0) & decks.oracle_id.isin(M.idx)]

# ---- optional Claude role tags (PROVIDES / REWARDS codes) -> per-card binary vectors
TAGS = None
tagfile = PROC / "card_tags.parquet"
if tagfile.exists() and "--no-tags" not in sys.argv:
    TAGS = pd.read_parquet(tagfile).set_index("oracle_id")

def card_map(cut_date):
    d = decks[decks.date < cut_date]
    cnt = d.groupby("oracle_id").deck.nunique(); keep = cnt.index[cnt >= 5]
    d = d[d.oracle_id.isin(keep)]
    di = {x: i for i, x in enumerate(d.deck.unique())}; ci = {x: i for i, x in enumerate(keep)}
    X = sp.csr_matrix((np.ones(len(d)), (d.deck.map(di).values, d.oracle_id.map(ci).values)), shape=(len(di), len(ci))); X.data[:] = 1
    C = (X.T @ X).tocoo()
    tot = C.sum(); row = np.asarray((X.T @ X).sum(1)).ravel()
    pc = row ** 0.75; pc = pc / pc.sum()
    pmi = np.log((C.data / tot) / ((row[C.row] / row.sum()) * pc[C.col]))
    m = (pmi > 0) & (C.row != C.col)
    P = sp.csr_matrix((pmi[m], (C.row[m], C.col[m])), shape=C.shape)
    U = TruncatedSVD(DIM, random_state=0).fit_transform(P)
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
    return dict(zip(keep, U))

def text_inputs(ids):
    e = M.E[[M.epos[o] for o in ids]]
    f = M.feat.loc[ids]
    extra = np.column_stack([f[["col_W", "col_U", "col_B", "col_R", "col_G", "mv", "type_creature", "type_instant", "type_sorcery",
                                "type_artifact", "type_enchantment", "type_planeswalker", "legendary"]].values.astype(float),
                             M.Smat[[M.idx[o] for o in ids]], M.Pmat[[M.idx[o] for o in ids]]])
    parts = [e, extra * 0.3]
    if TAGS is not None:
        t = TAGS.reindex(ids).fillna(0).values.astype(float); parts.append(t * 0.5)
    return np.hstack(parts)

def text_to_map(vmap, need_ids):
    """Fit text->map on cards with history; return predicted positions for need_ids, and cross-fitted predictions
    for the history cards themselves (so training rows see the same kind of noisy position a new card gets)."""
    hist = np.array(list(vmap)); Y = np.stack([vmap[o] for o in hist]); X = text_inputs(hist)
    cross = np.zeros_like(Y)
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):
        cross[b] = Ridge(alpha=10.0).fit(X[a], Y[a]).predict(X[b])
    r = Ridge(alpha=10.0).fit(X, Y)
    pred = r.predict(text_inputs(need_ids)) if len(need_ids) else np.zeros((0, DIM))
    nz = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    r2 = 1 - ((cross - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    return dict(zip(hist, nz(cross))), dict(zip(need_ids, nz(pred))), r2

def map_feats(a_ids, b_ids, real, cross, pred, train_mode, rel):
    """train_mode: the NEWER card of each pair uses its cross-fitted (text-predicted) position, the older its real one.
       test mode: cards with history use real positions, new cards use predicted positions."""
    out = np.zeros((len(a_ids), 3))
    for k, (a, b) in enumerate(zip(a_ids, b_ids)):
        def pos(o, newer):
            if train_mode and newer: return cross.get(o, pred.get(o))
            return real.get(o, pred.get(o))
        ra, rb = rel.get(a, 0), rel.get(b, 0)
        va, vb = pos(a, ra >= rb), pos(b, rb > ra)
        pa = cross.get(a, pred.get(a)); pb = cross.get(b, pred.get(b))
        out[k, 0] = va @ vb if va is not None and vb is not None else np.nan
        out[k, 1] = pa @ pb if pa is not None and pb is not None else np.nan
        out[k, 2] = (a in real) + (b in real)
    return pd.DataFrame(out, columns=["map_cos", "map_cos_textonly", "map_n_real"])

RELD = pd.to_datetime(M.cards.first_release).map(lambda t: t.toordinal() if t == t else 0).to_dict()

def candidates_for_year(Y, fmt):
    """New cards released in year Y in this format, and the cards they could have paired with."""
    q0 = Y * 10 + 1; q1 = q0 + 10
    pqf = M.pq[(M.pq.format == fmt) & (M.pq.q >= q0) & (M.pq.q < q1 + 3)]
    seen = set(pqf.a) | set(pqf.b)
    rel = M.cards.first_release.reindex(list(seen))
    new = [o for o in seen if isinstance(rel.get(o), str) and rel[o][:4] == str(Y)]
    return new, sorted(seen)

PARAMS = M.PARAMS
if __name__ == "__main__":
    results = []
    for Y in range(2019, 2025):
        cut = Y * 10 + 1; cut_date = f"{Y}-01-01"
        train = M.aggregate(M.pq[M.pq.q < cut]); train = train[train.start < cut]
        test = M.aggregate(M.pq[(M.pq.start >= cut) & (M.pq.start < cut + 10)])
        if len(test) < 200: continue
        real = card_map(cut_date)
        need = sorted((set(test.a) | set(test.b) | set(train.a) | set(train.b)) - set(real))
        # candidate sets for per-card grading
        cand = {f: candidates_for_year(Y, f) for f in ["standard", "pioneer", "modern"]}
        need = sorted(set(need) | {o for f in cand for o in cand[f][1] if o not in real})
        cross, pred, r2 = text_to_map(real, need)
        Xtr1, Xte1 = M.build(train), M.build(test)
        Ftr = map_feats(train.a.values, train.b.values, real, cross, pred, True, RELD)
        Fte = map_feats(test.a.values, test.b.values, real, cross, pred, False, RELD)
        w = np.sqrt(train.support.values)
        m1 = lgb.train(PARAMS, lgb.Dataset(Xtr1, train.y, weight=w), 600)
        Xtr2, Xte2 = pd.concat([Xtr1, Ftr], axis=1), pd.concat([Xte1, Fte], axis=1)
        m2 = lgb.train(PARAMS, lgb.Dataset(Xtr2, train.y, weight=w), 600)
        p1, p2, pm = m1.predict(Xte1), m2.predict(Xte2), Fte.map_cos.fillna(0).values
        strong = ((test.n >= 5) & (np.exp(test.y) >= 2)).values
        r = {"year": Y, "text_to_map_R2": round(r2, 3), "test_pairs": len(test)}
        for nm, v in [("v1", p1), ("map_only", pm), ("v2", p2)]:
            o = np.argsort(-v)
            r[f"{nm}_spearman"] = round(spearmanr(test.y, v).correlation, 3)
            r[f"{nm}_top2pct_strong"] = round(strong[o[: len(v) // 50]].mean(), 3)
        # ---- per-card grading: for each new card, rank all candidate partners
        lk = {(a, b): (n, y) for a, b, n, y in zip(test.a, test.b, test.n, test.y)}
        lk.update({(b, a): v for (a, b), v in list(lk.items())})
        rec = {"v1": [], "map_only": [], "v2": []}
        for fmt, (new, seen) in cand.items():
            for N in new:
                others = [o for o in seen if o != N]
                truth = np.array([(lk.get((N, o), (0, 0))[0] >= 5) and (np.exp(lk.get((N, o), (0, 0))[1]) >= 2) for o in others])
                if truth.sum() == 0: continue
                A = np.array([N] * len(others)); B = np.array(others)
                g = pd.DataFrame({"a": A, "b": B, "format": fmt, "start": cut})
                X1 = M.pair_features(A, B, np.full(len(A), fmt), np.full(len(A), Y)); F = map_feats(A, B, real, cross, pred, False, RELD)
                sc = {"v1": m1.predict(X1), "map_only": F.map_cos.fillna(0).values, "v2": m2.predict(pd.concat([X1, F], axis=1))}
                for k, v in sc.items():
                    top = np.argsort(-v)[:10]
                    rec[k].append(truth[top].sum() / min(10, truth.sum()))
        for k, v in rec.items(): r[f"{k}_partner_recall@10"] = round(float(np.mean(v)), 3)
        r["graded_new_cards"] = len(rec["v1"])
        results.append(r); print(r, flush=True)
    R = pd.DataFrame(results).set_index("year"); tag = "with_tags" if TAGS is not None else "no_tags"
    R.to_csv(OUT / f"synergy_v2_backtest_{tag}.csv"); print(R.T.to_string())
