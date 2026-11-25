"""SYNERGY MODEL, step 1: measure how much more often two cards appear together than chance predicts.

For each format (Standard, Pioneer, Modern) and each calendar quarter:
  * take winning MAIN decks only (sideboards are situational), non-land cards only
  * deck colours = colours of its non-land cards
  * for a pair (A, B) count n = decks containing both, and
    E = expected count if A and B were independent, computed ONLY among decks whose colours include
        both cards' colours  (so "two red cards in red decks" is not mistaken for synergy)
Saved per quarter, then aggregated by the model step over each pair's first year together.
  lift = (n + 1) / (E + 1)      > 1 means "they go together more than chance", < 1 "they avoid each other"
Output: data/processed/pair_quarters.parquet
"""
import pathlib, numpy as np, pandas as pd, scipy.sparse as sp
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC = ROOT / "data/processed"
cards = pd.read_parquet(PROC / "cards.parquet").set_index("oracle_id")
feat = pd.read_parquet(PROC / "features.parquet").set_index("oracle_id")
nonland = set(feat.index[(feat.type_land == 0)])
BIT = {"W": 1, "U": 2, "B": 4, "R": 8, "G": 16}
cmask = {o: sum(BIT[c] for c in (cs if cs is not None else []) if c in BIT) for o, cs in cards.colors.items()}

d = pd.read_parquet(PROC / "decks.parquet", columns=["deck", "format", "date", "oracle_id", "main"])
d = d[d.format.isin(["standard", "pioneer", "modern"]) & (d.main > 0) & d.oracle_id.isin(nonland)]
d["q"] = d.date.str[:4].astype(int) * 10 + (d.date.str[5:7].astype(int) - 1) // 3 + 1      # e.g. 20233 = 2023 Q3
out = []
for (fmt, q), g in d.groupby(["format", "q"]):
    decks = g.deck.unique()
    if len(decks) < 60: continue
    cnt = g.groupby("oracle_id").deck.nunique()
    cand = cnt.index[cnt >= 4].values                       # cards seen in >= 4 decks this quarter
    if len(cand) < 2: continue
    di = {x: i for i, x in enumerate(decks)}; ci = {x: i for i, x in enumerate(cand)}
    gg = g[g.oracle_id.isin(ci)]
    X = sp.csr_matrix((np.ones(len(gg)), (gg.deck.map(di).values, gg.oracle_id.map(ci).values)), shape=(len(decks), len(cand)))
    X.data[:] = 1
    dmask = g.assign(m=g.oracle_id.map(cmask)).groupby("deck").m.agg(lambda s: np.bitwise_or.reduce(s.values.astype(int)))
    dmask = dmask.reindex(decks).fillna(0).astype(int).values
    # for each colour combination U: number of decks containing U, and per-card counts within them
    NU = np.zeros(32); CU = np.zeros((32, len(cand)))
    for U in range(32):
        sub = (dmask & U) == U; NU[U] = sub.sum()
        if NU[U]: CU[U] = np.asarray(X[sub].sum(0)).ravel()
    C = (X.T @ X).toarray()
    cm = np.array([cmask.get(o, 0) for o in cand])
    ia, ib = np.triu_indices(len(cand), 1)
    U = cm[ia] | cm[ib]; nU = NU[U]
    with np.errstate(divide="ignore", invalid="ignore"):
        E = np.where(nU > 0, CU[U, ia] * CU[U, ib] / nU, 0)
    n = C[ia, ib]
    keep = ((n >= 2) | (E >= 1.0)) & (nU >= 20)
    out.append(pd.DataFrame({"format": fmt, "q": q, "a": cand[ia[keep]], "b": cand[ib[keep]], "n": n[keep].astype(np.int32),
                             "E": E[keep].astype(np.float32), "nU": nU[keep].astype(np.int32),
                             "nA": CU[U[keep], ia[keep]].astype(np.int32), "nB": CU[U[keep], ib[keep]].astype(np.int32)}))
pq = pd.concat(out, ignore_index=True)
pq.to_parquet(PROC / "pair_quarters.parquet")
print(len(pq), "pair-quarter rows;", pq.groupby("format").size().to_dict())
lift = (pq.n + 1) / (pq.E + 1)
print("share of rows with lift>=2:", round((lift >= 2).mean(), 3), "| lift<=0.5:", round((lift <= .5).mean(), 3))
