"""DECK BUILDER + honest backtest.

build_deck(seeds, candidates, quality, synergy): greedy. Start from the seed cards; repeatedly add the non-land card
with the best marginal value = quality + LAMBDA * average synergy with the cards already in the deck, restricted to
the seeds' colours, with a soft mana-curve penalty; until ~23 non-land slots. Copies: 4 for the strongest picks,
fewer for the rest. Then a mana base (best dual/utility lands in the colours + basics by colour weight) and a
15-card sideboard of interaction.

Backtest (python src/21_deck_builder.py backtest): for past Standard quarters, cluster that quarter's decks into
archetypes; for each archetype give the builder only its 3 signature cards + information available BEFORE the quarter
(prior-quarter play rates, blind single-card predictions for new cards, a synergy model trained on earlier years).
Score = how many of the archetype's real core cards (in >= 50% of its decks) the built list contains.
Baselines: same seeds + most-played cards in those colours last quarter; same seeds + highest quality, no synergy.
"""
import os, sys, pathlib, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"; INS = OUT / "insights"
spec = importlib.util.spec_from_file_location("r14", ROOT / "src/14_synergy_ranker.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
S, M = R.S, R.M
feat, name = M.feat, M.cards.name
NONLAND = set(feat.index[feat.type_land == 0])
BIT = "WUBRG"
def colset(o):
    c = M.cards.colors.get(o); return set(c) if c is not None else set()

# ------------------------------------------------------------------------------------------ the builder
def build_deck(seeds, cands, quality, syn, lam=1.0, slots=23):
    """seeds: list of oracle ids; cands: candidate ids; quality: dict id->[0,1]; syn: callable (list a, list b)->matrix."""
    cols = set().union(*[colset(o) for o in seeds]) or set(BIT)
    cands = [o for o in cands if o in NONLAND and colset(o) <= cols and o not in seeds]
    cands = sorted(cands, key=lambda o: -quality.get(o, 0))[:220]          # keep it tractable
    if not cands: return [], cols
    allc = list(seeds) + cands
    SY = syn(allc, allc); np.fill_diagonal(SY, 0)
    pos = {o: i for i, o in enumerate(allc)}
    deck = list(seeds); q = np.array([quality.get(o, 0) for o in allc])
    qz = (q - q.mean()) / (q.std() + 1e-9); sz_mu, sz_sd = SY.mean(), SY.std() + 1e-9
    mv = np.array([feat.mv.get(o, 3) for o in allc])
    while len(deck) < 11:                                            # ~11 distinct cards for ~23 slots
        best, bv = None, -1e9
        di = [pos[o] for o in deck]
        for o in cands:
            if o in deck: continue
            j = pos[o]; s = (SY[j, di].mean() - sz_mu) / sz_sd
            curve = -0.35 * max(0, sum(mv[i] >= 5 for i in di) + (mv[j] >= 5) - 3)
            v = qz[j] + lam * s + curve
            if v > bv: best, bv = o, v
        if best is None: break
        deck.append(best)
    # copies: rank by value inside the final deck
    di = [pos[o] for o in deck]
    val = {o: qz[pos[o]] + lam * (SY[pos[o], di].mean() - sz_mu) / sz_sd for o in deck}
    order = sorted(deck, key=lambda o: -val[o]); copies = {}
    for k, o in enumerate(order): copies[o] = 4 if (o in seeds or k < 6) else (3 if k < 8 else 2)
    tot = sum(copies.values())
    for o in reversed(order):                                        # trim / pad to the slot count
        while tot > slots and copies[o] > 1 and o not in seeds: copies[o] -= 1; tot -= 1
    for o in order:
        while tot < slots and copies[o] < 4: copies[o] += 1; tot += 1
    return [(o, copies[o]) for o in order], cols

def mana_base(deck, cols, land_pool, land_quality, n_lands=None):
    avgmv = np.average([feat.mv[o] for o, _ in deck], weights=[c for _, c in deck]) if deck else 3
    n = n_lands or (22 if avgmv < 2.3 else 23 if avgmv < 2.9 else 24 if avgmv < 3.6 else 25)
    pips = {c: sum(feat[f"pip_{c}"][o] * k for o, k in deck) for c in cols}
    lands = []
    if len(cols) >= 2:
        ok = [o for o in land_pool if (set(M.cards.produced_mana.get(o) if M.cards.produced_mana.get(o) is not None else []) - {"C"}) and
              (set(M.cards.produced_mana.get(o)) - {"C"}) <= cols and len(set(M.cards.produced_mana.get(o)) - {"C"}) >= 2]
        for o in sorted(ok, key=lambda o: -land_quality.get(o, 0))[: max(2, len(cols) * 2)]:
            lands.append((o, 4 if len(cols) == 2 else 3))
    nb = sum(k for _, k in lands)
    while nb > n - 6 and lands: o, k = lands.pop(); nb -= k
    left = n - nb; totp = sum(pips.values()) or 1
    BASIC = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
    basics = {BASIC[c]: int(round(left * pips[c] / totp)) for c in cols} if cols else {"Wastes": left}
    diff = left - sum(basics.values())
    if basics: basics[max(basics, key=basics.get)] += diff
    return lands, basics, n

def sideboard(deck, cols, cands, quality, k=15):
    main = {o for o, _ in deck}
    sb = [o for o in cands if o in NONLAND and o not in main and colset(o) <= cols and feat.interaction.get(o, 0) + feat.t_exile_gy.get(o, 0) + feat.t_discard_opp.get(o, 0) + feat.t_counter.get(o, 0) > 0]
    sb = sorted(sb, key=lambda o: -quality.get(o, 0))[:6]
    out, tot = [], 0
    for i, o in enumerate(sb):
        c = 3 if i < 3 else 2
        c = min(c, k - tot)
        if c <= 0: break
        out.append((o, c)); tot += c
    return out

# ------------------------------------------------------------------------------------------ archetypes of a period
def archetypes_of(d, min_size=8):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import HDBSCAN
    m = d[(d.main > 0) & d.oracle_id.isin(NONLAND)]
    docs = m.groupby("deck").apply(lambda g: " ".join(f"{o} " * min(int(c), 4) for o, c in zip(g.oracle_id, g.main)), include_groups=False)
    if len(docs) < 40: return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(token_pattern=r"[^ ]+", sublinear_tf=True).fit_transform(docs.values)
    Z = TruncatedSVD(min(40, X.shape[1] - 1), random_state=0).fit_transform(X); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    lab = HDBSCAN(min_cluster_size=max(min_size, len(docs) // 60), min_samples=3).fit_predict(Z)
    allfreq = m.groupby("oracle_id").deck.nunique() / len(docs)
    out = []
    for a in sorted(set(lab) - {-1}):
        dk = docs.index[lab == a]; mm = m[m.deck.isin(dk)]
        freq = mm.groupby("oracle_id").deck.nunique() / len(dk)
        sig = (freq * np.log(freq / allfreq.reindex(freq.index))).where(freq >= 0.7).dropna().sort_values(ascending=False)
        out.append(dict(n=len(dk), freq=freq, seeds=list(sig.index[:3]), core=set(freq.index[freq >= 0.5])))
    return out

# ------------------------------------------------------------------------------------------ backtest
if __name__ == "__main__" and sys.argv[1:2] == ["backtest"]:
    dk = pd.read_parquet(PROC / "decks.parquet"); dk = dk[dk.format == "standard"]
    dk["q"] = dk.date.str[:4].astype(int) * 10 + (dk.date.str[5:7].astype(int) - 1) // 3 + 1
    bt = pd.read_csv(OUT / "backtest_predictions.csv"); bt = bt[bt.format == "standard"]
    nm2id = feat.reset_index().drop_duplicates("name").set_index("name").oracle_id
    bt["oid"] = bt.name.map(nm2id)
    rel = M.cards.first_release; st = M.cards.first_set_type
    results = []
    probs = {}
    QUARTERS = [int(x) for x in os.environ.get("QUARTERS", "20212,20214,20222,20224,20232,20234,20241").split(",")]
    for Q in QUARTERS:
        Y = Q // 10; cut = Y * 10 + 1
        if Y not in probs:   # synergy model trained only on focal years before Y, labels truncated at start of Y
            tr = []; rng = np.random.default_rng(0)
            for Yp in R.YEARS:
                if Yp >= Y: continue
                d_ = R.build_year(Yp); l = R.labels(d_, cut); keep = (l > 0) | (rng.random(len(d_)) < 0.3); tr.append(d_[keep].assign(lab=l[keep]))
            tr = pd.concat(tr, ignore_index=True); FE = [c for c in tr.columns if c not in R.META + ["lab"]]
            pm = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8,
                                bagging_freq=1, verbose=-1, num_threads=4), lgb.Dataset(tr[FE], (tr.lab == 2).astype(int)), 300)
            real = S.card_map(f"{Y}-01-01"); probs[Y] = (pm, FE, real)
        pm, FE, real = probs[Y]
        prevQ = Q - 1 if Q % 10 > 1 else (Y - 1) * 10 + 4
        prev = dk[dk.q == prevQ]; cur_ = dk[dk.q == Q]
        if prev.deck.nunique() < 60 or cur_.deck.nunique() < 60: continue
        prev_share = prev[prev.main > 0].groupby("oracle_id").deck.nunique() / prev.deck.nunique()
        qstart = f"{Y}-{(Q % 10 - 1) * 3 + 1:02d}-01"; qend = f"{Y}-{(Q % 10) * 3:02d}-28"
        new = [o for o in NONLAND if isinstance(rel.get(o), str) and st.get(o) in ("expansion", "core") and f"{Y - 2}-{(Q % 10 - 1) * 3 + 1:02d}-01" <= rel[o] <= qend]
        cand_pool = sorted(set(prev_share.index) & NONLAND | set(new))
        # quality: last quarter's play share for cards that existed; blind model prediction (calibrated on earlier years) for the rest
        hist = bt[bt.test_year < Y]
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(hist.pred, hist.target) if len(hist) > 100 else None
        btY = bt[bt.test_year <= Y].drop_duplicates("oid", keep="last").set_index("oid").pred
        quality = {}
        for o in cand_pool:
            if o in prev_share.index: quality[o] = float(prev_share[o])
            elif o in btY.index and iso is not None: quality[o] = float(iso.predict([btY[o]])[0])
            else: quality[o] = 0.0
        need = sorted(set(cand_pool) - set(real)); cross, pred, _ = S.text_to_map(real, need)
        popY = prev_share
        def syn(a_list, b_list):
            A_ = np.repeat(a_list, len(b_list)); B_ = np.tile(b_list, len(a_list))
            X = M.pair_features(A_, B_, np.full(len(A_), "standard"), np.full(len(A_), Y))
            X = pd.concat([X, S.map_feats(A_, B_, real, cross, pred, False, S.RELD)], axis=1)
            X["cand_pop_prev"] = pd.Series(B_).map(popY).fillna(0).values
            P_ = pm.predict(X[FE]).reshape(len(a_list), len(b_list)); return (P_ + P_.T) / 2 if P_.shape[0] == P_.shape[1] else P_
        archs = archetypes_of(cur_)
        for a in archs:
            seeds = [o for o in a["seeds"] if o in quality or o in cand_pool]
            if len(seeds) < 2: continue
            truth = a["core"] - set(seeds)
            if len(truth) < 3: continue
            deck, cols = build_deck(seeds, cand_pool, quality, syn)
            cols_ok = [o for o in cand_pool if o in NONLAND and colset(o) <= cols and o not in seeds]
            popb = seeds + sorted(cols_ok, key=lambda o: -prev_share.get(o, 0))[: len(deck) - len(seeds)]
            qb = seeds + sorted(cols_ok, key=lambda o: -quality.get(o, 0))[: len(deck) - len(seeds)]
            def score(lst):
                s_ = set(lst) - set(seeds); hit = len(s_ & truth); return hit / len(truth), hit / max(1, len(s_))
            r = {"quarter": Q, "archetype_decks": a["n"], "seeds": "; ".join(name[o] for o in seeds), "core_size": len(truth),
                 "builder_recall": score([o for o, _ in deck])[0], "builder_precision": score([o for o, _ in deck])[1],
                 "popularity_recall": score(popb)[0], "quality_only_recall": score(qb)[0]}
            # how many truly NEW cards (not played last quarter) in the core did each method find?
            newcore = {o for o in truth if o not in prev_share.index}
            if newcore:
                r["new_core_cards"] = len(newcore)
                r["builder_found_new"] = len(newcore & {o for o, _ in deck}); r["popularity_found_new"] = len(newcore & set(popb))
            # NEW-CARD DISCOVERY: among cards NOT played last quarter, in the shell's colours, rank candidates three ways
            if newcore:
                newc = [o for o in cand_pool if o not in prev_share.index and o in NONLAND and colset(o) <= cols and o not in seeds]
                if len(newc) >= 10:
                    SY = syn(newc, seeds).mean(1); qq = np.array([quality.get(o, 0) for o in newc])
                    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
                    for nm_, sc in [("syn", SY), ("rating", qq), ("combo", z(SY) + z(qq))]:
                        top = {newc[k] for k in np.argsort(-sc)[:10]}
                        r[f"new_hit10_{nm_}"] = len(top & newcore) / min(10, len(newcore))
                    r["new_candidates"] = len(newc)
                    rnd = len(newcore) / len(newc) * 10 / min(10, len(newcore)); r["new_hit10_random"] = min(1.0, rnd)
            results.append(r); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    Rr = pd.DataFrame(results); Rr.to_csv(OUT / "deck_builder_backtest.csv", index=False)
    print("\nMEAN over", len(Rr), "archetype-quarters:")
    print(Rr[["builder_recall", "builder_precision", "popularity_recall", "quality_only_recall"]].mean().round(3).to_dict())
    nh = [c for c in Rr.columns if c.startswith("new_hit10_")]
    if nh: print("NEW-CARD DISCOVERY (share of real new core cards in top 10):", Rr[nh].mean().round(3).to_dict(), "| archetypes graded:", int(Rr[nh[0]].notna().sum()))
    if "new_core_cards" in Rr:
        n = Rr.new_core_cards.sum(); print(f"new core cards (unplayed the quarter before): {int(n)} | builder found {int(Rr.builder_found_new.sum())} | popularity found {int(Rr.popularity_found_new.sum())}")
