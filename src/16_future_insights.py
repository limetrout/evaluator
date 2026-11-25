"""Apply both models to the post-rotation Standard pool (Scryfall f:future) and produce insights.

Outputs (output/insights/):
  card_partners.csv       each card's top 12 predicted partners (v3 ranker), with current co-play
  top_pairs.csv           strongest predicted pairs in the pool (probability model), proven / hidden / untested
  packages.csv            synergy packages (graph communities of strong pairs) = candidate archetypes
  package_members.csv     members of each package
  rotation.csv            currently played Standard cards: rotating vs surviving
  single_card_*.csv       best cards by colour, sleepers, synergy-dependent cards, new spoilers
"""
import pathlib, importlib.util, numpy as np, pandas as pd, lightgbm as lgb, networkx as nx
from networkx.algorithms.community import louvain_communities
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"; INS = OUT / "insights"
spec = importlib.util.spec_from_file_location("s12", ROOT / "src/12_synergy_v2.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
M = S.M
ranker = lgb.Booster(model_file=str(OUT / "models/synergy_ranker.txt")); prob = lgb.Booster(model_file=str(OUT / "models/synergy_prob.txt"))
FEATS = pd.read_csv(OUT / "models/synergy_features.csv").iloc[:, 0].tolist()
name = M.cards.name

# ---------------------------------------------------------------- pool + single-card scores
single = pd.read_csv(OUT / "post_rotation_scores.csv")
nm2id = M.feat.reset_index().drop_duplicates("name").set_index("name").oracle_id
single["oid"] = single.name.map(nm2id)
pool = set(pd.read_csv(PROC / "future_pool.csv").oracle_id)
single = single[single.oid.isin(pool)].drop_duplicates("oid").set_index("oid")
nonland = [o for o in single.index if M.feat.type_land.get(o, 1) == 0 and o in M.idx]
NEWSETS = {"NAU", "MDD"}
cand = [o for o in nonland if single.percentile[o] >= 30 or single.pool_set[o] in NEWSETS or single.current_std_play_pct[o] > 0]
print("pool:", len(single), "| non-land:", len(nonland), "| scored for synergy:", len(cand), flush=True)

# ---------------------------------------------------------------- recent Standard decks (2025+) for "what is played now"
dk = S.decks[(S.decks.format == "standard") & (S.decks.date >= "2025-01-01")]
NREC = dk.deck.nunique(); DS = dk.groupby("oracle_id").deck.agg(set)
def together(a, b):
    A, B = DS.get(a, set()), DS.get(b, set()); n = len(A & B); E = len(A) * len(B) / max(NREC, 1)
    return n, ((n + 1) / (E + 1) if A and B else np.nan)
last12 = S.decks[(S.decks.format == "standard") & (S.decks.date >= (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime("%Y-%m-%d"))]
POP = last12.groupby("oracle_id").deck.nunique() / max(1, last12.deck.nunique())

# ---------------------------------------------------------------- score all pairs (chunks)
REAL = S.card_map("2099-01-01")
cross, pred, _ = S.text_to_map(REAL, sorted(set(cand) - set(REAL)))
ids = np.array(cand); cm = np.array([M.CM[M.idx[o]] for o in ids])
ia, ib = np.triu_indices(len(ids), 1); ok = M.POP[cm[ia] | cm[ib]] <= 3; ia, ib = ia[ok], ib[ok]
print("pairs:", len(ia), flush=True)
res = []
for s0 in range(0, len(ia), 300_000):
    A, B = ids[ia[s0:s0 + 300_000]], ids[ib[s0:s0 + 300_000]]
    X = M.pair_features(A, B, np.full(len(A), "standard"), np.full(len(A), 2026))
    X = pd.concat([X, S.map_feats(A, B, REAL, cross, pred, False, S.RELD)], axis=1)
    X["cand_pop_prev"] = pd.Series(B).map(POP).fillna(0).values; Xab = X[FEATS]
    pab, rab = prob.predict(Xab), ranker.predict(Xab)
    X["cand_pop_prev"] = pd.Series(A).map(POP).fillna(0).values; Xba = X[FEATS]
    pba, rba = prob.predict(Xba), ranker.predict(Xba)
    tagcols = [c for c in X.columns if c.startswith("tag_") and c not in ("tag_total", "tag_n_tagged", "tag_both_reward", "tag_both_provide")]
    res.append(pd.DataFrame({"a": A, "b": B, "prob": (pab + pba) / 2, "rank_ab": rab, "rank_ba": rba, "map_cos": X.map_cos.values,
                             "tag_total": X.tag_total.values, "tribe": X.tag_tribe.values,
                             "tag_codes": [",".join(c[4:] for c in tagcols if r[c] > 0) for r in X[tagcols].to_dict("records")] if len(X) < 0 else ""}))
    print(f"  scored {min(s0 + 300_000, len(ia))}/{len(ia)}", flush=True)
P = pd.concat(res, ignore_index=True)
P[["a", "b", "prob", "rank_ab", "rank_ba", "map_cos"]].to_parquet(PROC / "future_pairs.parquet")

# explanations only for pairs we will show (cheap)
TAGC = [c[2:] for c in pd.read_parquet(PROC / "card_tags.parquet").columns if c.startswith("P_")]
TG = pd.read_parquet(PROC / "card_tags.parquet").set_index("oracle_id")
CODE_TXT = {"TOK": "tokens", "TRE": "treasure/food/clues", "ART": "artifacts", "ENC": "enchantments", "SPL": "instants/sorceries", "NCS": "noncreature spells",
            "CRE": "creatures entering", "GOW": "go-wide", "CTR": "+1/+1 counters", "OCT": "other counters", "GY": "self-mill/graveyard", "REC": "graveyard recursion",
            "BIG": "big creatures", "SAC": "sacrifice", "DIE": "death triggers", "ETB": "ETB effects", "LIF": "lifegain", "DRW": "card draw", "DIS": "discard/mill opponent",
            "DMG": "direct damage", "MAN": "extra mana", "LND": "lands/landfall", "XSP": "big mana sinks", "LEG": "legends", "EQA": "equipment/auras", "VEH": "vehicles",
            "EVA": "evasion", "ATK": "attack triggers", "TAP": "tap/untap", "CPY": "copying", "CST": "cost reduction", "FLS": "flash", "PSN": "poison", "NRG": "energy", "HND": "hand size"}
def why(a, b):
    out = []
    for x, y in ((a, b), (b, a)):
        if x in TG.index and y in TG.index:
            for c in TAGC:
                if TG.at[x, f"P_{c}"] and TG.at[y, f"R_{c}"]: out.append(f"{name[x]} gives {CODE_TXT.get(c, c)} -> {name[y]} rewards it")
    tr = (set(str(TG.tribes.get(a, "")).split(",")) | set(str(TG.tribes.get(b, "")).split(","))) - {"", "nan"}
    if tr: out.append("tribal: " + "/".join(sorted(tr)))
    return "; ".join(out[:3])

def info(o):
    r = single.loc[o]; return r["name"], r.pool_set, str(r.tier)[:2].strip(), r.percentile, r.current_std_play_pct

# ---------------------------------------------------------------- per-card partners
both = pd.concat([P[["a", "b", "rank_ab", "prob"]].rename(columns={"a": "focal", "b": "partner", "rank_ab": "rank"}),
                  P[["b", "a", "rank_ba", "prob"]].rename(columns={"b": "focal", "a": "partner", "rank_ba": "rank"})])
top = both.sort_values("rank", ascending=False).groupby("focal").head(12)
rows = []
for r in top.itertuples():
    n, lift = together(r.focal, r.partner)
    rows.append(dict(card=name[r.focal], partner=name[r.partner], ranker_score=round(r.rank, 3), strong_partner_prob=round(r.prob, 3),
                     decks_together_now=n, observed_lift=None if lift != lift else round(lift, 1), partner_tier=info(r.partner)[2]))
CP = pd.DataFrame(rows); CP.to_csv(INS / "card_partners.csv", index=False)

# ---------------------------------------------------------------- top pairs overall
GOOD = {"A", "B+", "B", "C"}
P["qa"] = P.a.map(single.percentile); P["qb"] = P.b.map(single.percentile)
TP = P.sort_values("prob", ascending=False).head(4000).copy()
recs = []
for r in TP.itertuples():
    n, lift = together(r.a, r.b)
    playA, playB = len(DS.get(r.a, ())) >= 2, len(DS.get(r.b, ())) >= 2
    status = "proven" if n >= 3 else ("hidden" if (playA and playB) else "untested")
    recs.append(dict(card_a=name[r.a], card_b=name[r.b], strong_partner_prob=round(r.prob, 3), status=status, decks_together_now=n,
                     observed_lift=None if lift != lift else round(lift, 1), tier_a=info(r.a)[2], tier_b=info(r.b)[2],
                     set_a=info(r.a)[1], set_b=info(r.b)[1], why=why(r.a, r.b)))
TPD = pd.DataFrame(recs); TPD.to_csv(INS / "top_pairs.csv", index=False)

# ---------------------------------------------------------------- packages (graph communities)
playable = {o for o in cand if str(single.tier[o])[:2].strip() in GOOD or single.current_std_play_pct[o] >= 2 or single.pool_set[o] in NEWSETS}
thr = P.prob.quantile(0.985)
E = P[(P.prob >= thr) & P.a.isin(playable) & P.b.isin(playable)]
G = nx.Graph(); G.add_weighted_edges_from(zip(E.a, E.b, E.prob))
comms = [c for c in louvain_communities(G, weight="weight", resolution=1.2, seed=0) if len(c) >= 5]
pk, mem = [], []
COLN = {1: "W", 2: "U", 4: "B", 8: "R", 16: "G"}
for k, c in enumerate(sorted(comms, key=len, reverse=True)):
    sub = G.subgraph(c); deg = dict(sub.degree(weight="weight"))
    core = sorted(c, key=lambda o: -deg[o])
    colmask = np.bitwise_or.reduce([M.CM[M.idx[o]] for o in core[:8]]) if core else 0
    cols = "".join(v for b, v in COLN.items() if colmask & b) or "C"
    tagsum = TG.reindex(list(c)).fillna(0)
    themes = (tagsum[[f"R_{t}" for t in TAGC]].sum().rename(lambda x: x[2:]) + 0.5 * tagsum[[f"P_{t}" for t in TAGC]].sum().rename(lambda x: x[2:]))
    themes = themes.drop(["CRE", "SPL", "LEG", "FLS"], errors="ignore").sort_values(ascending=False)
    tribe = pd.Series([t for o in c for t in str(TG.tribes.get(o, "")).split(",") if t and t != "nan"]).value_counts()
    theme_txt = ", ".join(CODE_TXT.get(t, t) for t in themes.index[:3]) + (f" | tribal: {tribe.index[0]}" if len(tribe) and tribe.iloc[0] >= 3 else "")
    played_share = np.mean([len(DS.get(o, ())) >= 2 for o in c])
    q = np.mean([single.percentile[o] for o in c]); dens = sub.size(weight="weight") / max(1, len(c))
    pk.append(dict(package=k + 1, colors=cols, size=len(c), themes=theme_txt, avg_card_percentile=round(q, 1),
                   synergy_density=round(dens, 2), share_already_played=round(played_share, 2),
                   status="established (mostly played now)" if played_share >= .5 else ("emerging" if played_share >= .2 else "untested / new"),
                   core_cards="; ".join(name[o] for o in core[:8])))
    for o in core:
        nm_, st, tr, pc, cur = info(o)
        mem.append(dict(package=k + 1, card=nm_, set=st, tier=tr, card_percentile=pc, current_play_pct=cur, centrality=round(deg[o], 2)))
PK = pd.DataFrame(pk); PK["package_score"] = (PK.synergy_density * (PK.avg_card_percentile / 100)).round(2)
PK = PK.sort_values("package_score", ascending=False); PK.to_csv(INS / "packages.csv", index=False)
pd.DataFrame(mem).to_csv(INS / "package_members.csv", index=False)

# ---------------------------------------------------------------- rotation impact
d26 = S.decks[(S.decks.format == "standard") & (S.decks.date >= "2026-01-01") & S.decks.oracle_id.isin(set(M.feat.index[M.feat.type_land == 0]))]
n26 = d26.deck.nunique(); share = d26.groupby("oracle_id").deck.nunique() / max(1, n26)
rot = pd.DataFrame({"card": share.index.map(name), "play_pct_2026": (share * 100).round(1).values, "survives_rotation": share.index.isin(pool)})
rot.sort_values("play_pct_2026", ascending=False).to_csv(INS / "rotation.csv", index=False)
deck_surv = d26.assign(s=d26.oracle_id.isin(pool)).groupby("deck").s.mean()

# ---------------------------------------------------------------- single-card highlights
sc = single.reset_index()
sc["color"] = sc.oid.map(lambda o: "".join(v for b, v in COLN.items() if M.CM[M.idx[o]] & b) if o in M.idx else "") .replace("", "C")
sc["color_group"] = sc.color.map(lambda c: c if len(c) == 1 else "Multi")
nl = sc[~sc.is_land]
nl.sort_values("percentile", ascending=False).groupby("color_group").head(8).sort_values(["color_group", "percentile"], ascending=[True, False])[
    ["color_group", "name", "pool_set", "tier", "percentile", "current_std_play_pct", "similar_past_cards"]].to_csv(INS / "single_card_best_by_color.csv", index=False)
sleep = nl[(nl.percentile >= 95) & (nl.current_std_play_pct < 1) & ~nl.reprint].sort_values("percentile", ascending=False)
sleep[["name", "pool_set", "tier", "percentile", "current_std_play_pct", "similar_past_cards", "top_reasons"]].to_csv(INS / "single_card_sleepers.csv", index=False)
dep = nl[(nl.percentile < 75) & (nl.current_std_play_pct >= 8)].sort_values("current_std_play_pct", ascending=False)
dep[["name", "pool_set", "tier", "percentile", "current_std_play_pct"]].to_csv(INS / "single_card_synergy_dependent.csv", index=False)
newc = sc[sc.pool_set.isin(NEWSETS)].sort_values("percentile", ascending=False)
newc[["name", "pool_set", "type_line", "tier", "percentile", "similar_past_cards"]].to_csv(INS / "single_card_new_spoilers.csv", index=False)

# ---------------------------------------------------------------- console summary
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
print(f"\nRecent Standard decks: {NREC} (2025+), 2026 decks: {n26}. Avg share of a 2026 deck's non-land cards that survive rotation: {deck_surv.mean():.0%}")
print("\nTOP ROTATING (most-played 2026 cards that leave):"); print(rot[~rot.survives_rotation].sort_values("play_pct_2026", ascending=False).head(12).to_string(index=False))
print("\nTOP SURVIVING:"); print(rot[rot.survives_rotation].sort_values("play_pct_2026", ascending=False).head(12).to_string(index=False))
print("\nPACKAGES:"); print(PK.drop(columns=["package"]).head(15).to_string(index=False))
for st in ["proven", "hidden", "untested"]:
    t = TPD[(TPD.status == st) & TPD.tier_a.isin(GOOD) & TPD.tier_b.isin(GOOD)]
    print(f"\nTOP {st.upper()} PAIRS (both tier C+):"); print(t.head(12)[["card_a", "card_b", "strong_partner_prob", "decks_together_now", "why"]].to_string(index=False))
print("\nSLEEPERS:"); print(sleep.head(15)[["name", "pool_set", "tier", "percentile", "similar_past_cards"]].to_string(index=False))
print("\nSYNERGY-DEPENDENT (low rating, high play):"); print(dep.head(12)[["name", "pool_set", "tier", "current_std_play_pct"]].to_string(index=False))
print("\nNEW SPOILERS:"); print(newc[["name", "pool_set", "tier", "percentile"]].to_string(index=False))
for o in [x for x in cand if single.pool_set[x] in NEWSETS][:10]:
    t = CP[CP.card == name[o]].head(6)
    print(f"\n{name[o]} best partners: " + "; ".join(t.partner))
