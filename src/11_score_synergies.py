"""Apply the synergy model to the post-rotation Standard pool and look for strong / hidden synergies.

status  proven    : already played together in winning Standard decks (>= 3 decks in our recent data)
        hidden    : both cards ARE played in Standard, but (almost) never together
        untested  : at least one of the two cards is not being played in Standard (yet)
"""
import importlib.util, pathlib, numpy as np, pandas as pd, lightgbm as lgb
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"
spec = importlib.util.spec_from_file_location("pm", ROOT / "src/10_pair_model.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

# ---- final model: all history
hist = M.aggregate(M.pq)
Xh = M.build(hist)
model = lgb.train(M.PARAMS, lgb.Dataset(Xh, hist.y, weight=np.sqrt(hist.support.values)), 600)

# ---- candidate cards: post-rotation pool, non-land, not in the bottom 20% of the single-card model
pool = pd.read_csv(OUT / "post_rotation_scores.csv")
name2id = pd.read_parquet(PROC / "features.parquet").set_index("name").oracle_id
pool["oid"] = pool.name.map(name2id)
pool = pool[pool.oid.isin(M.idx) & ~pool.is_land].drop_duplicates("oid")
dk = pd.read_parquet(PROC / "decks.parquet", columns=["deck", "format", "date", "oracle_id", "main"])
dk = dk[(dk.format == "standard") & (dk.date >= "2025-01-01") & (dk.main > 0)]
N_RECENT = dk.deck.nunique()
deck_sets = dk.groupby("oracle_id").deck.agg(set)
played = set(deck_sets.index[deck_sets.map(len) >= 2])
print("recent Standard decks:", N_RECENT, "| cards played in >= 2 of them:", len(played), flush=True)
cand = pool[(pool.percentile >= 20) | pool.oid.isin(played)]
ids = cand.oid.values; print("candidate cards:", len(ids), flush=True)
cm = np.array([M.CM[M.idx[o]] for o in ids])
ia, ib = np.triu_indices(len(ids), 1)
ok = M.POP[cm[ia] | cm[ib]] <= 3; ia, ib = ia[ok], ib[ok]
print("pairs to score:", len(ia), flush=True)
preds, contribs = np.zeros(len(ia), dtype=np.float32), []
names = None
for s in range(0, len(ia), 400_000):
    sl = slice(s, s + 400_000)
    X = M.pair_features(ids[ia[sl]], ids[ib[sl]], np.full(len(ia[sl]), "standard"), np.full(len(ia[sl]), 2027))
    names = list(X.columns); preds[sl] = model.predict(X)
# quality of each card from the single-card model (percentile within the pool)
pct = cand.set_index("oid").percentile
qa = np.array([pct[o] for o in ids])
qual = np.minimum(qa[ia], qa[ib])                           # the weaker card of the pair
pair_value = preds + 1.5 * (qual / 100)                      # synergy (log-lift) + playability of the weaker card
top_idx = np.unique(np.concatenate([np.argsort(-preds)[:20000], np.argsort(-pair_value)[:20000]]))
X = M.pair_features(ids[ia[top_idx]], ids[ib[top_idx]], np.full(len(top_idx), "standard"), np.full(len(top_idx), 2027))
C = model.predict(X, pred_contrib=True)[:, :-1]

LBL = {"role_tokens": "tokens → token payoff", "role_sac_fodder": "sacrifice fodder → sacrifice payoff", "role_counters": "+1/+1 counters → counter payoff",
       "role_artifacts": "artifacts → artifact payoff", "role_enchant": "enchantments → enchantment payoff", "role_spells": "cheap spells → spell payoff",
       "role_graveyard": "fills graveyard → graveyard payoff", "role_reanim_target": "big creature → reanimation", "role_lifegain": "lifegain → lifegain payoff",
       "role_mana_cre": "mana creature → mana payoff", "role_lands": "extra lands → landfall", "role_etb": "ETB creature → blink/ETB doubler",
       "role_dies": "death trigger → sacrifice outlet", "role_draw": "card draw → draw payoff", "role_legend": "legendary → legend payoff",
       "role_equip_aura": "equipment/aura → payoff", "role_vehicle": "creature → vehicle crew", "role_evasion": "evasive creature → combat-damage payoff",
       "role_cast_creature": "cheap creatures → creature-cast payoff", "role_big_power": "big creature → power-matters payoff",
       "role_opp_draw_discard": "mill/discard → payoff", "role_poison": "shared poison/toxic plan", "role_energy": "shared energy plan",
       "mention_max": "one card names the other's type/token/keyword", "mention_sum": "one card names the other's type/token/keyword",
       "shared_subtype": "shared creature type / mechanic", "both_payoff_same": "both reward the same thing", "both_source_same": "both provide the same resource",
       "emb_cos": "similar text/strategy", "same_colors": "same colours", "colors_shared": "colour overlap", "mv_min": "cheap pair (curve)",
       "mv_max": "curve", "n_creatures": "creature count", "n_inst_sorc": "spell count", "role_total": "several role matches"}
def why(k):
    c = C[k]; o = [j for j in np.argsort(-c) if c[j] > 0.02 and not names[j].startswith(("emb_prod", "emb_absdiff", "emb_sum", "fmt_", "year"))][:3]
    return "; ".join(dict.fromkeys(LBL.get(names[j], names[j]) for j in o))

def observed(a, b):
    A, B = deck_sets.get(a, set()), deck_sets.get(b, set())
    n = len(A & B); E_ = len(A) * len(B) / max(N_RECENT, 1)
    return n, ((n + 1) / (E_ + 1) if (A and B) else np.nan)
info = cand.set_index("oid")
rows = []
for k, j in enumerate(top_idx):
    a, b = ids[ia[j]], ids[ib[j]]; n, lift = observed(a, b)
    st = "proven" if n >= 3 else ("hidden" if (a in played and b in played) else "untested")
    rows.append(dict(card_a=info.name[a], card_b=info.name[b], predicted_synergy=round(float(np.exp(preds[j])), 2),
                     status=st, decks_together_recent=n, observed_lift=round(lift, 1) if lift == lift else None,
                     pair_value=round(float(pair_value[j]), 3), weaker_card_pct=round(float(qual[j]), 1),
                     tier_a=info.tier[a][:2].strip(), tier_b=info.tier[b][:2].strip(), set_a=info.pool_set[a], set_b=info.pool_set[b], why=why(k)))
R = pd.DataFrame(rows).sort_values("pair_value", ascending=False)
R.to_csv(OUT / "synergy_pairs_post_rotation.csv", index=False)
def diverse(df, cap=3, k=30):
    cnt, out = {}, []
    for r in df.itertuples():
        if cnt.get(r.card_a, 0) >= cap or cnt.get(r.card_b, 0) >= cap: continue
        cnt[r.card_a] = cnt.get(r.card_a, 0) + 1; cnt[r.card_b] = cnt.get(r.card_b, 0) + 1; out.append(r)
        if len(out) >= k: break
    return pd.DataFrame(out).drop(columns="Index")
GOOD = ["A", "B+", "B", "C"]
both_good = R[R.tier_a.isin(GOOD) & R.tier_b.isin(GOOD)]
cols = ["card_a", "card_b", "predicted_synergy", "status", "decks_together_recent", "observed_lift", "tier_a", "tier_b", "why"]
lists = {}
for st in ["proven", "hidden", "untested"]:
    lists[st] = diverse(both_good[both_good.status == st])
    print(f"\n=== {st.upper()}: both cards tier C or better, ranked by synergy + playability ===")
    print(lists[st][cols].head(25).to_string(index=False))
# build-arounds: a weak-looking card (D/F) whose synergy partner is good
ba = R[(R.tier_a.isin(["D", "F"]) & R.tier_b.isin(["A", "B+", "B"])) | (R.tier_b.isin(["D", "F"]) & R.tier_a.isin(["A", "B+", "B"]))]
lists["build_around"] = diverse(ba.sort_values("predicted_synergy", ascending=False), cap=2)
print("\n=== BUILD-AROUNDS: a low-rated card that the synergy model says shines next to a strong card ===")
print(lists["build_around"][cols].head(25).to_string(index=False))
pd.concat([v.assign(list=k) for k, v in lists.items()]).to_csv(OUT / "synergy_highlights.csv", index=False)
print("\nstatus counts:", R.status.value_counts().to_dict())

# ---- sanity checks on pairs we know about
def score_pair(x, y):
    a, b = name2id.get(x), name2id.get(y)
    if a is None or b is None or a not in M.idx or b not in M.idx: return None
    X = M.pair_features(np.array([a]), np.array([b]), np.array(["standard"]), np.array([2027]))
    v = float(model.predict(X)[0]); pr = (preds < v).mean() if len(preds) else np.nan
    return round(float(np.exp(v)), 2), round(100 * pr, 1), observed(a, b)
for x, y in [("Badgermole Cub", "Llanowar Elves"), ("Badgermole Cub", "Lightning Strike"), ("Llanowar Elves", "Elvish Mystic")]:
    print("check:", x, "+", y, "-> predicted synergy, percentile among all pool pairs, (decks together, observed lift):", score_pair(x, y))
