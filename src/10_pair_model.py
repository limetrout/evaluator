"""SYNERGY MODEL, step 2: learn which card traits make two cards go together, then test it forward in time.

Pair features (all symmetric, all from the printed cards, so they work for brand-new cards):
  role matches   card A PROVIDES something card B REWARDS (tokens -> token payoffs, +1/+1 counters -> counter
                 payoffs, artifacts -> artifact payoffs, cheap spells -> spell payoffs, self-mill -> graveyard payoffs,
                 lifegain -> lifegain triggers, mana creatures -> mana payoffs, ETB creatures -> blink, ...)
  mentions       B's text names A's creature type / card type / token type / keyword ("Elves you control")
  similarity     semantic similarity of the two cards' text (language-model embeddings) + a few embedding axes
  context        colour overlap, mana curve, card types, format, era
Target: log lift = log((n+1)/(E+1)) over the pair's first 4 quarters together (see 09_pair_targets.py).
Backtest: for each year Y, train only on co-occurrence observed BEFORE Y, predict pairs that involve a card
released in year Y. Nothing from the future is used.
"""
import re, sys, pathlib, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"
feat = pd.read_parquet(PROC / "features.parquet"); cards = pd.read_parquet(PROC / "cards.parquet").set_index("oracle_id")
feat = feat.set_index("oracle_id")
T = feat.clean_text.fillna(""); TL = feat.type_line.fillna("")

# ---------------------------------------------------------------- per-card roles (what it provides / rewards)
def has(rx): return T.str.contains(rx, regex=True).astype(np.int8)
is_cre, is_art, is_ench = feat.type_creature, feat.type_artifact, feat.type_enchantment
is_spell = ((feat.type_instant + feat.type_sorcery) > 0).astype(np.int8)
S, P = {}, {}
S["tokens"] = (feat.t_token > 0).astype(np.int8)
P["tokens"] = has(r"whenever (?:a|another|one or more) (?:\w+ )?(?:creature )?tokens?|tokens? you control|for each (?:creature )?token|whenever (?:a|another|one or more) (?:other )?(?:nontoken )?creatures? (?:you control )?enters?|creatures you control get \+")
S["sac_fodder"] = ((feat.t_token > 0) | (feat.t_treasure > 0) | (feat.t_clue_food_blood > 0) | (feat.t_dies_trig > 0) | ((is_cre == 1) & (feat.mv <= 1))).astype(np.int8)
P["sac_fodder"] = ((feat.t_sac_outlet > 0) | (feat.t_other_dies > 0) | has(r"whenever you sacrifice")).astype(np.int8)
S["counters"] = (feat.t_counters_p1 > 0).astype(np.int8)
P["counters"] = ((feat.t_proliferate > 0) | has(r"with (?:a|one or more) \+1/\+1 counters? on|counters? (?:would be )?(?:put|placed) on|one or more \+1/\+1 counters|for each \+1/\+1 counter|that many plus one|twice that many")).astype(np.int8)
S["artifacts"] = ((is_art == 1) | has(r"(?:treasure|clue|food|blood|map|powerstone|gold|junk|incubator) tokens?|artifact tokens?")).astype(np.int8)
P["artifacts"] = ((feat.t_artifact_matters > 0) | has(r"affinity for artifacts|improvise|metalcraft|for each artifact|whenever (?:an?|another|one or more) (?:nontoken )?artifacts?")).astype(np.int8)
S["enchant"] = (is_ench == 1).astype(np.int8)
P["enchant"] = ((feat.t_enchant_matters > 0) | has(r"whenever (?:an?|another) enchantment|for each enchantment|constellation")).astype(np.int8)
S["spells"] = ((is_spell == 1) & (feat.mv <= 3)).astype(np.int8)
P["spells"] = ((feat.t_spells_matters > 0) | (feat.get("kw_prowess", 0) > 0) | has(r"magecraft|whenever you cast (?:an instant|a noncreature|your second spell)|copy target instant")).astype(np.int8)
S["graveyard"] = ((feat.t_mill_self > 0) | (feat.t_loot > 0) | (feat.t_surveil > 0) | (feat.t_discard_self > 0) | has(r"discard (?:a|two|your hand)")).astype(np.int8)
P["graveyard"] = ((feat.t_cast_from_gy > 0) | (feat.t_reanimate > 0) | has(r"cards? in your graveyard|delve|escape|threshold|delirium|from your graveyard|card types among cards in your graveyard")).astype(np.int8)
S["reanim_target"] = ((is_cre == 1) & (feat.mv >= 5)).astype(np.int8)
P["reanim_target"] = (feat.t_reanimate > 0).astype(np.int8)
S["lifegain"] = ((feat.t_gain_life > 0) | (feat.get("kw_lifelink", 0) > 0)).astype(np.int8)
P["lifegain"] = (feat.t_lifegain_trig > 0).astype(np.int8)
S["mana_cre"] = ((is_cre == 1) & (feat.t_add_mana > 0)).astype(np.int8)
P["mana_cre"] = (has(r"tap (?:an untapped )?(?:\w+ )?creatures? (?:you control )?for mana|whenever you tap a creature|creature for mana") | ((feat.has_x > 0) & (feat.mv >= 1)) | (feat.mv >= 6)).astype(np.int8)
S["lands"] = ((feat.t_ramp_land > 0) | (feat.t_tutor_land > 0) | has(r"additional land|return (?:target|a) land|land cards? from your graveyard")).astype(np.int8)
P["lands"] = (feat.t_landfall > 0).astype(np.int8)
S["etb"] = ((is_cre == 1) & (feat.t_etb > 0)).astype(np.int8)
P["etb"] = has(r"exile (?:target|another|up to \w+ target|each)[^.]{0,40}you control[^.]{0,60}return|blink|flicker|triggers an additional time|enters[^.]{0,40}(?:causes|triggers) a triggered ability")
S["dies"] = (feat.t_dies_trig > 0).astype(np.int8)
P["dies"] = (feat.t_sac_outlet > 0).astype(np.int8)
S["draw"] = (feat.t_draw > 0).astype(np.int8)
P["draw"] = (feat.t_draw_trig > 0).astype(np.int8)
S["legend"] = feat.legendary.astype(np.int8)
P["legend"] = (feat.t_legend_matters > 0).astype(np.int8) | has(r"historic").astype(np.int8)
S["equip_aura"] = ((feat.sub_equipment + feat.sub_aura) > 0).astype(np.int8)
P["equip_aura"] = has(r"equipped creature you control|equipment you control|auras? you control|enchanted creatures? you control|for each (?:aura|equipment)|equipment spells|aura spells")
S["vehicle"] = ((is_cre == 1) & (feat.power.fillna(0) >= 2)).astype(np.int8)
P["vehicle"] = feat.sub_vehicle.astype(np.int8)
S["evasion"] = ((is_cre == 1) & (feat.evasive > 0)).astype(np.int8)
P["evasion"] = has(r"whenever (?:a|one or more) creatures? you control deals? combat damage to a player|equipped creature|enchanted creature")
S["cast_creature"] = ((is_cre == 1) & (feat.mv <= 3)).astype(np.int8)
P["cast_creature"] = has(r"whenever you cast a creature spell|creature spells you cast cost")
S["big_power"] = ((is_cre == 1) & (feat.power.fillna(0) >= 4)).astype(np.int8)
P["big_power"] = has(r"power 4 or greater|greatest power|ferocious|total power")
S["opp_draw_discard"] = has(r"each opponent discards|target opponent discards|each opponent mills|target player mills")
P["opp_draw_discard"] = has(r"cards in (?:that player's|an opponent's|each opponent's) (?:hand|graveyard)|whenever an opponent (?:discards|mills)")
S["poison"] = has(r"toxic|poison|infect") ; P["poison"] = S["poison"]
S["energy"] = has(r"\{e\}"); P["energy"] = S["energy"]
ROLES = list(S)
Smat = np.stack([S[r].reindex(feat.index).fillna(0).values for r in ROLES], 1).astype(np.int8)
Pmat = np.stack([np.asarray(P[r].reindex(feat.index).fillna(0)) for r in ROLES], 1).astype(np.int8)

# ---- "mentions": words naming card A (its creature types, card types, tokens it makes, keywords) that appear in B's text
STOP = {"creature", "card", "land", "instant", "sorcery", "legendary", "basic", "spell", "token"}
def sing(w): return w[:-1] if w.endswith("s") and not w.endswith("ss") else w
def names_of(o):
    tl = TL[o].lower(); ws = set()
    for part in tl.split("//"):
        t = part.split("—"); ws |= {w for w in t[0].split() if w not in STOP}
        if len(t) > 1: ws |= set(t[1].split())
    ws |= set(re.findall(r"create [^.]{0,40}?(\w+) (?:artifact )?(?:creature )?tokens?", T[o]))
    ws |= {k.lower() for k in (cards.keywords.get(o) if o in cards.index and cards.keywords.get(o) is not None else [])}
    return {sing(w) for w in ws if len(w) > 2}
idx = {o: i for i, o in enumerate(feat.index)}
NAMES = [names_of(o) for o in feat.index]
TEXTW = [{sing(w) for w in re.findall(r"[a-z\-]+", T[o])} for o in feat.index]
COMMON_KW = {"flying", "trample", "haste", "vigilance", "reach", "flash", "deathtouch", "lifelink", "first strike", "menace"}

E = np.load(PROC / "embeddings.npy"); eid = pd.read_csv(PROC / "embeddings_ids.csv").oracle_id
epos = {o: i for i, o in enumerate(eid)}
from sklearn.decomposition import PCA
PCS = PCA(16, random_state=0).fit_transform(E)
F = feat
BIT = {"W": 1, "U": 2, "B": 4, "R": 8, "G": 16}
CM = np.array([sum(BIT[c] for c in (cards.colors.get(o) if o in cards.index and cards.colors.get(o) is not None else []) if c in BIT) for o in feat.index])
POP = np.array([bin(x).count("1") for x in range(32)])

def pair_features(a_ids, b_ids, fmt=None, year=None):
    ia = np.array([idx[o] for o in a_ids]); ib = np.array([idx[o] for o in b_ids])
    X = {}
    m_ab = Smat[ia] * Pmat[ib]; m_ba = Smat[ib] * Pmat[ia]
    for k, r in enumerate(ROLES): X[f"role_{r}"] = np.maximum(m_ab[:, k], m_ba[:, k])
    X["role_total"] = (np.maximum(m_ab, m_ba)).sum(1)
    X["both_payoff_same"] = (Pmat[ia] * Pmat[ib]).sum(1)          # two payoffs for the same thing
    X["both_source_same"] = (Smat[ia] * Smat[ib]).sum(1)
    men_ab = np.array([len(NAMES[i] & TEXTW[j]) for i, j in zip(ia, ib)]); men_ba = np.array([len(NAMES[j] & TEXTW[i]) for i, j in zip(ia, ib)])
    X["mention_max"] = np.maximum(men_ab, men_ba); X["mention_sum"] = men_ab + men_ba
    X["shared_subtype"] = np.array([len((NAMES[i] & NAMES[j]) - COMMON_KW) for i, j in zip(ia, ib)])
    ea = E[[epos[o] for o in a_ids]]; eb = E[[epos[o] for o in b_ids]]
    X["emb_cos"] = (ea * eb).sum(1)
    pa = PCS[[epos[o] for o in a_ids]]; pb = PCS[[epos[o] for o in b_ids]]
    for j in range(8): X[f"emb_prod{j}"] = pa[:, j] * pb[:, j]; X[f"emb_absdiff{j}"] = np.abs(pa[:, j] - pb[:, j]); X[f"emb_sum{j}"] = pa[:, j] + pb[:, j]
    ca, cb = CM[ia], CM[ib]
    X["colors_shared"] = POP[ca & cb]; X["colors_union"] = POP[ca | cb]; X["same_colors"] = (ca == cb).astype(int); X["any_colorless"] = ((ca == 0) | (cb == 0)).astype(int)
    mva, mvb = F.mv.values[ia], F.mv.values[ib]
    X["mv_min"] = np.minimum(mva, mvb); X["mv_max"] = np.maximum(mva, mvb); X["mv_diff"] = np.abs(mva - mvb)
    ca_, cb_ = F.type_creature.values[ia], F.type_creature.values[ib]
    X["n_creatures"] = ca_ + cb_
    sa, sb = is_spell.values[ia], is_spell.values[ib]; X["n_inst_sorc"] = sa + sb
    X["rarity_max"] = np.maximum(F.rarity.values[ia], F.rarity.values[ib]); X["rarity_min"] = np.minimum(F.rarity.values[ia], F.rarity.values[ib])
    X["interaction_n"] = (F.interaction.values[ia] > 0).astype(int) + (F.interaction.values[ib] > 0).astype(int)
    X["legend_n"] = F.legendary.values[ia] + F.legendary.values[ib]
    X["text_len_min"] = np.minimum(F.n_words.values[ia], F.n_words.values[ib])
    X["triggers_sum"] = F.n_triggered.values[ia] + F.n_triggered.values[ib]
    if fmt is not None:
        for f in ["standard", "pioneer", "modern"]: X[f"fmt_{f}"] = (np.asarray(fmt) == f).astype(int)
        X["year"] = np.asarray(year, dtype=float)
    return pd.DataFrame(X)

# ---------------------------------------------------------------- targets per pair (first 4 quarters together)
rel = pd.to_datetime(cards.first_release).dt.year * 10 + (pd.to_datetime(cards.first_release).dt.month - 1) // 3 + 1
FSTART = {"standard": 20161, "pioneer": 20194, "modern": 20161}
def qadd(q, k): y, r = divmod(q, 10); t = y * 4 + (r - 1) + k; return (t // 4) * 10 + t % 4 + 1
pq = pd.read_parquet(PROC / "pair_quarters.parquet")
pq = pq[pq.a.isin(idx) & pq.b.isin(idx)]
pq["start"] = np.maximum(np.maximum(pq.a.map(rel).values, pq.b.map(rel).values), pq.format.map(FSTART).values)
pq["end"] = [qadd(s, 4) for s in pq.start.values]
pq = pq[(pq.q >= pq.start) & (pq.q < pq.end)]

def aggregate(rows):
    g = rows.groupby(["format", "a", "b"], sort=False).agg(n=("n", "sum"), E=("E", "sum"), start=("start", "first")).reset_index()
    g["y"] = np.log((g.n + 1) / (g.E + 1)); g["support"] = g.n + g.E
    return g[g.support >= 3]

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4)
USE_PROFILES = False   # tested: slightly better overall ranking, but worse top picks (63% -> 59% strong in top 2%) -> off

def partner_profiles(hist):
    """For each card seen so far: a lift-weighted average of the embeddings of the cards it has gone WITH
    (lift >= 1.5, seen together in >= 3 decks). Built only from `hist`, i.e. data from the past."""
    h = hist[(hist.y >= np.log(1.5)) & (hist.n >= 3)]
    w = h.y.values
    A = np.concatenate([h.a.values, h.b.values]); B = np.concatenate([h.b.values, h.a.values]); W = np.concatenate([w, w])
    df_ = pd.DataFrame({"c": A, "p": B, "w": W})
    df_["e"] = df_.p.map(epos)
    prof = {}
    for c, g in df_.groupby("c"):
        if len(g) < 2: continue
        v = (E[g.e.values] * g.w.values[:, None]).sum(0); prof[c] = v / (np.linalg.norm(v) + 1e-9)
    keys = np.array(list(prof)); M = np.stack([prof[k] for k in keys]) if len(keys) else np.zeros((0, E.shape[1]))
    return prof, keys, M

def profile_feats(a_ids, b_ids, prof, keys, M, rel_lookup):
    """cos(card A, B's partner profile) and vice versa; for cards with no history use their closest older analog's profile."""
    ea = E[[epos[o] for o in a_ids]]; eb = E[[epos[o] for o in b_ids]]
    def prof_or_analog(ids, emb):
        P_ = np.zeros_like(emb); direct = np.zeros(len(ids))
        miss = []
        for i, o in enumerate(ids):
            if o in prof: P_[i] = prof[o]; direct[i] = 1
            else: miss.append(i)
        if miss and len(keys):
            kemb = E[[epos[k] for k in keys]]
            S_ = emb[miss] @ kemb.T
            j = S_.argmax(1); P_[miss] = M[j]
        return P_, direct
    pa, da = prof_or_analog(a_ids, ea); pb, db = prof_or_analog(b_ids, eb)
    ab = (ea * pb).sum(1); ba = (eb * pa).sum(1)
    return pd.DataFrame({"prof_max": np.maximum(ab, ba), "prof_min": np.minimum(ab, ba), "prof_sum": ab + ba,
                         "prof_direct_n": da + db})

def build(g, prof=None):
    X = pair_features(g.a.values, g.b.values, g.format.values, g.start.values // 10)
    if prof is not None: X = pd.concat([X, profile_feats(g.a.values, g.b.values, *prof, None)], axis=1)
    return X

if __name__ == "__main__":
    res, allp = [], []
    for Y in range(2019, 2025):
        cut = Y * 10 + 1
        train = aggregate(pq[pq.q < cut])                          # only co-occurrence observed before year Y
        train = train[train.start < cut]
        test = aggregate(pq[(pq.start >= cut) & (pq.start < cut + 10)])   # pairs that only became possible in year Y
        if len(test) < 200: continue
        # partner profiles from the past only; for training rows use profiles built WITHOUT the pair's own quarter-window
        # (approximation: profiles from data older than each training pair would be ideal; we use a 1-year-earlier cut)
        if USE_PROFILES:
            prof_tr = partner_profiles(aggregate(pq[pq.q < cut - 10])); prof_te = partner_profiles(train)
            Xtr, Xte = build(train, prof_tr), build(test, prof_te)
        else:
            Xtr, Xte = build(train), build(test)
        w = np.sqrt(train.support.values)
        m = lgb.train(PARAMS, lgb.Dataset(Xtr, train.y, weight=w), 600)
        p = m.predict(Xte)
        base_cols = ["emb_cos", "colors_shared", "colors_union", "same_colors", "any_colorless"]
        mb = lgb.train(PARAMS, lgb.Dataset(Xtr[base_cols], train.y, weight=w), 300); pb = mb.predict(Xte[base_cols])
        heur = Xte.role_total.values + Xte.mention_max.values + 0.5 * Xte.shared_subtype.values + 0.01 * Xte.emb_cos.values
        test = test.assign(pred=p, base=pb, heur=heur, year=Y); allp.append(test)
        strong = ((test.n >= 5) & (np.exp(test.y) >= 2)).values
        r = {"year": Y, "test_pairs": len(test), "strong_pairs": int(strong.sum())}
        for nm, v in [("model", p), ("baseline_similarity+colour", pb), ("baseline_role_rules", heur)]:
            r[f"{nm}_spearman"] = round(spearmanr(test.y, v).correlation, 3)
            o = np.argsort(-v); k = max(1, len(v) // 50)
            r[f"{nm}_top2pct_strong"] = round(strong[o[:k]].mean(), 3)
            r[f"{nm}_strong_in_top10pct"] = round(strong[o[: len(v) // 10]].sum() / max(1, strong.sum()), 3)
        res.append(r); print(r, flush=True)
    R = pd.DataFrame(res).set_index("year"); R.to_csv(OUT / "synergy_backtest.csv")
    A = pd.concat(allp); A.to_parquet(PROC / "synergy_backtest_preds.parquet")
    print("\nbase rate of strong synergy among test pairs:", round(((A.n >= 5) & (np.exp(A.y) >= 2)).mean(), 3))
    print(R.T.to_string())
