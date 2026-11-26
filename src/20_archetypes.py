"""ARCHETYPES of current Standard, what each loses at rotation, and the best replacements from the future pool.

1. Decks from the current format window (default: since 2026-06) -> non-land main-deck card sets -> TF-IDF ->
   SVD -> HDBSCAN clusters = archetypes. Each gets: share of decks, average finish, colours, core cards.
2. Rotation impact: share of the archetype's core (weighted by how often and how many copies) that is legal after
   rotation.
3. Replacements: for every rotating core card, rank future-legal cards in the archetype's colours by
   similarity to the rotating card + synergy with the surviving core (v3 probability model) + single-card rating.
Outputs: output/insights/archetypes.csv, archetype_cards.csv, archetype_replacements.csv
"""
import os, pathlib, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import HDBSCAN
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"; INS = OUT / "insights"
spec = importlib.util.spec_from_file_location("s12", ROOT / "src/12_synergy_v2.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
M = S.M
SINCE = os.environ.get("SINCE", "2026-06-01")
name = M.cards.name; feat = M.feat
pool = set(pd.read_csv(PROC / "future_pool.csv").oracle_id)
single = pd.read_csv(OUT / "post_rotation_scores.csv")
single["oid"] = single.name.map(feat.reset_index().drop_duplicates("name").set_index("name").oracle_id); single = single.drop_duplicates("oid").set_index("oid")
GUILD = {"W": "Mono-White", "U": "Mono-Blue", "B": "Mono-Black", "R": "Mono-Red", "G": "Mono-Green", "WU": "Azorius", "UB": "Dimir", "BR": "Rakdos",
         "RG": "Gruul", "WG": "Selesnya", "WB": "Orzhov", "UR": "Izzet", "BG": "Golgari", "WR": "Boros", "UG": "Simic", "WUB": "Esper", "UBR": "Grixis",
         "BRG": "Jund", "WRG": "Naya", "WUG": "Bant", "WBG": "Abzan", "WUR": "Jeskai", "UBG": "Sultai", "WBR": "Mardu", "URG": "Temur", "": "Colorless"}

d = pd.read_parquet(PROC / "decks.parquet")
d = d[(d.format == "standard") & (d.date >= SINCE)]
ev = pd.read_parquet(PROC / "events.parquet").set_index("event")
nonland = set(feat.index[feat.type_land == 0])
main = d[(d.main > 0) & d.oracle_id.isin(nonland)]
docs = main.groupby("deck").apply(lambda g: " ".join(f"{o} " * min(int(c), 4) for o, c in zip(g.oracle_id, g.main)), include_groups=False)
print(f"{len(docs)} decks since {SINCE}", flush=True)
X = TfidfVectorizer(token_pattern=r"[^ ]+", sublinear_tf=True).fit_transform(docs.values)
Z = TruncatedSVD(40, random_state=0).fit_transform(X); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
lab = HDBSCAN(min_cluster_size=max(6, len(docs) // 60), min_samples=3, metric="euclidean").fit_predict(Z)
decks = pd.DataFrame({"deck": docs.index, "arch": lab})
# finishing position: rank percentile within its event (0 = winner); unknown rank -> NaN
meta = d.drop_duplicates("deck").set_index("deck")
decks["event"] = decks.deck.map(meta.event); decks["rank"] = decks.deck.map(meta["rank"])
decks["n_event"] = decks.event.map(ev.n_decks).clip(lower=2)
decks["finish"] = (decks["rank"] - 1) / (decks.n_event - 1)
BIT = "WUBRG"
def colors_of(oids):
    cs = [c for o in oids for c in (M.cards.colors.get(o) if M.cards.colors.get(o) is not None else [])]
    s = pd.Series(cs).value_counts(); return "".join(c for c in BIT if s.get(c, 0) >= 0.12 * len(oids))
rows, card_rows, core_map = [], [], {}
allfreq = main.groupby("oracle_id").deck.nunique() / len(docs)
for a, g in decks[decks.arch >= 0].groupby("arch"):
    m = main[main.deck.isin(g.deck)]
    freq = m.groupby("oracle_id").deck.nunique() / len(g); copies = m.groupby("oracle_id").main.mean()
    core = freq[freq >= 0.5].sort_values(ascending=False)
    weight = (core * copies.reindex(core.index))
    surv = float(weight[[o in pool for o in core.index]].sum() / weight.sum()) if len(core) else np.nan
    col = colors_of(list(core.index))
    sig = (freq / allfreq.reindex(freq.index)).where(freq >= 0.4).dropna().sort_values(ascending=False)
    nm = f"{GUILD.get(col, col)} ({name[sig.index[0]]})" if len(sig) else GUILD.get(col, col)
    rows.append(dict(arch=a, archetype=nm, colors=col, decks=len(g), share_pct=round(100 * len(g) / len(docs), 1),
                     avg_finish_pct=round(100 * g.finish.mean(), 1) if g.finish.notna().any() else None,
                     top8_rate_pct=round(100 * (g["rank"] <= 8).mean(), 1), core_survival_pct=round(100 * surv, 1),
                     rotating_core=", ".join(name[o] for o in core.index if o not in pool),
                     surviving_core=", ".join(name[o] for o in core.index if o in pool)))
    core_map[a] = (col, list(core.index), freq, copies)
    for o, f in freq[freq >= 0.25].sort_values(ascending=False).items():
        card_rows.append(dict(archetype=nm, card=name[o], play_rate_in_archetype=round(f, 2), avg_copies=round(copies[o], 1), legal_after_rotation=o in pool))
A = pd.DataFrame(rows).sort_values("decks", ascending=False)
A.to_csv(INS / "archetypes.csv", index=False); pd.DataFrame(card_rows).to_csv(INS / "archetype_cards.csv", index=False)
print(f"{len(A)} archetypes, {100 * (decks.arch < 0).mean():.0f}% of decks unclustered")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(A.drop(columns=["arch"]).to_string(index=False))

# ---------------------------------------------------------------- replacements for rotating core cards
prob = lgb.Booster(model_file=str(OUT / "models/synergy_prob.txt")); FEATS = pd.read_csv(OUT / "models/synergy_features.csv").iloc[:, 0].tolist()
REAL = S.card_map("2099-01-01")
last12 = S.decks[(S.decks.format == "standard") & (S.decks.date >= "2025-11-01")]
POP = last12.groupby("oracle_id").deck.nunique() / max(1, last12.deck.nunique())
fpool = [o for o in pool if o in M.idx and o in nonland and o in single.index]
cross, pred, _ = S.text_to_map(REAL, sorted(set(fpool) - set(REAL)))
def synergy(cands, core):
    A_ = np.repeat(cands, len(core)); B_ = np.tile(core, len(cands))
    Xp = M.pair_features(A_, B_, np.full(len(A_), "standard"), np.full(len(A_), 2026))
    Xp = pd.concat([Xp, S.map_feats(A_, B_, REAL, cross, pred, False, S.RELD)], axis=1)
    Xp["cand_pop_prev"] = pd.Series(B_).map(POP).fillna(0).values
    return prob.predict(Xp[FEATS]).reshape(len(cands), len(core)).mean(1)
E = M.E; epos = M.epos
rep = []
for a, (col, core, freq, copies) in core_map.items():
    nm = A.set_index("arch").archetype[a]
    rot = [o for o in core if o not in pool]; keep = [o for o in core if o in pool]
    if not rot: continue
    allowed = set(col) or set(BIT)
    cands = [o for o in fpool if set(M.cards.colors.get(o) if M.cards.colors.get(o) is not None else []) <= allowed and o not in keep]
    if not cands or not keep: continue
    syn = synergy(np.array(cands), np.array(keep))
    q = np.array([single.percentile[o] for o in cands]) / 100
    for r in rot:
        sim = E[[epos[o] for o in cands]] @ E[epos[r]]
        mvd = np.abs(np.array([feat.mv[o] for o in cands]) - feat.mv[r])
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
        score = 1.0 * z(sim) + 0.8 * z(syn) + 0.6 * z(q) - 0.25 * mvd
        for j in np.argsort(-score)[:5]:
            rep.append(dict(archetype=nm, rotating_card=name[r], rotating_card_play=round(freq[r], 2), replacement=name[cands[j]],
                            similarity=round(float(sim[j]), 2), synergy_with_core=round(float(syn[j]), 3), replacement_rating_pct=round(100 * q[j], 1),
                            replacement_played_now_pct=float(single.current_std_play_pct[cands[j]]), score=round(float(score[j]), 2)))
R = pd.DataFrame(rep); R.to_csv(INS / "archetype_replacements.csv", index=False)
print("\nREPLACEMENTS (top 3 per rotating core card):")
for (nm, rc), g in R.groupby(["archetype", "rotating_card"], sort=False):
    print(f"  {nm} | {rc} -> " + "; ".join(f"{x.replacement} ({x.replacement_rating_pct:.0f})" for x in g.head(3).itertuples()))
