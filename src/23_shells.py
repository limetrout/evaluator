"""SHELLS: cores to build around (not full decks).

Archetype shells - every current archetype, post-rotation: engine (surviving core + best replacement for each rotating
core card), options (future cards in its colours with the highest predicted synergy with the engine, weighted by
rating), cheap interaction in its colours, fitting new spoilers.
Brew shells - new ideas: grow groups of 4-6 cards (rated B or better to start, C+ to join, <= 2 colours) whose members
all have high predicted strong-partner probability with each other; drop groups that mostly restate an existing
archetype's engine; then add options and interaction the same way.
Synergy = v3 probability model over the post-rotation pool (data/processed/future_pairs.parquet).
Outputs: output/insights/shells.json (+ shells.csv)
"""
import json, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"; INS = OUT / "insights"
feat = pd.read_parquet(PROC / "features.parquet").set_index("oracle_id"); cards = pd.read_parquet(PROC / "cards.parquet").set_index("oracle_id")
name = cards.name; nm2id = feat.reset_index().drop_duplicates("name").set_index("name").oracle_id
single = pd.read_csv(OUT / "post_rotation_scores.csv"); single["oid"] = single.name.map(nm2id); single = single.drop_duplicates("oid").set_index("oid")
tier = single.tier.astype(str).str[:2].str.strip(); pct = single.percentile; play = single.current_std_play_pct
TG = pd.read_parquet(PROC / "card_tags.parquet").set_index("oracle_id"); CODES = [c[2:] for c in TG.columns if c.startswith("P_")]
P = pd.read_parquet(PROC / "future_pairs.parquet")[["a", "b", "prob"]]
ids = sorted(set(P.a) | set(P.b)); ix = {o: i for i, o in enumerate(ids)}
PR = np.zeros((len(ids), len(ids)), dtype=np.float32)
PR[P.a.map(ix).values, P.b.map(ix).values] = P.prob.values; PR[P.b.map(ix).values, P.a.map(ix).values] = P.prob.values
GOOD3 = {"A", "B+", "B"}; GOOD4 = GOOD3 | {"C"}
def cols(o): c = cards.colors.get(o); return set(c) if c is not None else set()
CODE_TXT = {"TOK": "tokens", "TRE": "treasure/food/clues", "ART": "artifacts", "ENC": "enchantments", "SPL": "instants & sorceries", "NCS": "noncreature spells",
            "CRE": "creatures", "GOW": "go-wide", "CTR": "+1/+1 counters", "OCT": "counters", "GY": "graveyard", "REC": "recursion", "BIG": "big creatures",
            "SAC": "sacrifice", "DIE": "death triggers", "ETB": "enter effects", "LIF": "lifegain", "DRW": "card draw", "DIS": "discard", "DMG": "burn",
            "MAN": "ramp", "LND": "landfall", "XSP": "big spells", "LEG": "legends", "EQA": "equipment & auras", "VEH": "vehicles", "EVA": "evasion",
            "ATK": "attack triggers", "TAP": "tap/untap", "CPY": "copying", "CST": "cost reduction", "FLS": "flash", "PSN": "poison", "NRG": "energy", "HND": "hand size"}
def theme(members):
    t = TG.reindex(members).fillna(0)
    sc = t[[f"R_{c}" for c in CODES]].sum().rename(lambda x: x[2:]) + 0.5 * t[[f"P_{c}" for c in CODES]].sum().rename(lambda x: x[2:])
    sc = sc.drop(["CRE", "LEG", "FLS"], errors="ignore").sort_values(ascending=False)
    tr = pd.Series([x for o in members for x in str(TG.tribes.get(o, "")).split(",") if x and x != "nan"]).value_counts()
    words = [CODE_TXT[c] for c in sc.index[:2] if sc[c] >= 1.5]
    if len(tr) and tr.iloc[0] >= 2: words.insert(0, f"{tr.index[0].title()}s")
    return " + ".join(words) or "efficient threats"
GUILD = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green", "WU": "Azorius", "UB": "Dimir", "BR": "Rakdos", "RG": "Gruul", "WG": "Selesnya",
         "WB": "Orzhov", "UR": "Izzet", "BG": "Golgari", "WR": "Boros", "UG": "Simic", "": "Colorless"}
def colorname(cs): k = "".join(c for c in "WUBRG" if c in cs); return GUILD.get(k, k)
def card(o, note=""):
    return dict(n=name[o], tier=tier.get(o, ""), pct=round(float(pct.get(o, 0)), 0), play=float(play.get(o, 0)), set=single.pool_set.get(o, ""), note=note)
def options(engine, colorset, k=10, exclude=()):
    eng = [ix[o] for o in engine if o in ix]
    if not eng: return []
    cand = [o for o in ids if o not in engine and o not in exclude and cols(o) <= colorset and tier.get(o) in GOOD4]
    if not cand: return []
    syn = PR[np.ix_([ix[o] for o in cand], eng)].mean(1); q = np.array([pct.get(o, 0) for o in cand]) / 100
    sc = syn / (syn.max() + 1e-9) + 0.5 * q
    return [o for o in np.array(cand)[np.argsort(-sc)][:k]]
cheap_int = set(feat.index[(feat.cheap_interaction > 0) | ((feat.t_counter + feat.t_discard_opp) > 0) & (feat.mv <= 2)])
def interaction(colorset, exclude=(), k=6):
    c = [o for o in single.index if o in cheap_int and cols(o) and cols(o) <= colorset and o not in exclude and not single.is_land.get(o, False)]
    return sorted(c, key=lambda o: -(pct.get(o, 0) + 2 * play.get(o, 0)))[:k]

shells = []
# ------------------------------------------------------------------ archetype shells
A = pd.read_csv(INS / "meta_projection.csv"); AC = pd.read_csv(INS / "archetype_cards.csv"); RP = pd.read_csv(INS / "archetype_replacements.csv")
pool = set(pd.read_csv(PROC / "future_pool.csv").oracle_id)
arch_engines = []
for a in A.itertuples():
    ac = AC[AC.archetype == a.archetype]
    core = ac[(ac.play_rate_in_archetype >= 0.5)]; flex = ac[(ac.play_rate_in_archetype >= 0.25) & (ac.play_rate_in_archetype < 0.5) & ac.legal_after_rotation]
    eng = [(nm2id[c], f"in {int(round(100 * r))}% of lists now") for c, r in zip(core.card, core.play_rate_in_archetype) if c in nm2id and nm2id[c] in pool]
    rep = RP[RP.archetype == a.archetype].sort_values("score", ascending=False).drop_duplicates("rotating_card")
    used = {o for o, _ in eng}
    for r in rep.itertuples():
        o = nm2id.get(r.replacement)
        if o is not None and o not in used: eng.append((o, f"replaces {r.rotating_card}")); used.add(o)
    colorset = set(a.colors) if isinstance(a.colors, str) else set()
    engine = [o for o, _ in eng]
    opts = options(engine, colorset, 10, exclude=set(engine))
    newc = [o for o in options(engine, colorset, 40, exclude=set(engine)) if single.pool_set.get(o) in ("NAU", "MDD")][:3]
    arch_engines.append(set(engine))
    shells.append(dict(kind="archetype", name=a.archetype, colors="".join(c for c in "WUBRG" if c in colorset), rank=int(a.projected_rank),
        summary=f"{a.share_now_pct}% of the field now, keeps {int(a.core_survival_pct)}% of its core" + (f"; loses {a.biggest_losses}" if isinstance(a.biggest_losses, str) and a.biggest_losses else "; loses nothing"),
        theme=theme(engine), engine=[card(o, n) for o, n in eng],
        current_flex=[card(nm2id[c], f"in {int(round(100 * r))}% of lists now") for c, r in zip(flex.card, flex.play_rate_in_archetype) if c in nm2id][:6],
        options=[card(o) for o in opts], interaction=[card(o) for o in interaction(colorset, set(engine))], new_cards=[card(o) for o in newc]))

# ------------------------------------------------------------------ brew shells
starters = [o for o in ids if tier.get(o) in GOOD3]
joinable = np.array([tier.get(o) in GOOD4 for o in ids])
S_ = [ix[o] for o in starters]
sub = PR[np.ix_(S_, S_)].copy(); np.fill_diagonal(sub, 0)
pairs = np.dstack(np.unravel_index(np.argsort(-sub, axis=None)[:3000], sub.shape))[0]
brews, seen = [], []
for i, j in pairs:
    if i >= j: continue
    a_, b_ = starters[i], starters[j]
    group = [a_, b_]; cs = cols(a_) | cols(b_)
    if len(cs) > 2: continue
    while len(group) < 6:
        gi = [ix[o] for o in group]
        score = PR[:, gi].min(1) * 0.5 + PR[:, gi].mean(1) * 0.5
        score[gi] = -1; score[~joinable] = -1
        for k_, o in enumerate(ids):
            if score[k_] > 0 and len(cs | cols(o)) > 2: score[k_] = -1
        k_ = int(np.argmax(score))
        if score[k_] < 0.25: break
        group.append(ids[k_]); cs |= cols(ids[k_])
    if len(group) < 4: continue
    g = set(group)
    if any(len(g & s) / len(g) >= 0.5 for s in arch_engines) or any(len(g & s) / len(g) > 0.34 for s in seen): continue
    seen.append(g)
    gi = [ix[o] for o in group]; cohes = float(PR[np.ix_(gi, gi)][np.triu_indices(len(gi), 1)].mean())
    quality = float(np.mean([pct.get(o, 0) for o in group]))
    played_now = float(np.mean([play.get(o, 0) > 0 for o in group]))
    brews.append(dict(group=group, colors=cs, cohesion=cohes, quality=quality, played_now=played_now))
    if len(brews) >= 40: break
brews = sorted(brews, key=lambda b: -(b["cohesion"] * b["quality"] / 100))[:16]
for k, b in enumerate(brews):
    eng = b["group"]; cs = b["colors"]
    opts = options(eng, cs, 10, exclude=set(eng))
    shells.append(dict(kind="brew", name=f"{colorname(cs)} {theme(eng)}", colors="".join(c for c in "WUBRG" if c in cs), rank=k + 1,
        summary=f"cohesion {b['cohesion']:.2f} (avg predicted strong-partner probability inside the shell), average card rating {b['quality']:.0f}th percentile, "
                f"{int(round(100 * b['played_now']))}% of its cards see play now",
        theme=theme(eng), engine=[card(o) for o in eng], current_flex=[], options=[card(o) for o in opts],
        interaction=[card(o) for o in interaction(cs, set(eng))],
        new_cards=[card(o) for o in options(eng, cs, 40, exclude=set(eng)) if single.pool_set.get(o) in ("NAU", "MDD")][:3]))
(INS / "shells.json").write_text(json.dumps(shells, ensure_ascii=False, indent=1))
rows = [dict(kind=s["kind"], rank=s["rank"], shell=s["name"], colors=s["colors"], engine="; ".join(c["n"] for c in s["engine"]),
             options="; ".join(c["n"] for c in s["options"]), interaction="; ".join(c["n"] for c in s["interaction"]), summary=s["summary"]) for s in shells]
pd.DataFrame(rows).to_csv(INS / "shells.csv", index=False)
for s in shells:
    print(f"\n[{s['kind']} #{s['rank']}] {s['name']}  ({s['colors']}) - {s['summary']}")
    print("   ENGINE: " + "; ".join(f"{c['n']}" + (f" ({c['note']})" if c['note'] else "") for c in s["engine"]))
    print("   OPTIONS: " + "; ".join(c["n"] for c in s["options"][:8]))
    print("   INTERACTION: " + "; ".join(c["n"] for c in s["interaction"]))
    if s["new_cards"]: print("   NEW: " + "; ".join(c["n"] for c in s["new_cards"]))
