"""Find synergy PACKAGES (candidate archetypes) in the post-rotation pool.

Graph: nodes = playable cards (tier C or better, or currently played >= 2%, or new spoilers);
edges = the strongest 0.2% of predicted strong-partner probabilities among them, each card keeping its 10 strongest
links; Louvain communities (resolution 2.5) = packages. Each package is described by its colours, the roles its
cards reward/provide (Claude tags), tribal focus, average single-card rating, and how much of it is already played.
"""
import pathlib, numpy as np, pandas as pd, networkx as nx
from networkx.algorithms.community import louvain_communities
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"; INS = OUT / "insights"
P = pd.read_parquet(PROC / "future_pairs.parquet")
single = pd.read_csv(OUT / "post_rotation_scores.csv")
feat = pd.read_parquet(PROC / "features.parquet")
nm2id = feat.drop_duplicates("name").set_index("name").oracle_id; single["oid"] = single.name.map(nm2id); single = single.drop_duplicates("oid").set_index("oid")
name = feat.set_index("oracle_id").name
cards = pd.read_parquet(PROC / "cards.parquet").set_index("oracle_id")
TG = pd.read_parquet(PROC / "card_tags.parquet").set_index("oracle_id"); TAGC = [c[2:] for c in TG.columns if c.startswith("P_")]
CODE_TXT = {"TOK": "tokens", "TRE": "treasure/food/clues", "ART": "artifacts", "ENC": "enchantments", "NCS": "noncreature spells", "GOW": "go-wide",
            "CTR": "+1/+1 counters", "OCT": "other counters", "GY": "self-mill", "REC": "graveyard recursion", "BIG": "big creatures", "SAC": "sacrifice",
            "DIE": "death triggers", "ETB": "ETB effects", "LIF": "lifegain", "DRW": "card draw", "DIS": "discard/mill", "DMG": "direct damage",
            "MAN": "ramp", "LND": "landfall", "XSP": "big mana sinks", "EQA": "equipment/auras", "VEH": "vehicles", "EVA": "evasion",
            "ATK": "attack triggers", "TAP": "tap/untap", "CPY": "copying", "CST": "cost reduction", "PSN": "poison", "NRG": "energy", "HND": "hand size"}
GOOD = {"A", "B+", "B", "C"}
tier = single.tier.astype(str).str[:2].str.strip()
playable = set(single.index[tier.isin(GOOD) | (single.current_std_play_pct >= 2) | single.pool_set.isin(["NAU", "MDD"])])
Q = P[P.a.isin(playable) & P.b.isin(playable)]
E = Q[Q.prob >= Q.prob.quantile(0.998)]
both = pd.concat([E[["a", "b", "prob"]], E.rename(columns={"a": "b", "b": "a"})[["a", "b", "prob"]]])
keep = both.sort_values("prob", ascending=False).groupby("a").head(10)
G = nx.Graph(); G.add_weighted_edges_from(zip(keep.a, keep.b, keep.prob))
comms = [c for c in louvain_communities(G, weight="weight", resolution=2.5, seed=0) if len(c) >= 5]
BIT = {"W": 1, "U": 2, "B": 4, "R": 8, "G": 16}
def cmask(o): cs = cards.colors.get(o); return sum(BIT[c] for c in (cs if cs is not None else []) if c in BIT)
pk, mem = [], []
for c in comms:
    sub = G.subgraph(c); deg = dict(sub.degree(weight="weight")); core = sorted(c, key=lambda o: -deg[o])
    cnt = pd.Series([v for o in core for v, b in BIT.items() if cmask(o) & b]).value_counts()
    cols = "".join(sorted([v for v, n in cnt.items() if n >= 0.25 * len(c)], key="WUBRG".index)) or "C"
    tg = TG.reindex(list(c)).fillna(0)
    th = (tg[[f"R_{t}" for t in TAGC]].sum().rename(lambda x: x[2:]) * 1.0 + tg[[f"P_{t}" for t in TAGC]].sum().rename(lambda x: x[2:]) * 0.4)
    th = th.drop(["CRE", "SPL", "LEG", "FLS"], errors="ignore") / len(c)
    th = th[th >= 0.15].sort_values(ascending=False)
    tribe = pd.Series([t for o in c for t in str(TG.tribes.get(o, "")).split(",") if t and t != "nan"]).value_counts()
    theme = ", ".join(CODE_TXT.get(t, t) for t in th.index[:3]) or "efficient cards"
    if len(tribe) and tribe.iloc[0] >= 3: theme += f" | {tribe.index[0]} tribal"
    played = np.mean([single.current_std_play_pct.get(o, 0) >= 1 for o in c])
    q = np.mean([single.percentile[o] for o in c]); dens = sub.size(weight="weight") / len(c)
    newc = [name[o] for o in c if single.pool_set.get(o) in ("NAU", "MDD")]
    pk.append(dict(colors=cols, size=len(c), themes=theme, avg_card_percentile=round(q, 1), synergy_density=round(dens, 2),
                   share_played_now=round(played, 2),
                   status="established" if played >= .5 else ("emerging" if played >= .2 else "untested"),
                   core_cards="; ".join(name[o] for o in core[:10]), new_spoilers="; ".join(newc), members_=core))
PK = pd.DataFrame(pk); PK["package_score"] = (PK.synergy_density * PK.avg_card_percentile / 100).round(2)
PK = PK.sort_values("package_score", ascending=False).reset_index(drop=True); PK.insert(0, "package", PK.index + 1)
for r in PK.itertuples():
    for o in r.members_:
        mem.append(dict(package=r.package, card=name[o], set=single.pool_set.get(o), tier=tier.get(o), card_percentile=single.percentile.get(o),
                        current_play_pct=single.current_std_play_pct.get(o), links_in_package=round(dict(G.subgraph(r.members_).degree(weight="weight"))[o], 2)))
PK.drop(columns=["members_"]).to_csv(INS / "packages.csv", index=False); pd.DataFrame(mem).to_csv(INS / "package_members.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140)
print(f"{len(PK)} packages from {G.number_of_nodes()} cards")
print(PK.drop(columns=["members_", "new_spoilers"]).to_string(index=False))
print("\nPackages containing new spoilers:"); print(PK[PK.new_spoilers != ""][["package", "colors", "themes", "new_spoilers"]].to_string(index=False))
