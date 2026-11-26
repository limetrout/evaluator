"""Data for the post-rotation card browser: every future-Standard card with its rules text, rating, current play
and its top partners from the v3 ranker. Output: output/browser/cards.json (records, one per line)."""
import json, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"
single = pd.read_csv(OUT / "post_rotation_scores.csv")
cards = pd.read_parquet(PROC / "cards.parquet").drop_duplicates("name").set_index("name")
partners = pd.read_csv(OUT / "insights/card_partners.csv")
sets = {s["code"].upper(): s["name"] for s in json.load(open(ROOT / "data/raw/sets.json"))["data"]}
single = single.drop_duplicates("name").reset_index(drop=True)
idx = {n: i for i, n in enumerate(single.name)}
P = partners[partners.partner.isin(idx)].sort_values("ranker_score", ascending=False).groupby("card").head(12)
recs = []
for i, r in single.iterrows():
    c = cards.loc[r["name"]] if r["name"] in cards.index else None
    pt = f"{c.power}/{c.toughness}" if c is not None and isinstance(c.power, str) else (f"Loyalty {c.loyalty}" if c is not None and isinstance(c.loyalty, str) else "")
    cols = list(c.colors) if c is not None and c.colors is not None else []
    if not cols and c is not None and c.color_identity is not None: cols = list(c.color_identity) if "Land" in r.type_line else []
    plist = [[idx[p.partner], int(p.decks_together_now)] for p in P[P.card == r["name"]].itertuples()]
    recs.append(dict(
        n=r["name"], s=r.pool_set, sn=sets.get(r.pool_set, r.pool_set), c=(c.mana_cost if c is not None else "") or "",
        t=r.type_line, x=(c.oracle_text if c is not None else "") or "", pt=pt, col="".join(cols),
        r=["C", "U", "R", "M"][int(r.rarity)] if r.rarity == r.rarity else "", tier=str(r.tier)[:2].strip(), tl=str(r.tier),
        pc=float(r.percentile), play=float(r.current_std_play_pct), rp=bool(r.reprint), land=bool(r.is_land),
        sim=r.similar_past_cards if isinstance(r.similar_past_cards, str) else "", p=plist, mv=float(r.mv)))
with open(OUT / "browser/cards.json", "w") as f:
    f.write("[\n" + ",\n".join(json.dumps(x, separators=(",", ":"), ensure_ascii=False) for x in recs) + "\n]\n")
print(len(recs), "cards;", sum(1 for x in recs if x["p"]), "with partner lists;", round((OUT / "browser/cards.json").stat().st_size / 1e6, 2), "MB")
print(pd.Series([x["s"] for x in recs]).value_counts().head(16).to_dict())
