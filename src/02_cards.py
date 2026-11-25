"""Build the master card table from Scryfall bulk data.

Output: data/processed/cards.parquet (one row per Oracle card) and data/processed/name_map.json
(every name a card has been printed under -> oracle_id, used to match MTGO decklists).
"""
import json, gzip, re, pathlib, collections, unicodedata
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"; PROC.mkdir(parents=True, exist_ok=True)
SET_TYPES = {"expansion", "core", "draft_innovation", "masters", "commander", "starter", "alchemy", "funny"}
PRIMARY = {"expansion", "core", "draft_innovation"}   # where cards debut into constructed

def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    n = n.split(" // ")[0] if " // " in n else n
    return re.sub(r"[^a-z0-9]", "", n)

oracle = {}
for l in gzip.open(RAW / "oracle_cards.jsonl.gz", "rt"):
    c = json.loads(l)
    if c.get("layout") in {"token", "double_faced_token", "emblem", "art_series", "planar", "scheme", "vanguard"}: continue
    if c.get("set_type") in {"token", "memorabilia"} or "oracle_id" not in c: continue
    oracle[c["oracle_id"]] = c

first, first_set, first_rarity, first_type, sets_by = {}, {}, {}, {}, collections.defaultdict(set)
names = {}
for l in gzip.open(RAW / "default_cards.jsonl.gz", "rt"):
    p = json.loads(l); oid = p.get("oracle_id")
    if oid not in oracle: continue
    for n in {p["name"], p.get("flavor_name"), p.get("printed_name")} | {f.get("name") for f in p.get("card_faces", [])} | {f.get("flavor_name") for f in p.get("card_faces", [])}:
        if n: names.setdefault(norm(n), oid)
    if p["set_type"] in PRIMARY and not p.get("promo") and p["released_at"]:
        sets_by[oid].add(p["set"])
        if oid not in first or p["released_at"] < first[oid]:
            first[oid], first_set[oid], first_rarity[oid], first_type[oid] = p["released_at"], p["set"], p["rarity"], p["set_type"]
# canonical names win over alternate ones
for oid, c in oracle.items(): names[norm(c["name"])] = oid

rows = []
for oid, c in oracle.items():
    faces = c.get("card_faces") or [c]
    f0 = faces[0]
    rows.append(dict(
        oracle_id=oid, name=c["name"], layout=c["layout"], mana_cost=c.get("mana_cost") or f0.get("mana_cost", ""),
        face_costs=[f.get("mana_cost", "") for f in faces], cmc=c.get("cmc", 0), type_line=c.get("type_line", ""),
        face_types=[f.get("type_line", "") for f in faces], oracle_text="\n//\n".join(f.get("oracle_text", "") for f in faces),
        face_texts=[f.get("oracle_text", "") for f in faces], power=f0.get("power"), toughness=f0.get("toughness"),
        loyalty=f0.get("loyalty"), defense=f0.get("defense"), colors=c.get("colors") or f0.get("colors", []),
        color_identity=c.get("color_identity", []), produced_mana=c.get("produced_mana", []), keywords=c.get("keywords", []),
        rarity=first_rarity.get(oid, c.get("rarity")), first_set_type=first_type.get(oid), reserved=c.get("reserved", False), game_changer=c.get("game_changer", False),
        first_release=first.get(oid), first_set=first_set.get(oid), n_primary_sets=len(sets_by[oid]),
        legal_standard=c["legalities"].get("standard"), legal_pioneer=c["legalities"].get("pioneer"),
        legal_modern=c["legalities"].get("modern"), legal_legacy=c["legalities"].get("legacy"), legal_vintage=c["legalities"].get("vintage"), legal_pauper=c["legalities"].get("pauper"), edhrec_rank=c.get("edhrec_rank"),
    ))
df = pd.DataFrame(rows)
df.to_parquet(PROC / "cards.parquet")
(PROC / "name_map.json").write_text(json.dumps(names))
print(len(df), "cards;", df.first_release.notna().sum(), "with a primary-set release;", len(names), "names")
