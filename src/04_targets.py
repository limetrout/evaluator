"""Turn raw MTGO events into per-card, per-format "how much did this card see play" targets.

Step 1  decks.parquet : one row per (event, deck, card) with copies in main / side
Step 2  monthly play rate per (format, month, card):
           inclusion = (# decks with the card in main + 0.5 * # decks with it only in sideboard) / # decks
Step 3  targets.parquet : one row per (card, format) the card was eligible in.
           target = mean of the card's 3 best months inside its first 18 months of eligibility
           ("peak early play rate"). Measured while the card is new, so it matches the question we
           actually care about (how good will a NEW card be?) and is robust to later bans/rotation.
"""
import json, re, pathlib, collections, unicodedata
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw/mtgo", ROOT / "data/processed"
names = json.loads((PROC / "name_map.json").read_text())
cards = pd.read_parquet(PROC / "cards.parquet").set_index("oracle_id")

def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    n = re.split(r"\s*//\s*|\s*/\s*", n)[0]
    return re.sub(r"[^a-z0-9]", "", n)

rows, unmatched, events = [], collections.Counter(), []
for f in sorted(RAW.glob("*.json")):
    try: d = json.loads(f.read_text())
    except Exception: continue
    fmt = f.stem.split("-")[0]
    date = str(d.get("starttime", ""))[:10] or re.search(r"(\d{4}-\d{2}-\d{2})", f.stem).group(1)
    kind = "minor" if re.search(r"daily|prelim", f.stem) else "major"
    rank = {s["loginid"]: int(s["rank"]) for s in d.get("standings") or [] if str(s.get("rank", "")).isdigit()}
    events.append((f.stem, fmt, date, kind, len(d.get("decklists", []))))
    for deck in d.get("decklists", []):
        did = f"{f.stem}#{deck.get('loginid') or deck.get('player')}"
        acc = collections.defaultdict(lambda: [0, 0])
        for part, side in (("main_deck", 0), ("sideboard_deck", 1)):
            for e in deck.get(part, []):
                nm = e.get("card_attributes", {}).get("card_name") or e.get("card_name", "")
                oid = names.get(norm(nm))
                if oid is None: unmatched[nm] += int(e.get("qty", 1)); continue
                acc[oid][side] += int(e.get("qty", 1))
        r = rank.get(deck.get("loginid"))
        for oid, (m, s) in acc.items():
            rows.append((f.stem, did, fmt, date, kind, r, oid, m, s))

decks = pd.DataFrame(rows, columns=["event", "deck", "format", "date", "kind", "rank", "oracle_id", "main", "side"])
decks.to_parquet(PROC / "decks.parquet")
ev = pd.DataFrame(events, columns=["event", "format", "date", "kind", "n_decks"]); ev.to_parquet(PROC / "events.parquet")
print(f"{len(ev)} events, {decks.deck.nunique()} decks, {decks.oracle_id.nunique()} distinct cards")
print("unmatched card names (top):", unmatched.most_common(15))
basics = set(cards.index[cards.type_line.str.startswith("Basic")])

# ---- monthly inclusion rates ----------------------------------------------------------------
decks["month"] = decks.date.str[:7]
ndecks = decks.groupby(["format", "month"]).deck.nunique().rename("n")
g = decks.assign(inc=np.where(decks.main > 0, 1.0, 0.5), copies=decks.main + decks.side).groupby(["format", "month", "oracle_id"]).agg(inc=("inc", "sum"), copies=("copies", "sum"))
g = g.join(ndecks); g["rate"] = g.inc / g.n; g["avg_copies"] = g.copies / g.inc.clip(lower=1)
g = g.reset_index(); g.to_parquet(PROC / "monthly_rates.parquet")
months_by_fmt = {f: sorted(m for (ff, m), n in ndecks.items() if ff == f and n >= 30) for f in ndecks.index.levels[0]}

# ---- eligibility + target per (card, format) --------------------------------------------------
FORMAT_START = {"standard": "2016-01", "pioneer": "2019-11", "modern": "2016-01", "legacy": "2016-01", "pauper": "2016-01", "vintage": "2016-01"}
STD_TYPES = {"expansion", "core"}
rate_lookup = g.set_index(["format", "oracle_id", "month"]).rate
out = []
for oid, c in cards.iterrows():
    if not c.first_release or oid in basics: continue
    rel = c.first_release[:7]
    for fmt, months in months_by_fmt.items():
        legal = c.get(f"legal_{fmt}")
        if legal in (None, "not_legal") and not (fmt == "standard" and c.first_set_type in STD_TYPES): continue
        if fmt == "standard" and (c.first_set_type not in STD_TYPES or rel < FORMAT_START[fmt]): continue   # debut in Standard only
        if fmt == "pauper" and legal == "not_legal": continue
        start = max(rel, FORMAT_START[fmt])
        win = [m for m in months if m >= start][:18]
        if len(win) < 6: continue                         # not enough history yet -> this card is a prediction target, not a training row
        if fmt == "standard":                              # Standard: must be within ~18 calendar months of debut
            y, mo = map(int, rel.split("-")); end = f"{y + (mo + 17) // 12}-{(mo + 17) % 12 + 1:02d}"
            win = [m for m in win if m <= end]
            if len(win) < 4: continue
        rates = sorted((rate_lookup.get((fmt, oid, m), 0.0) for m in win), reverse=True)
        out.append(dict(oracle_id=oid, format=fmt, target=float(np.mean(rates[:3])), ever_played=float(rates[0] > 0),
                        window_start=win[0], n_months=len(win), legacy_card=int(rel < FORMAT_START[fmt])))
t = pd.DataFrame(out); t.to_parquet(PROC / "targets.parquet")
print(t.groupby("format").agg(cards=("oracle_id", "size"), played=("ever_played", "mean"), mean_target=("target", "mean")))
