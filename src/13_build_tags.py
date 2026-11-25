"""Parse Claude's role tags (data/tags/out_*_p*.txt) into data/processed/card_tags.parquet.
One row per card: P_<code> / R_<code> binary columns (provides / rewards) + 'tribes' text.
NOTE: tags exist only for cards that were played (Std/Pio/Mod) or are in the post-rotation pool, so they are used
ONLY in the synergy model (which only sees played cards), never in the single-card model (missing tags would leak
'this card was never played')."""
import pathlib, re, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; T = ROOT / "data/tags"
CODES = ["TOK", "TRE", "ART", "ENC", "SPL", "NCS", "CRE", "GOW", "CTR", "OCT", "GY", "REC", "BIG", "SAC", "DIE", "ETB", "LIF", "DRW", "DIS",
         "DMG", "MAN", "LND", "XSP", "LEG", "EQA", "VEH", "EVA", "ATK", "TAP", "CPY", "CST", "FLS", "PSN", "NRG", "HND"]
idx = pd.read_csv(T / "index.csv").set_index("idx").oracle_id
rows, bad = {}, 0
for f in sorted(T.glob("out_*_p*.txt")):
    for line in f.read_text().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3 or not parts[0].strip().isdigit(): bad += 1; continue
        i = int(parts[0]); prov = {c.strip().upper() for c in parts[1].split(",") if c.strip()}
        rew = {c.strip().upper() for c in parts[2].split(",") if c.strip()}
        tribes = parts[3].strip().lower() if len(parts) > 3 else ""
        r = {f"P_{c}": int(c in prov) for c in CODES}; r.update({f"R_{c}": int(c in rew) for c in CODES})
        r["tribes"] = ",".join(t.strip() for t in re.split(r"[,\s]+", tribes) if t.strip())
        rows[i] = r
df = pd.DataFrame.from_dict(rows, orient="index"); df.index.name = "idx"
df["oracle_id"] = idx.reindex(df.index).values
missing = sorted(set(idx.index) - set(df.index))
df.reset_index(drop=True).to_parquet(ROOT / "data/processed/card_tags.parquet")
print(f"{len(df)} cards tagged, {len(missing)} missing, {bad} unparseable lines")
print("most common PROVIDES:", df[[f'P_{c}' for c in CODES]].sum().sort_values(ascending=False).head(8).to_dict())
print("most common REWARDS:", df[[f'R_{c}' for c in CODES]].sum().sort_values(ascending=False).head(8).to_dict())
if missing: print("missing idx sample:", missing[:10])
