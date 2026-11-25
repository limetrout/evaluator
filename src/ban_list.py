"""Cards banned (or Vintage-restricted) in constructed formats.
Current bans come from Scryfall legalities; Standard bans that have since rotated out are no longer visible there,
so they are listed by hand below (Standard bans 2016-2025)."""
import pathlib, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
ROTATED_STANDARD_BANS = """Emrakul, the Promised End; Reflector Mage; Smuggler's Copter; Aetherworks Marvel; Felidar Guardian;
Rampaging Ferocidon; Ramunap Excavator; Attune with Aether; Rogue Refiner; Field of the Dead; Oko, Thief of Crowns;
Once Upon a Time; Veil of Summer; Agent of Treachery; Fires of Invention; Growth Spiral; Wilderness Reclamation;
Teferi, Time Raveler; Cauldron Familiar; Uro, Titan of Nature's Wrath; Omnath, Locus of Creation; Lucky Clover;
Escape to the Wilds; Alrund's Epiphany; Divide by Zero; Faceless Haven; Fable of the Mirror-Breaker;
Reckoner Bankbuster; Invoke Despair; The Meathook Massacre; Cori-Steel Cutter; Heartfire Hero; Monstrous Rage;
Hopeless Nightmare; Proft's Eidetic Memory; Up the Beanstalk; Abuelo's Awakening; Leyline Binding; Nexus of Fate;
Collected Company; Expressive Iteration; Kroxa, Titan of Death's Hunger; Walking Ballista"""
def banned_ids():
    c = pd.read_parquet(ROOT / "data/processed/cards.parquet")
    cur = c[(c[["legal_standard", "legal_pioneer", "legal_modern", "legal_legacy", "legal_pauper"]] == "banned").any(axis=1) | (c.legal_vintage == "restricted")]
    names = {n.strip() for n in ROTATED_STANDARD_BANS.replace("\n", " ").split(";")}
    hist = c[c.name.isin(names) | c.name.str.split(" // ").str[0].isin(names)]
    return set(cur.oracle_id) | set(hist.oracle_id)
