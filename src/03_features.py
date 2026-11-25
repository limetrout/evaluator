"""Turn every card into a numeric feature vector.

Groups of features (all derived ONLY from what is printed on the card + when it was released):
  cost      mana value, coloured pips per colour, generic, X, hybrid/phyrexian, colour count, alt-cost mechanics
  body      power, toughness, stats vs. cost, loyalty, defense
  types     card types, supertypes, important subtypes, double-faced / adventure / split / modal layouts
  keywords  one-hot for the most common keyword abilities
  text      ~150 regex "ability detectors" (card draw, removal, counterspells, ETB triggers, tokens, ramp,
            drawbacks, conditionality, land quality ...) with counts and magnitudes
  meta      rarity, text length, number of abilities, release year
Deliberately EXCLUDED (would leak "this card turned out good"): reprint count, EDHREC rank, price, bans.
Output: data/processed/features.parquet
"""
import re, pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
cards = pd.read_parquet(PROC / "cards.parquet")
cards = cards[cards.first_release.notna() & ~cards.type_line.str.contains("Token|Card // Card|Conspiracy|Phenomenon|Plane |Scheme|Vanguard|Dungeon|Attraction|Contraption|Stickers", regex=True)].reset_index(drop=True)

NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "x": 3}
SELF = r"this (?:creature|artifact|enchantment|land|spell|permanent|card|planeswalker|vehicle|equipment|aura|saga|battle|class|token|siege|room|case|spacecraft|planet)"

def clean_text(row):
    t = row.oracle_text or ""
    t = re.sub(r"\([^)]*\)", "", t)                       # strip reminder text
    for n in {row["name"], *row["name"].split(" // ")}:
        t = t.replace(n, "~")
        if "," in n: t = t.replace(n.split(",")[0], "~")     # "Sheoldred" for "Sheoldred, the Apocalypse"
    t = re.sub(SELF, "~", t, flags=re.I)
    return t.lower()

def num(s):
    try: return float(s)
    except (TypeError, ValueError): return np.nan

def numword(w): return NUM.get(w, float(w) if str(w).isdigit() else 1)

# ---- regex ability detectors: name -> pattern (count of matches) -------------------------------
PAT = {
 # card advantage / selection
 "draw": r"\bdraws? (?:a|one|two|three|four|x|\w+) cards?", "draw_multi": r"\bdraws? (?:two|three|four|five|x|that many) cards",
 "cantrip_like": r"^draw a card\.?$", "loot": r"draw (?:a|two) cards?, then discard", "impulse": r"exile the top .{0,40}(?:you may (?:play|cast))",
 "look_top": r"look at the top", "scry": r"\bscry \d", "surveil": r"\bsurveil \d", "mill_self": r"mill (?:a|one|two|three|four|\d) cards?",
 "tutor": r"search your library for (?:a|an|up to \w+) (?!basic land|land)", "tutor_land": r"search your library for (?:a|up to \w+) (?:basic )?lands?",
 "regrowth": r"return (?:target|up to \w+ target|a) [^.]{0,60}card[^.]{0,30} from your graveyard to your hand",
 "reanimate": r"return [^.]{0,60}from (?:your|a) graveyard to the battlefield", "cast_from_gy": r"cast [^.]{0,40}from your graveyard|flashback|escape|unearth",
 "card_from_exile": r"you may (?:play|cast) [^.]{0,40}(?:exiled|from exile)",
 # interaction
 "counter": r"counter target (?:spell|noncreature spell|creature spell|activated|instant|sorcery|ability)", "counter_hard": r"counter target spell\.",
 "counter_soft": r"counter target spell unless", "destroy_creature": r"destroy target (?:\w+ )?creature", "destroy_perm": r"destroy target (?:nonland )?permanent",
 "destroy_art_ench": r"destroy target (?:artifact|enchantment)", "exile_target": r"exile target (?:\w+ )*(?:creature|permanent|artifact|enchantment|planeswalker)",
 "exile_gy": r"exile (?:target (?:player's )?graveyard|all (?:cards from )?graveyards|target card from a graveyard|each opponent's graveyard)",
 "burn_any": r"deals? (?:\d+|x) damage to any target", "burn_creature": r"deals? (?:\d+|x) damage to (?:target|each) (?:\w+ )?creature",
 "burn_player": r"deals? (?:\d+|x) damage to (?:each opponent|target (?:player|opponent))", "minus_stats": r"gets? -\d+/-\d+|gets? -x/-x",
 "fight": r"\bfights?\b|deals damage equal to its power to", "edict": r"(?:each opponent|target (?:player|opponent)) sacrifices",
 "bounce": r"return (?:target|up to \w+ target|each) [^.]{0,40}to (?:its|their) owner'?s'? hands?", "tuck": r"on the (?:top|bottom) of (?:its|their) owner'?s'? library",
 "wrath": r"destroy all|exile all (?:creatures|nonland)|deals? \d+ damage to each creature|all creatures get -", "tap_down": r"tap target|doesn't untap during",
 "discard_opp": r"target (?:player|opponent) (?:reveals|discards)|each opponent discards", "hand_info": r"reveals? (?:their|his or her) hand",
 "stax": r"can't cast|can't be cast|spells (?:your opponents cast )?cost \{?\d\}? more|can't activate|can't search",
 "protect": r"hexproof|indestructible|protection from|phase out|can't be the target", "uncounterable": r"can't be countered",
 # board / bodies
 "token": r"create (?:a|an|one|two|three|four|x|\w+) [^.]{0,60}tokens?", "token_multi": r"create (?:two|three|four|five|x|that many) ",
 "treasure": r"treasure", "clue_food_blood": r"\b(?:clue|food|blood|map) tokens?", "counters_p1": r"\+1/\+1 counters?", "proliferate": r"proliferate",
 "anthem": r"(?:creatures|other \w+s?) you control get \+", "pump": r"target creature (?:you control )?gets \+", "evasion_grant": r"gains? (?:flying|trample|menace|unblockable)|can't be blocked",
 "copy": r"\bcopy\b|copies", "extra_turn_combat": r"extra (?:turn|combat)", "steal": r"gain control of",
 # mana / economy
 "add_mana": r"\badd \{", "add_any_color": r"add (?:one mana of any color|\w+ mana in any combination)", "cost_reduce": r"costs? \{?\d\}? less|cost \{\d\} less|costs? x less",
 "ramp_land": r"put (?:a|that|up to \w+) [^.]{0,30}land cards? onto the battlefield|play an additional land", "untap": r"\buntap target",
 # triggers / engines
 "etb": r"when ~ enters|when ~ (?:enters|is turned)", "attack_trig": r"whenever ~ attacks|whenever ~ deals combat damage", "dies_trig": r"when ~ dies|when ~ is put into a graveyard",
 "cast_trig": r"whenever you cast", "upkeep_trig": r"at the beginning of (?:your|each) (?:upkeep|end step|combat|draw step)", "landfall": r"whenever (?:a|another) land (?:you control )?enters",
 "other_dies": r"whenever (?:a|another) (?:\w+ )?creature (?:you control )?dies", "lifegain_trig": r"whenever you gain life", "draw_trig": r"whenever you draw",
 "opp_trig": r"whenever an opponent", "sac_outlet": r"sacrifice (?:a|another|an) (?:creature|artifact|permanent|token|land|nontoken)",
 "for_each": r"for each", "x_scaling": r"equal to (?:the number|its power|your|that|the total)|where x is",
 # life
 "gain_life": r"gain (?:\d+|x|that much) life", "drain": r"(?:each opponent|target opponent|target player) loses \d+ life",
 "pay_life": r"pay \d+ life", "lose_life_self": r"you lose \d+ life|deals \d+ damage to you",
 # drawbacks
 "enters_tapped": r"~ enters tapped|enters the battlefield tapped", "self_sacrifice": r"sacrifice ~", "upkeep_cost": r"at the beginning of your upkeep, (?:sacrifice|you lose|pay)",
 "cant_block": r"~ can't block|defender", "symmetric": r"\beach player\b|\ball players\b", "discard_self": r"(?:as an additional cost[^.]*discard|you discard)",
 "additional_cost": r"as an additional cost to cast", "cant_attack": r"~ can't attack", "skip": r"\bskip\b",
 # flexibility / complexity
 "modal": r"choose one|choose two|choose one or more|choose any number|choose up to", "if_count": r"\bif\b", "as_long": r"as long as|only if|only as a sorcery|activate only",
 "unless_ctrl": r"unless you control", "instant_speed_act": r"any time you could cast an instant|flash",
 "transform": r"transform|daybound|nightfall|disturb", "graveyard_matters": r"graveyard", "tribal": r"(?:elf|goblin|merfolk|zombie|vampire|human|dragon|angel|wizard|soldier|knight|cleric|rogue|warrior|faerie|spirit|sliver|rat|bird|cat|dog|squirrel|rabbit|frog|otter|bat|lizard|mouse|raccoon)s? you control",
 "artifact_matters": r"artifacts? you control|artifact spell", "enchant_matters": r"enchantments? you control|enchantment spell", "spells_matters": r"instant (?:or|and) sorcery|noncreature spell",
 "legend_matters": r"legendary (?:spell|creature|permanent)", "counter_matters": r"counter on|counters on",
}
PAT = {k: re.compile(v, re.M) for k, v in PAT.items()}

TOP_KW = None
def build(cards):
    global TOP_KW
    kw_counts = pd.Series([k for ks in cards.keywords for k in ks]).value_counts()
    TOP_KW = list(kw_counts[kw_counts >= 25].index)
    out = []
    for _, r in cards.iterrows():
        t = clean_text(r); tl = r.type_line; f = {"oracle_id": r.oracle_id}
        cost = " ".join(c for c in r.face_costs if c) if r.layout in ("split", "adventure", "modal_dfc") else (r.mana_cost or r.face_costs[0] if len(r.face_costs) else "")
        main_cost = r.face_costs[0] if len(r.face_costs) and r.face_costs[0] else r.mana_cost or ""
        syms = re.findall(r"\{([^}]+)\}", main_cost)
        for c in "WUBRGC": f[f"pip_{c}"] = sum(s.count(c) for s in syms if "/" not in s and s == c)
        f["pips_total"] = sum(f[f"pip_{c}"] for c in "WUBRG")
        f["pips_max"] = max(f[f"pip_{c}"] for c in "WUBRG")
        f["generic"] = sum(int(s) for s in syms if s.isdigit())
        f["has_x"] = int("X" in syms); f["hybrid"] = sum("/" in s and "P" not in s for s in syms); f["phyrexian"] = sum("P" in s for s in syms)
        f["mv"] = r.cmc; f["n_colors"] = len(r.colors); f["n_identity"] = len(r.color_identity); f["colorless"] = int(len(r.colors) == 0)
        for c in "WUBRG": f[f"col_{c}"] = int(c in r.colors)
        f["n_faces"] = len(r.face_types); f["mv_face2"] = 0
        if len(r.face_costs) > 1 and r.face_costs[1]:
            f["mv_face2"] = sum(int(s) if s.isdigit() else (0 if s == "X" else 1) for s in re.findall(r"\{([^}]+)\}", r.face_costs[1]))
        for lay in ["transform", "modal_dfc", "adventure", "split", "flip", "saga", "class", "case", "prototype", "meld", "leveler", "mutate", "room"]:
            f[f"layout_{lay}"] = int(r.layout == lay)
        # types
        front = r.face_types[0] if len(r.face_types) else tl
        for ty in ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Planeswalker", "Land", "Battle", "Kindred", "Tribal"]:
            f[f"type_{ty.lower()}"] = int(ty in front); f[f"anytype_{ty.lower()}"] = int(ty in tl)
        f["legendary"] = int("Legendary" in front); f["basic"] = int("Basic" in tl); f["snow"] = int("Snow" in tl)
        for st in ["Equipment", "Aura", "Vehicle", "Saga", "Class", "Room", "Case", "Food", "Shrine", "Cartouche", "Gate", "Desert", "Cave", "Town", "Spacecraft", "Planet", "Siege", "Lesson", "Adventure", "Omen"]:
            f[f"sub_{st.lower()}"] = int(st in tl)
        sub = front.split("—")[1].split() if "—" in front else []
        f["n_subtypes"] = len(sub)
        f["mdfc_land_back"] = int(r.layout == "modal_dfc" and len(r.face_types) > 1 and "Land" in r.face_types[1])
        # body
        p, tg = num(r.power), num(r.toughness)
        f["power"] = p; f["toughness"] = tg; f["pt_star"] = int("*" in str(r.power) + str(r.toughness))
        f["stats"] = (p if p == p else 0) + (tg if tg == tg else 0) if f["type_creature"] or f["sub_vehicle"] else np.nan
        f["stats_minus_mv"] = f["stats"] - 2 * r.cmc if f["stats"] == f["stats"] else np.nan
        f["power_minus_mv"] = p - r.cmc if p == p else np.nan
        f["loyalty"] = num(r.loyalty); f["defense"] = num(r.defense)
        f["loyalty_abilities"] = len(re.findall(r"^[+−-]?[0-9x]+:", t, re.M)) if "Planeswalker" in tl else 0
        # keywords
        kws = set(r.keywords)
        for k in TOP_KW: f["kw_" + re.sub(r"\W", "_", k.lower())] = int(k in kws)
        f["n_keywords"] = len(kws)
        evasion = {"Flying", "Menace", "Trample", "Shadow", "Fear", "Intimidate", "Skulk", "Horsemanship"}
        f["evasive"] = int(bool(kws & evasion)) or int("can't be blocked" in t)
        # text detectors
        for k, rx in PAT.items(): f["t_" + k] = len(rx.findall(t))
        m = re.findall(r"draws? (a|one|two|three|four|five|x) cards?", t); f["cards_drawn"] = sum(numword(x) for x in m)
        m = re.findall(r"deals? (\d+|x) damage", t); f["damage_max"] = max([numword(x) if x == "x" else int(x) for x in m], default=0)
        m = re.findall(r"gain (\d+) life", t); f["life_max"] = max([int(x) for x in m], default=0)
        m = re.findall(r"create (a|an|one|two|three|four|five|x) ", t); f["tokens_made"] = sum(numword(x) for x in m)
        m = re.findall(r"(\d+)/(\d+)[^.]{0,30}token", t); f["token_stats"] = sum(int(a) + int(b) for a, b in m)
        # activated abilities (lines containing a cost colon)
        lines = [l for l in t.split("\n") if l.strip() and l.strip() != "//"]
        f["n_lines"] = len(lines); f["n_activated"] = sum(bool(re.match(r"^[^\"]*?:", l)) and not re.match(r"^[+−-]?[0-9x]+:", l) for l in lines)
        f["n_triggered"] = sum(bool(re.match(r"^(when|whenever|at )", l)) for l in lines)
        f["n_static_or_kw"] = f["n_lines"] - f["n_activated"] - f["n_triggered"]
        f["tap_ability"] = len(re.findall(r"\{t\}", t)); f["mana_ability_only"] = int(bool(re.search(r"\{t\}: add", t)))
        f["n_chars"] = len(t); f["n_words"] = len(t.split()); f["n_sentences"] = t.count(".")
        # land-specific quality
        if "Land" in front:
            f["land_colors"] = len(set(r.produced_mana) - {"C"}); f["land_untapped_always"] = int("tapped" not in t)
            f["land_conditional_untap"] = int(bool(re.search(r"tapped unless|unless you control|if you control two or fewer", t)))
            f["land_fetch"] = int(bool(re.search(r"sacrifice ~: search|search your library for an? (?:basic )?\w* ?(?:or \w+ )?card", t)))
            f["land_utility"] = int(bool(re.search(r"\{t\}[^:]*: (?!add)", t)) or f["t_etb"] > 0 or "surveil" in t)
            f["land_pain"] = int(bool(re.search(r"deals 1 damage to you|pay \d life", t))); f["land_basic_types"] = sum(b in tl for b in ["Plains", "Island", "Swamp", "Mountain", "Forest"])
        else:
            for k in ["land_colors", "land_untapped_always", "land_conditional_untap", "land_fetch", "land_utility", "land_pain", "land_basic_types"]: f[k] = 0
        # cheap interaction flags (well known to matter a lot)
        interaction = f["t_counter"] + f["t_destroy_creature"] + f["t_destroy_perm"] + f["t_exile_target"] + f["t_burn_any"] + f["t_burn_creature"] + f["t_minus_stats"] + f["t_edict"] + f["t_fight"] + f["t_bounce"] + f["t_destroy_art_ench"]
        f["interaction"] = interaction; f["cheap_interaction"] = int(interaction > 0 and r.cmc <= 2)
        f["instant_speed"] = int(f["type_instant"] or "Flash" in kws)
        f["cheap_card_adv"] = int((f["t_draw"] or f["t_impulse"] or f["t_regrowth"]) and r.cmc <= 3)
        f["value_creature"] = int(f["type_creature"] and (f["t_etb"] or f["t_dies_trig"] or f["t_attack_trig"]))
        f["rarity"] = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3, "special": 2, "bonus": 2}.get(r.rarity, 1)
        f["release_year"] = int(r.first_release[:4]) + int(r.first_release[5:7]) / 12
        out.append(f)
    return pd.DataFrame(out)

feat = build(cards)
feat["clean_text"] = [clean_text(r) for _, r in cards.iterrows()]
feat["type_line"] = cards.type_line.values; feat["name"] = cards.name.values; feat["first_release"] = cards.first_release.values; feat["first_set"] = cards.first_set.values
feat.to_parquet(PROC / "features.parquet")
print(feat.shape)
