"""Score every card in the post-rotation Standard pool (Feb 2027: Foundations + Aetherdrift ... Star Trek + Nauctis).

For each card: model prediction (as if it were brand new, in Standard), the current observed Standard play
rate (last 3 months of MTGO data, where it exists), and the top reasons behind the prediction.
Re-run after downloading fresh Scryfall data to score newly spoiled Nauctis cards.
"""
import json, gzip, pathlib, numpy as np, pandas as pd, lightgbm as lgb
import importlib.util
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT, RAW = ROOT / "data/processed", ROOT / "output", ROOT / "data/raw"
spec = importlib.util.spec_from_file_location("train", ROOT / "src/05_train.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

POOL_SETS = ["mdd", "fdn", "dft", "tdm", "fin", "eoe", "spm", "tla", "ecl", "tmt", "sos", "msh", "hob", "fra", "trk", "nau"]
in_pool, set_of, other_set = set(), {}, {}
FUT = PROC / "future_pool.csv"     # Scryfall f:future (post-rotation Standard), written by the insights step
for l in gzip.open(RAW / "default_cards.jsonl.gz", "rt"):
    p = json.loads(l)
    if p["set"] in POOL_SETS and p.get("oracle_id") and not p.get("promo"):
        in_pool.add(p["oracle_id"]); set_of.setdefault(p["oracle_id"], p["set"].upper())
    if p.get("oracle_id") and p.get("legalities", {}).get("future") == "legal" and not p.get("promo") and p["set"] != "om1":
        other_set.setdefault(p["oracle_id"], p["set"].upper())
for o, st in other_set.items(): set_of.setdefault(o, st)     # pool-set printing preferred, any other printing as fallback
feat = pd.read_parquet(PROC / "features.parquet")
if FUT.exists(): in_pool = set(pd.read_csv(FUT).oracle_id)
pool = feat[feat.oracle_id.isin(in_pool) & ~feat.type_line.str.startswith("Basic")].copy()
pool["format"] = "standard"; pool["window_start"] = "2027-02"; pool["target"] = 0.0; pool["ever_played"] = 0; pool["n_months"] = 0; pool["legacy_card"] = 0
pool["release_year"] = pool.release_year.clip(lower=2024)   # treat as "current era" cards
pool = T.prepare(pool)
pool = pool.reset_index(drop=True)

# Final model is trained on all history EXCEPT the pool cards themselves, so every pool card gets a genuine
# blind prediction that can be compared with how it is actually doing in Standard right now.
hist = T.df[~T.df.oracle_id.isin(set(pool.oracle_id))].reset_index(drop=True)
pred, text_pred, gbm, _, X = T.fit_predict(hist, pool)
pool["model_score"] = np.expm1(pred).clip(min=0).round(2)    # ~ predicted % of top Standard decks
pool["text_score"] = np.expm1(text_pred).clip(min=0).round(2)
pool["pool_set"] = pool.oracle_id.map(set_of)
pool["percentile"] = (pool.model_score.rank(pct=True) * 100).round(1)

# observed play in current Standard (last 3 months with data)
mr = pd.read_parquet(PROC / "monthly_rates.parquet"); st = mr[mr.format == "standard"]
last = sorted(st.month.unique())[-3:]
obs = st[st.month.isin(last)].groupby("oracle_id").rate.sum() / 3
pool["current_std_play_pct"] = (pool.oracle_id.map(obs).fillna(0) * 100).round(1)

# explanations: per-card feature contributions from the tree model
contrib = gbm.predict(X, pred_contrib=True)[:, :-1]
names = list(X.columns)
HIDE = {"text_score", "release_year", "window_year", "rel_day"} | {n for n in names if n.startswith(("fmt_", "emb_pc", "an_"))}
keep = [j for j, n in enumerate(names) if n not in HIDE]
LABEL = {"rarity": "rarity (0=C 1=U 2=R 3=M)", "mv": "mana value", "generic": "generic mana in cost", "n_static_or_kw": "static/keyword lines",
    "t_etb": "enters-the-battlefield effect", "interaction": "removal/counter effects", "cheap_interaction": "cheap removal/counter (MV<=2)",
    "tap_ability": "{T} abilities", "stats": "power+toughness", "stats_minus_mv": "stats vs. cost", "power_minus_mv": "power vs. cost",
    "n_identity": "colors in identity", "n_colors": "number of colors", "t_counter_matters": "cares about counters", "n_keywords": "keywords",
    "t_if_count": "conditional clauses ('if')", "power": "power", "toughness": "toughness", "n_lines": "rules-text lines", "n_activated": "activated abilities",
    "life_max": "life gained", "pips_total": "colored mana symbols", "t_wrath": "board wipe", "damage_max": "damage dealt", "t_spells_matters": "cares about instants/sorceries",
    "pip_R": "red mana symbols", "pip_U": "blue mana symbols", "pip_B": "black mana symbols", "pip_W": "white mana symbols", "pip_G": "green mana symbols",
    "has_x": "X in cost", "kw_cycling": "cycling", "land_colors": "colors the land makes", "t_draw": "draws cards", "cards_drawn": "cards drawn",
    "t_reanimate": "reanimation", "token_stats": "token size", "tokens_made": "tokens made", "t_additional_cost": "additional cost", "t_protect": "protection",
    "t_attack_trig": "attack trigger", "t_other_dies": "death-trigger payoff", "t_pump": "pump spell", "n_chars": "text length", "sub_lesson": "Lesson",
    "layout_adventure": "adventure", "kw_prowess": "prowess", "kw_evoke": "evoke", "t_symmetric": "affects all players", "t_cantrip_like": "cantrip",
    "type_creature": "is a creature", "type_instant": "is an instant", "type_land": "is a land", "legendary": "legendary", "evasive": "evasion",
    "instant_speed": "instant speed", "t_token": "makes tokens", "t_counter": "counterspell", "t_burn_any": "burn to any target", "t_tutor": "tutor",
    "land_untapped_always": "always enters untapped", "land_conditional_untap": "conditionally untapped land", "t_enters_tapped": "enters tapped"}
def why(i):
    c = contrib[i]; order = [keep[j] for j in np.argsort(-np.abs(c[keep]))[:5]]
    def val(j):
        v = X.iloc[i, j]
        return "" if pd.isna(v) else f" {int(v) if float(v).is_integer() else round(float(v), 1)}"
    return "; ".join(f"{LABEL.get(names[j], names[j])}{val(j)} → {'helps' if c[j] > 0 else 'hurts'}" for j in order)
pool["top_reasons"] = [why(i) for i in range(len(pool))]
# nearest historical Standard analogs (released earlier) with their actual peak play
cards_all = pd.read_parquet(PROC / "cards.parquet").set_index("oracle_id")
st_hist = hist[hist.format == "standard"]
Pe = T.E[st_hist.emb_idx.values]; S = T.E[pool.emb_idx.values] @ Pe.T
S[st_hist.rel_day.values[None, :] >= pool.rel_day.values[:, None]] = -2
top = np.argsort(-S, 1)[:, :3]
pool["similar_past_cards"] = ["; ".join(f"{st_hist.name.values[j]} ({100 * st_hist.target.values[j]:.0f}%)" for j in row) for row in top]
# tier = position in the pool, with what that position historically meant (6-year blind backtest, 2018-2023)
TIERS = [(2, "A (top 2%: ~70% saw play, ~45% staple)"), (5, "B+ (top 5%: ~55% / ~30%)"), (10, "B (top 10%: ~45% / ~25%)"),
         (25, "C (top 25%: ~30% / ~14%)"), (50, "D (top half: ~20% / ~9%)"), (101, "F (bottom half: ~3% / ~1%)")]
from_top = 100 - pool.percentile
pool["tier"] = [next(t for cut, t in TIERS if x <= cut) for x in from_top]
pool["reprint"] = pool.first_release < "2024-11-01"
pool["is_land"] = pool.type_line.str.contains("Land") & ~pool.type_line.str.contains("//")
cols = ["name", "pool_set", "type_line", "mv", "rarity", "tier", "percentile", "model_score", "current_std_play_pct", "reprint", "is_land", "similar_past_cards", "top_reasons"]
res = pool.sort_values("model_score", ascending=False)[cols]
res.to_csv(OUT / "post_rotation_scores.csv", index=False)
nl = res[~res.is_land]
print("TOP NEW NON-LAND CARDS:\n", nl[~nl.reprint].head(40).drop(columns=["type_line", "is_land", "reprint"]).to_string(index=False))
print("\nNauctis spoilers:\n", res[res.pool_set == "NAU"].to_string(index=False))
