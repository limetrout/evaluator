"""Projected post-rotation archetype standing (HEURISTIC - no historical data exists to train it on).

strength_now   = share of decks x finishing quality (avg finish vs. the 50% midpoint, top-8 rate)
retained       = core survival  +  (1 - survival) x replacement adequacy
  replacement adequacy for a rotating card = best replacement's single-card rating percentile / 100 x 0.6
  (a proven card is worth more than an untested replacement, hence the 0.6 discount)
projected      = strength_now x retained
Output: output/insights/meta_projection.csv
"""
import pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; INS = ROOT / "output/insights"
A = pd.read_csv(INS / "archetypes.csv"); R = pd.read_csv(INS / "archetype_replacements.csv")
C = pd.read_csv(INS / "archetype_cards.csv")
best = R.sort_values("score", ascending=False).groupby(["archetype", "rotating_card"]).head(1)
rows = []
for a in A.itertuples():
    cards = C[C.archetype == a.archetype]; core = cards[cards.play_rate_in_archetype >= 0.5]
    w = core.play_rate_in_archetype * core.avg_copies
    rot = core[~core.legal_after_rotation]
    adequacy = []
    for r in rot.itertuples():
        b = best[(best.archetype == a.archetype) & (best.rotating_card == r.card)]
        adequacy.append(0.6 * b.replacement_rating_pct.iloc[0] / 100 if len(b) else 0.0)
    lost_w = (rot.play_rate_in_archetype * rot.avg_copies).values
    surv = 1 - (lost_w.sum() / w.sum() if w.sum() else 0)
    recovered = float((lost_w * np.array(adequacy)).sum() / w.sum()) if w.sum() and len(lost_w) else 0
    perf = 1 + (50 - a.avg_finish_pct) / 50 * 0.5 + (a.top8_rate_pct - 25) / 100
    strength = a.share_pct * perf
    rows.append(dict(archetype=a.archetype, colors=a.colors, share_now_pct=a.share_pct, avg_finish_pct=a.avg_finish_pct, top8_rate_pct=a.top8_rate_pct,
                     core_survival_pct=round(100 * surv, 0), retained_pct=round(100 * (surv + recovered), 0),
                     strength_now=round(strength, 2), projected=round(strength * (surv + recovered), 2),
                     biggest_losses=", ".join(rot.sort_values("play_rate_in_archetype", ascending=False).card.head(4)),
                     best_replacements=", ".join(best[best.archetype == a.archetype].sort_values("score", ascending=False).replacement.drop_duplicates().head(4))))
P = pd.DataFrame(rows).sort_values("projected", ascending=False); P.insert(0, "projected_rank", range(1, len(P) + 1))
P.to_csv(INS / "meta_projection.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(P[["projected_rank", "archetype", "share_now_pct", "avg_finish_pct", "core_survival_pct", "retained_pct", "projected", "biggest_losses", "best_replacements"]].to_string(index=False))
