# MTG Card Scoring Model

Scores Magic cards by **what is printed on them** (cost, stats, types, keywords, rules text), learned from
roughly ten years of real competitive results. The goal is to estimate how much play a **new** card will
see, before anyone has played with it.

## The idea in one paragraph

Each card in Magic's history gets broken into a few hundred measurable traits. Examples: "costs 2", "is an
instant", "destroys a creature", "draws a card when it enters", "makes a token", "has a drawback". The model then
looks at about ten years of MTGO tournament results and learns which combinations of those traits show up in
winning decks, and how often. Because it learns from the *traits* and not from card names, it can score a card
nobody has played yet, such as a fresh Nauctis spoiler.

## Data

| Source | What we take | Notes |
|---|---|---|
| **mtgo.com/decklists** (official Wizards) | Every Challenge, Qualifier, Showcase, PTQ (Standard, Pioneer, Modern, Legacy, Pauper, Vintage) + Daily/Preliminary events (Standard, Pioneer, Modern), 2016 to now | Full 75-card lists, placements and records |
| **Scryfall** bulk data | Every card's Oracle text, cost, types, stats, keywords, first printing | |
| ~~MTGGoldfish~~ | not used | Its robots.txt forbids using its data to train models |
| MTGTop8 | not used yet | Allows only one request every 5 seconds; could add paper events later |

## What "success" means (the target)

For each card in each format, we measure the **peak early play rate**: the share of top decks that ran the
card, averaged over its 3 best months within the first 18 months it was legal. Sideboard-only copies count
half.

* It is measured while the card is **new**, which is exactly the situation we want to predict.
* It isn't hurt when a card later gets banned or rotates out.
* Only cards that **debuted** in a Standard set count toward the Standard target.

## The model

1. **Text reader**: a linear model over word patterns in the rules text (e.g. "draw a card", "target creature
   an opponent controls"). It catches phrasings that the hand-built detectors miss.
2. **Main model**: gradient-boosted decision trees (LightGBM) over about 330 card features plus the text
   reader's score. It learns interactions such as "card draw is worth a lot at 2 mana but little at 6".
3. **All formats at once**: Standard, Pioneer, Modern, Legacy, Pauper and Vintage all feed the training, with
   format as an input. That gives far more examples of what makes a card strong. Standard rows count double.

**Excluded on purpose**, because they leak whether a card turned out good: reprint count, EDHREC rank, price,
and ban status.

## How we know if it works (honest test)

The model trains **only on cards released before 2024**, then predicts 2024–2025 cards it has never seen. We
compare those predictions with real play. Results are in `output/evaluation_temporal.csv`:

* **Spearman**: how well the predicted ranking matches the real ranking. 0 means random, 1 means perfect.
* **hit@50**: of the 50 cards the model liked most, the share that actually reached at least 2% of decks.
* **staples_in_top5pct**: of the cards that became real staples (10%+ of decks), the share the model ranked
  in its top 5%.

It is compared against simple rules like "rarity only" and "rarity + mana value". If it can't beat those, it
isn't learning anything.

## Limits (please read)

* It scores cards **in a vacuum**. Synergy and the specific metagame aren't on the card. A narrow tribal
  payoff can be great in the right deck and terrible elsewhere. Treat a high score as "this card is
  efficiently powerful" and a low score as "this card needs a specific deck to be good". It isn't a final
  verdict.
* The future is only partly predictable from text alone. Expect the model to be good at separating "unplayable"
  from "maybe" from "probably good", and much rougher at the very top of the ranking.
* MTGO data only. Paper events aren't included yet.

## Running it

```bash
pip install scikit-learn pandas numpy lightgbm requests pyarrow scipy
python src/01_scrape_mtgo.py 2      # download decklists (slow: hours; resumable)
python src/02_cards.py              # card table from Scryfall (download bulk files to data/raw first)
python src/03_features.py           # card -> features
python src/04_targets.py            # decklists -> play-rate targets
python src/05_train.py              # train on pre-2024, test on 2024+, print report
python src/06_score_pool.py         # score the post-rotation Standard pool (+ Nauctis spoilers)
```

## Outputs

* `output/post_rotation_scores.csv`: every card in the Feb-2027 Standard pool with its model score,
  percentile, current real Standard play, and the top reasons behind the score.
* `output/holdout_standard_predictions.csv`: how the model did on cards it never saw.
* `output/feature_importance.csv`: which card traits matter most.
