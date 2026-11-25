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

## How we know if it works (rolling backtest)

For each year from 2018 to 2023 the model is retrained using **only what was known before that year**, then it
predicts that year's new cards blind. That gives 6,119 Standard cards, 1,026 of which saw play and 494 of which
became staples, each predicted without the model ever seeing its results (`src/07_backtest.py`).

| Model's rank within its year | Saw play (2%+ of top decks) | Became a staple (10%+) |
|---|---|---|
| Top 2% | 69% | 46% |
| 2-5% | 54% | 31% |
| 5-10% | 44% | 26% |
| 10-25% | 31% | 14% |
| 25-50% | 20% | 9% |
| Bottom 50% | 3.5% | 1.2% |

(For comparison, about 17% of all new cards see play and 8% become staples.)

How well the ranking matches reality (Spearman correlation, 0 = random, 1 = perfect):

| | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|
| Model | 0.48 | 0.60 | 0.63 | 0.62 | 0.60 | 0.59 |
| Baseline: rarity + mana value | 0.37 | 0.52 | 0.52 | 0.49 | 0.51 | 0.49 |

Among rares and mythics only it scores 0.40, against 0.24 for the baseline. Pioneer: 0.53 vs 0.40. Modern: 0.41 vs 0.34.

**Known blind spots:** expensive bombs (Atraxa, Sunfall and the Virtues were all underrated) and build-around or
synergy cards (Lord Skitter, Subterranean Schooner).

## Reading the ratings (`output/post_rotation_scores.csv`)

* **tier**: the card's position in the pool, labeled with the backtest odds above. Start here.
* **model_score**: the predicted % of top decks. It runs low, so compare it between cards and don't read it as
  a literal percentage.
* **current_std_play_pct**: actual Standard play right now. A high tier with no play means "sleeper or hidden
  flaw: test it"; a low tier with high play means "synergy card, good in one deck".
* **similar_past_cards**: the 3 closest earlier cards and how much they were played. Often the most useful column.
* **top_reasons**: plain-English reasons, with the card's value and whether each one helped or hurt.
* **reprint / is_land**: filters.

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
