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

---

# Synergy model (pairs of cards)

**Question:** which cards make each other better? Can we predict that for cards that have never been played together?

## What it measures
**Lift**: how much more often two cards appear together in winning main decks than chance predicts. It only
counts decks whose colors can cast both cards, so "two red cards in red decks" doesn't look like synergy
(`src/09_pair_targets.py`). Each pair is measured over its first four quarters in Standard, Pioneer and Modern.
A lift of 3 means "together three times more often than chance".

Sanity check: the highest-lift Standard pairs since 2016 are Warriors tribal (Chief of the Edge, Mardu
Woe-Reaper), Hardened Scales counters, Fleshbag Marauder + Merciless Executioner, and Goblin Piledriver Goblins.

## How it predicts (`src/10_pair_model.py`)
Every feature comes from the two printed cards, so the model can score pairs nobody has tried:
* **role matches**: card A *provides* what card B *rewards* (tokens, +1/+1 counters, artifacts, enchantments,
  cheap spells, self-mill, lifegain, mana creatures, extra lands, enter-the-battlefield creatures, death triggers,
  legends, equipment/auras, energy, poison...)
* **mentions**: B's text names A's creature type, card type, token type or keyword
* **meaning similarity** from the language model, plus color overlap and mana curve

## Forward-only test (no hindsight)
For each year Y, the model trains only on co-occurrence seen **before** Y, then predicts pairs that only became
possible in Y (at least one card is new). A **strong synergy** means at least 2× lift over 5 or more decks;
23% of test pairs qualify.

| Year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| Ranking agreement: model | **0.30** | **0.18** | **0.27** | **0.27** | **0.31** | **0.31** |
| Ranking agreement: similarity + color baseline | 0.19 | 0.12 | 0.13 | 0.16 | 0.14 | 0.12 |
| Ranking agreement: hand-written role rules | 0.23 | 0.18 | 0.19 | 0.22 | 0.19 | 0.15 |
| Model's top 2% that were strong synergies | **64%** | **52%** | **69%** | **74%** | **56%** | **65%** |

The model beats both baselines every year, and its top picks are strong synergies about 2.8× as often as
chance. Coverage is limited: it finds only about 19% of all strong synergies in its top 10%. Tried and rejected:
"partner profiles" (does the new card resemble an old card's usual partners?), which made the top picks worse.

## Output (`src/11_score_synergies.py`)
* `output/synergy_pairs_post_rotation.csv`: the top predicted pairs among post-rotation Standard cards (out of
  about 4 million scored), with status:
  * **proven**: already played together in recent winning decks
  * **hidden**: both cards are played, but almost never together
  * **untested**: at least one card isn't being played
* `output/synergy_highlights.csv`: curated lists (proven, hidden, untested, build-arounds). Each card appears at
  most 3 times, both cards are tier C or better, and pairs are ranked by synergy plus playability.

**Caveats:**
* Lift measures "belongs in the same deck", not "wins games". Two weak cards can have a huge lift.
* Artifact pairs are over-represented, because historical artifact decks had extreme lifts and the recent
  Star Trek and Marvel sets are artifact-heavy.
* Recent Standard data is still thin (about 600 decks since 2025), so "hidden" vs "untested" will sharpen as more
  data downloads.

## Synergy model v3: per-card ranker (current best)

The question you actually ask is "what goes with card X?", so v3 trains on exactly that. For each new card from past
years, it ranks **every** card it could have been paired with in its format (`src/14_synergy_ranker.py`,
LightGBM learning-to-rank). It uses the v1 features plus:
* **Card map**: from past decks, cards that live in the same decks sit close together (PMI + SVD, 64 numbers
  per card). Brand-new cards are placed on the map from their text by a text-to-map model.
* **Claude role tags** for 6,790 cards (what each card provides and rewards, plus creature types), made with Claude
  Code on the subscription plan with no API key (`output/card_tags.csv`, `tags_raw/`).
* Each candidate's popularity in the previous year (known at the time).

**Test: share of a new card's real strong partners that appear in its top 10.** Strictly forward; each year only
uses earlier data.

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | Avg |
|---|---|---|---|---|---|---|---|
| **v3 ranker** | **17.5%** | **16.9%** | **18.5%** | **21.5%** | **24.1%** | **22.8%** | **20.2%** |
| v1 pair model | 12.9% | 12.1% | 13.4% | 14.6% | 13.8% | 12.2% | 13.2% |
| Card map only | 14.7% | 11.2% | 12.2% | 14.0% | 11.4% | 15.6% | 13.2% |
| Popularity only | 5.5% | 7.2% | 4.7% | 4.0% | 6.6% | 6.3% | 5.7% |

**Extra check on 2025-26 Standard** (never used for training; card map from pre-2025 decks only, so new cards are
placed purely from their text): 117 new cards, v3 ranker **26.6%**, probability model **27.8%**, card map alone
25.3%, popularity alone 3.7%. Example: Badgermole Cub's blind top 10 contains 7 of its real partners (Llanowar
Elves, Sazh's Chocobo, Ouroboroid, Earthbender Ascension, Bristly Bill, Bushwhack, Lumbering Worldwagon).

**Tried and kept or dropped:**
* Claude tags: a small gain (20.2% vs 19.9% without). Kept.
* v1 + card map as a pair regression: better overall ranking, worse per-card. Dropped.
* Probability variant: slightly lower per-card recall (18.8%), but its scores are comparable across cards, so it's
  used for "strongest synergies in the whole pool" lists.
* Two bugs were caught by checks and fixed: a leak in the 2025-26 check (the card map was built from all decks), and
  lands appearing as partners.

Saved models: `output/models/synergy_ranker.txt` and `synergy_prob.txt` (`src/15_final_synergy.py`).
