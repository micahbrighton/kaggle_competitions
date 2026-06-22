# Spaceship Titanic

Kaggle "Getting Started" competition: predict whether each passenger was `Transported` to an
alternate dimension when the Spaceship Titanic collided with a spacetime anomaly. Binary
classification, evaluated on accuracy. Target is balanced (50.4% / 49.6%), so accuracy is a
sane metric and a naive majority-class baseline is ~0.5037.

Full step-by-step experiment log (every test, including the ones that didn't pan out) is in
[EXPERIMENTS.md](EXPERIMENTS.md). This file is the condensed version: what we ended up doing,
why, and what we tried that didn't work.

## Repo layout

```
data/                  train.csv, test.csv, sample_submission.csv (gitignored — see below)
notebooks/
  01_eda.ipynb         exploratory analysis: missingness, target relationships, collinearity
  02_baseline_modeling.ipynb   feature ablation, grid search, ensembling, threshold tuning
  03_generate_submission.ipynb   self-contained final fit + Kaggle submission file generation
src/
  features.py          shared feature engineering + imputation, used by train and test alike
submissions/           generated submission CSVs (gitignored is not needed, small files)
EXPERIMENTS.md         full chronological experiment log with every result, incl. post-mortem
requirements.txt       pinned dependencies for the .venv
```

Data isn't committed (decided to keep the repo lean and re-download on demand). To get it:
```
kaggle competitions download -c spaceship-titanic -p data/
cd data && unzip spaceship-titanic.zip && rm spaceship-titanic.zip
```

## Final methodology

**Imputation** (in `src/features.py`, derived from EDA in `01_eda.ipynb`):
- `HomePlanet`, `Cabin` deck/side: backfilled from travel-group mates first (HomePlanet is
  ~100% consistent within a group, Cabin ~70%), then global mode for any still missing.
- `CryoSleep`: any passenger with spend > 0 cannot be in cryo (hard rule, zero exceptions in
  training data) — inferred as `False` when spend data allows it, mode otherwise.
- Spend columns: imputed as 0 if the passenger is in cryo (structural, not "unknown"),
  median-of-awake-passengers otherwise.
- `Destination`: global mode only — group consistency was too weak (49%) to backfill safely.
- `Age`, `VIP`: global median / mode (no strong grouped pattern found for either).

**Feature set** (settled via systematic ablation, see "What we tried" below):
`CryoSleep`, `HomePlanet`, `Destination`, `Side` (cabin), `GroupSizeBin` (solo/small/large),
`VIP`, `IsSolo`, `Age`, `CabinNum` (raw, continuous), and the 5 raw spend columns
(`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`).

Explicitly **excluded**: `TotalSpend` (redundant given the 5 raw columns — adding it on top
helped nothing, but replacing the raw columns with it cost ~7pp), `Cabin Deck` (kept
`HomePlanet` instead — the two are nearly redundant, Cramér's V = 0.746, and perform
identically alone), `Cabin_region` banding (helps linear/distance models, not the tree-based
models we ended up using), `Name`/surname (never explored — noted as a gap, see below).

**Model**: weighted soft-voting ensemble of tuned CatBoost (`depth=4, iterations=400,
learning_rate=0.05`) and LightGBM (`num_leaves=10, n_estimators=200, learning_rate=0.05`),
weighted 2:1 in CatBoost's favor. Classification threshold: default 0.5 (tuning it didn't
help — see below).

**Validation accuracy: 0.8087** (5-fold stratified CV, `random_state=42` throughout).
Progression: naive baseline 0.5037 → CryoSleep-rule baseline 0.7131 → feature-engineered
LogisticRegression 0.7937 → tuned XGBoost 0.8058 → tuned CatBoost 0.8072 → final ensemble
0.8087.

**Public leaderboard score: 0.80406** — a 0.0046 gap below CV, which is within one standard
deviation of our own CV fold-to-fold variance (0.0049), so not a red flag on its own. The
likely real (if small) contributor: we reused the same 5-fold CV split for every one of the
~60+ decisions logged in `EXPERIMENTS.md` (every feature ablation, grid search, ensemble
weight, threshold check). Each comparison was individually sound, but collectively,
repeatedly picking whatever scored best *on the same folds* mildly overfits the pipeline to
those folds rather than just to the underlying problem. A genuinely untouched final holdout,
evaluated only once at the very end, would have given a less optimistic estimate. Full
post-mortem in `EXPERIMENTS.md`.

## What we tried that didn't work (and why that's useful)

- **Dropping `CryoSleep` because it's collinear with spend.** A single train/val split
  suggested dropping it helped (+0.5pp); 5-fold CV reversed this — keeping it won on every
  single fold. Lesson: don't trust a single split when the margin is a few tenths of a
  percent; collinearity hurts coefficient interpretability far more than it hurts a
  regularized model's accuracy.
- **Collapsing the 5 spend columns into one `TotalSpend`.** Cost ~7pp for both linear and
  tree models — *where* money was spent (which amenity) carries information that the total
  alone erases. Adding `TotalSpend` on top of the raw columns, by contrast, was harmless but
  also useless (exact linear redundancy, no new information for the model).
- **`Cabin_region` banding** (an idea from a public notebook that uses Logistic
  Regression/KNN/SVM). Slightly *hurt* our tree-based models — XGBoost can already split a
  continuous feature at any threshold it wants, so pre-binning just adds noise and
  dimensionality without adding new model capability. The technique is real, just
  conditional on the model family it was paired with.
- **Unweighted ensembling of all 5 tuned models.** Each weaker model added (RandomForest,
  then LogisticRegression) dragged the blended accuracy down monotonically versus the best
  single model. Fix was weighting toward the stronger models and keeping the ensemble small
  (just CatBoost + LightGBM, 2:1) — adding XGBoost back in at *any* weight, even 1:5:4, made
  things worse again, meaning it isn't diverse enough from the other two to earn a place.
- **Threshold tuning.** Directly maximizing OOF accuracy confirmed 0.5 was already optimal
  for our ensemble. A public solution's "match the training base rate" heuristic
  (threshold≈0.53) actively hurt us (0.8066 vs 0.8087) — that trick corrects for a model
  whose probabilities are biased toward over/under-predicting the positive class; ours
  isn't, so it just overrode an already-correct threshold.

## Gaps / ideas not yet explored

Identified by comparing against public solutions (see `EXPERIMENTS.md` "Comparison to public
solutions" section) but not tested:
- Surname-derived family size (we dropped `Name` entirely and never used it as an imputation
  key or feature — one notebook chains surname → HomePlanet for the last few missing values).
- Chained/cascading imputation beyond one level of group-backfill (e.g., deck → HomePlanet,
  surname → HomePlanet) to close out the last few missing values without falling back to
  global mode.
- Log-transforming the skewed spend columns (might matter more for linear/distance models
  than the tree-based ones we ended up using).
