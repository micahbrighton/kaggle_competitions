# Spaceship Titanic — Experiment Log

Validation split: stratified 80/20, `random_state=42`, fixed across all experiments below for fair comparison.
Model for all ablation rows: untuned `LogisticRegression(max_iter=1000, random_state=42)` — no hyperparameter tuning yet, by design, to isolate feature effects.

| # | Features | Model | Val Accuracy | Key observation | Next action |
|---|---|---|---|---|---|
| 0a | (none — predict majority class) | Naive baseline | 0.5037 | Target is balanced (50.4/49.6) | — |
| 0b | CryoSleep (rule: per-class majority) | Rule baseline | 0.7131 | CryoSleep alone is a strong signal | — |
| 1 | CryoSleep only | LogisticRegression | 0.7131 | Matches rule baseline exactly — sanity check passed | — |
| 2 | HomePlanet only | LogisticRegression | 0.5888 | Weak alone | — |
| 3 | Cabin Deck only | LogisticRegression | 0.5796 | Weak alone | — |
| 4 | GroupSize (IsSolo) only | LogisticRegression | 0.5469 | Weakest alone | — |
| 5 | CryoSleep + HomePlanet | LogisticRegression | 0.7131 | No lift over CryoSleep alone | — |
| 6 | CryoSleep + Cabin Deck | LogisticRegression | 0.7056 | Slightly worse than CryoSleep alone | — |
| 7 | CryoSleep + HomePlanet + Cabin Deck | LogisticRegression | 0.7044 | Still no lift; HomePlanet/Deck redundant with each other | — |
| 8 | CryoSleep + HomePlanet + Deck + GroupSize | LogisticRegression | 0.7096 | Categorical combos plateau ~0.70-0.71 | — |
| 9 | All engineered features | LogisticRegression | 0.7918 | Big jump — driven by numeric/spend features, not categoricals | Isolate numerics |
| 10 | Numerics only (Age, TotalSpend, spend cols, CabinNum, GroupSize) | LogisticRegression | 0.7792 | Numerics alone nearly match full feature set | — |
| 11 | CryoSleep + Age + TotalSpend (no raw spend cols) | LogisticRegression | 0.7131 | TotalSpend+Age without raw spend cols barely beats CryoSleep alone | — |
| 12 | CryoSleep + Age + raw spend cols (no TotalSpend) | LogisticRegression | 0.7803 | Raw spend columns carry more signal than the aggregated TotalSpend | — |
| 13 | CryoSleep + all numerics | LogisticRegression | 0.7786 | Adding CryoSleep to full numerics doesn't help | — |
| 14 | All categoricals + CryoSleep (no numerics) | LogisticRegression | 0.7113 | Categoricals plateau regardless of combo | — |
| 15 | **All categoricals + numerics, no CryoSleep** | LogisticRegression | **0.7970** | **Best so far** — dropping CryoSleep beat keeping it | Confirmed via collinearity check |

## Collinearity findings (informing feature set for grid search)

- `CryoSleep` ⇒ `TotalSpend == 0` is a hard one-directional rule (0 exceptions among 3037 cryo passengers). Spend columns already encode nearly all of CryoSleep's signal — keeping both adds redundancy that destabilizes the linear model (experiment 15 vs 9).
- `HomePlanet` × `Deck`: Cramér's V = 0.746 — strongly redundant, explains why combining them added little (experiments 5-8).
- `Deck` × `Side`: Cramér's V = 0.040 — independent, safe to keep both.
- `HomePlanet` × `Destination`: V = 0.258; `HomePlanet` × `GroupSizeBin`: V = 0.214 — mild, not concerning.
- Among spend columns, `FoodCourt` (r=0.74) and `Spa`/`VRDeck` (r=0.59) dominate `TotalSpend`; `RoomService`/`ShoppingMall` correlate weakly (~0.22) with the total and likely carry independent signal — keep raw columns, not just the aggregate.

## CryoSleep drop/keep — re-tested with 5-fold CV (single-split result was misleading)

The single-split comparison (experiment 15) suggested dropping `CryoSleep` helped (+0.5pp). That held only because of which rows landed in that one validation split. Re-tested with 5-fold stratified CV:

| Variant | CV mean | CV std | Folds all agree? |
|---|---|---|---|
| With CryoSleep | **0.7908** | 0.0072 | Yes — wins on all 5/5 folds |
| Without CryoSleep | 0.7850 | 0.0067 | — |

**Decision: keep `CryoSleep`.** It wins on every fold (not just on average), which is the real signal that a single train/val split can't distinguish from noise when the gap is a few tenths of a percent. Lesson: don't drop a theoretically-grounded feature based on a single small split when the margin is this thin — use CV.

## HomePlanet vs Deck head-to-head (5-fold CV)

| Variant | CV mean | CV std |
|---|---|---|
| HomePlanet only | 0.7849 | 0.0073 |
| Deck only | 0.7843 | 0.0068 |
| Both together | 0.7850 | 0.0067 |

Statistically indistinguishable, consistent with Cramér's V=0.746 (they carry almost the same information). **Decision: keep `HomePlanet` (3 categories), drop `Deck` (8 categories)** — same accuracy, less overfitting risk, simpler model.

## Finalized feature set for grid search

`CryoSleep`, `HomePlanet`, `Destination`, `Side`, `GroupSizeBin`, `VIP`, `IsSolo`, `Age`, raw spend columns (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` — not the `TotalSpend` aggregate), `CabinNum`.

## TotalSpend: keep or drop the 5 raw spend columns?

Two separate questions, tested with 5-fold CV on the finalized feature set:

| Test | LogisticRegression | RandomForest |
|---|---|---|
| 5 raw spend cols + TotalSpend (adding the sum on top) | 0.7936 | 0.7942 |
| 5 raw spend cols only (no TotalSpend) | 0.7937 | 0.7935 |
| TotalSpend only (dropping the 5 raw cols) | 0.7259 | 0.7195 |

**Adding `TotalSpend` on top of the raw columns is free but useless** (no meaningful change either direction). **Replacing the raw columns with just `TotalSpend` costs ~7pp** for both model families — collapsing to a sum erases which amenity the money was spent on, and that distinction carries real signal (consistent with `RoomService`/`ShoppingMall` having weak correlation with `TotalSpend`, r≈0.22-0.23, found in the collinearity check). **Decision: keep the 5 raw spend columns, drop `TotalSpend`.**

## Grid search: LogisticRegression vs RandomForest vs XGBoost

Final feature set: `CryoSleep`, `HomePlanet`, `Destination`, `Side`, `GroupSizeBin`, `VIP`, `IsSolo`, `Age`, `CabinNum`, 5 raw spend columns. Same 5-fold stratified CV throughout.

| Model | Best CV Accuracy | Best Params |
|---|---|---|
| **XGBoost** | **0.8058** | `learning_rate=0.2, max_depth=3, n_estimators=100` |
| RandomForest | 0.8032 | `max_depth=10, min_samples_leaf=1, n_estimators=400` |
| LogisticRegression | 0.7937 | `C=1` (tuning barely moved it from the untuned baseline) |

Both RandomForest and XGBoost initially landed on edge-of-grid hyperparameters (RF: `n_estimators=400` at the top; XGB: `n_estimators=100` at the bottom, `learning_rate=0.2` at the top). Re-ran with expanded ranges (RF `n_estimators` up to 800; XGB `n_estimators` down to 50, `learning_rate` up to 0.4, `max_depth` down to 2) — **both searches converged back to the exact same hyperparameters**, confirming 0.8058 is a genuine optimum for this feature set, not an artifact of a too-narrow grid.

**Current best model: XGBoost, CV accuracy 0.8058.** Progress so far: naive baseline 0.5037 → CryoSleep-rule baseline 0.7131 → best feature-engineered logistic regression 0.7937 → tuned XGBoost 0.8058.

## Next step

Train final XGBoost model on full training data with best params, generate predictions on test.csv, create submission file.
