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

## Comparison to public solutions

Reviewed three public writeups (Samuel Cortinhas's Kaggle "complete guide", SoumyaCO's GitHub repo, Fernandao Lacerda Dantas's Medium writeup, 0.8066 score) plus general search context — public solutions plateau around 0.80-0.82 (top 6-28% of teams; ignore leaderboard #1s, they're known to be leaked-dataset cheats). Our 0.8058 CV score is already in the competitive range.

Gaps identified vs. Samuel Cortinhas's notebook specifically:
1. Chained imputation (group -> deck -> surname -> destination, reaching 0 missing `HomePlanet` without a global-mode fallback) vs. our single group-backfill + mode fallback.
2. `Cabin_number` binned into ~300-wide regions, vs. our raw continuous `CabinNum`.
3. Screens 7 models including LightGBM/CatBoost/SVM/KNN, soft-votes top 2 across 10-fold CV, and tunes the submission classification threshold to match the known target base rate — three techniques we haven't tried.

### Tested: Cabin_region banding (XGBoost, tuned params, 5-fold CV)

| Variant | Mean | Std |
|---|---|---|
| Raw CabinNum (continuous) | **0.8058** | 0.0046 |
| CabinRegion only (banded into 7 regions of 300) | 0.8052 | 0.0083 |
| Both CabinNum + CabinRegion | 0.8040 | 0.0074 |

**Decision: keep raw CabinNum, skip banding.** Banding helped Samuel because his model lineup includes Logistic Regression/KNN/SVM, which can't discover nonlinear thresholds in a continuous feature on their own — binning hands them that structure. XGBoost is tree-based and can already split on `CabinNum` at any threshold, so pre-binning adds noise (coarser boundaries, more one-hot dimensions) without adding new model capability. Lesson: feature engineering tricks from other writeups are often model-family-specific, not universal — worth testing rather than assuming transfer.

## LightGBM + CatBoost added to model comparison

Grid searched on the same finalized feature set and 5-fold CV. Both initially landed on edge values; expanded grids confirmed/slightly improved:

| Model | Best CV Accuracy | Best Params |
|---|---|---|
| **CatBoost** | **0.8072** | `depth=4, iterations=400, learning_rate=0.05` (confirmed interior optimum after expanding depth range to [2,3,4]) |
| LightGBM | 0.8067 | `num_leaves=10, n_estimators=200, learning_rate=0.05` (still at a lower edge on num_leaves after one expansion; diminishing returns, stopped chasing further) |
| XGBoost | 0.8058 | (unchanged from before) |
| RandomForest | 0.8032 | (unchanged from before) |
| LogisticRegression | 0.7937 | (unchanged from before) |

CatBoost is now the best single model, edging out XGBoost.

## Soft-voting ensemble

Built `VotingClassifier(voting="soft")` combinations using each model's own tuned best params, evaluated with the same 5-fold CV.

**Unweighted ensembles all underperformed CatBoost alone**, and got monotonically worse as weaker models were added:

| Variant | Mean | Std |
|---|---|---|
| CatBoost alone | 0.8072 | — |
| Top 3 boosters (XGB+LGBM+CatBoost), equal weight | 0.8067 | 0.0062 |
| Top 3 + RandomForest, equal weight | 0.8058 | 0.0074 |
| All 5 incl. LogisticRegression, equal weight | 0.8054 | 0.0060 |

Diagnosis: unweighted soft-voting treats every model equally regardless of skill, so blending in weaker models (RandomForest 0.8032, LogisticRegression 0.7937) drags the average toward them instead of adding helpful diversity.

**Fix: weight the vote toward the stronger models, and keep the ensemble small.**

| Variant | Mean | Std |
|---|---|---|
| **CatBoost + LightGBM, weighted 2:1** | **0.8087** | 0.0049 |
| CatBoost + LightGBM, equal weight (1:1) | 0.8079 | 0.0064 |
| CatBoost alone | 0.8072 | — |
| CatBoost + XGB + LightGBM, weighted 3:2:2 | 0.8062 | 0.0058 |

**Current best model: weighted soft-voting ensemble of CatBoost + LightGBM (2:1), CV accuracy 0.8087.** Even equal-weighting just these two closest competitors beat CatBoost alone, confirming real complementary signal between them — it just needed correct weighting to surface. Adding XGBoost back in made things worse again: XGBoost isn't diverse enough from CatBoost/LightGBM to add value, it just reintroduces the equal-weight drag problem. Lesson: a small, well-weighted ensemble beat every larger one tried — "more models" is not automatically better.

Progress so far: naive baseline 0.5037 -> CryoSleep-rule 0.7131 -> feature-engineered LogisticRegression 0.7937 -> tuned XGBoost 0.8058 -> tuned CatBoost 0.8072 -> weighted CatBoost+LightGBM ensemble 0.8087.

## Broader weight/model combination sweep

Tested 14 more combinations (finer CatBoost:LightGBM ratios, CatBoost+XGB pairs, LightGBM+XGB pairs, triples with XGBoost heavily down-weighted) to check whether the 2:1 CatBoost:LightGBM result was the actual optimum or just a lucky first guess.

| Combination | Mean |
|---|---|
| CatBoost+LGBM 3:2 | 0.8087 (tied) |
| CatBoost+LGBM 2:1 | 0.8087 (tied) |
| CatBoost+LGBM 5:2 | 0.8081 |
| CatBoost+LGBM 1:1 | 0.8079 |
| CatBoost+LGBM+XGB 5:4:1 (XGB barely weighted) | 0.8078 |
| CatBoost+XGB or LightGBM+XGB (any ratio) | 0.8061-0.8071 |

**Confirmed: CatBoost+LightGBM (no XGBoost) at ~2:1 is the real optimum**, not a lucky single guess — 3:2 ties it exactly. Every combination that includes XGBoost underperforms the plain CatBoost+LightGBM pair, even when XGBoost is down-weighted to 1 part against 5 and 4 — it isn't complementary enough with the other two boosters to earn a place in this ensemble, despite being individually competent alone (0.8058).

## Final model: weighted CatBoost + LightGBM ensemble (2:1), CV accuracy 0.8087

Progress: naive 0.5037 -> CryoSleep-rule 0.7131 -> LogisticRegression 0.7937 -> XGBoost 0.8058 -> CatBoost 0.8072 -> **weighted CatBoost+LightGBM ensemble 0.8087**.

## Threshold tuning (does not help)

Computed out-of-fold predicted probabilities for the final CatBoost+LightGBM (2:1) ensemble via `cross_val_predict`, then tried two ways to pick a better classification threshold than the default 0.5:

| Method | Threshold | OOF Accuracy |
|---|---|---|
| Default | 0.500 | 0.8087 |
| Directly maximize OOF accuracy (sweep thresholds) | 0.500 | 0.8087 (identical — confirms 0.5 is already optimal) |
| Match training base rate of 50.4% (Samuel Cortinhas's approach) | 0.530 | 0.8066 (worse) |

**Decision: keep the default 0.5 threshold.** Our ensemble's predicted probabilities are already well-calibrated for accuracy, so there's nothing to gain by moving the cutoff. Samuel's base-rate-matching heuristic actively hurt here — it's a useful correction when a model's probabilities are systematically biased toward over/under-predicting the positive class, but ours isn't, so forcing the predicted proportion to match 50.4% just overrode an already-correct threshold. Consistent with the Cabin-banding lesson: techniques from other writeups are conditional on quirks of *their* model, not universally transferable.

## Final submission

Trained the final model (CatBoost+LightGBM, weighted 2:1, default 0.5 threshold) on the full
training set, predicted on `test.csv`, submitted to Kaggle.

| | Accuracy |
|---|---|
| 5-fold CV estimate | 0.8087 |
| **Public leaderboard** | **0.80406** |
| Gap | 0.0046 (0.95 standard deviations of our own CV fold std, 0.0049) |

## Post-mortem: why CV and leaderboard differ

The gap is **within one standard deviation of our own CV fold-to-fold variance** — not a sign
of a broken pipeline, a leak, or a meaningfully overfit model. If we'd gotten a wildly
different CV-vs-leaderboard split (multiple points apart) that would have been a red flag
worth digging into (e.g. a feature leaking target information, or train/test distribution
mismatch). This isn't that.

That said, there's a real, nameable contributor to why CV likely runs slightly optimistic
here, beyond pure noise: **we reused the exact same 5-fold CV split (`random_state=42`) for
every decision in this entire project** — every feature-ablation comparison, every grid
search, every ensemble-weight sweep, the threshold-tuning check, all ~60+ experiments logged
in this file. Each individual comparison was sound on its own, but collectively, repeatedly
picking whatever scored best *on those same folds* means we've implicitly tuned the whole
pipeline to be good at predicting those specific folds, not just at the underlying problem.
This is a subtle form of selection bias often missed by individual contributors: no single
step leaked the target, but the *aggregate of many small "pick the better option" decisions*
on a fixed validation set inflates the validation estimate a little, even when each step
followed best practice (CV, not single-split; checked for edge-of-grid; verified
collinearity findings with CV, etc.).

What would have caught/avoided this: holding out a single untouched test split at the very
start of the project, only ever evaluating candidate pipelines against it (or even better,
never looking at it until the literal end), and using the 5-fold CV purely for the internal
decisions like we did. We didn't do that here, and the cost was small (the leaderboard
result isn't bad — it's about what the public solutions we reviewed scored), but it is the
clean reason CV-vs-leaderboard gaps tend to skew optimistic across many-experiment projects
in general, not just this one.

The public leaderboard score is also computed on roughly half the full test set (Kaggle
splits public/private), so it carries its own sampling noise on top of everything above —
the private leaderboard score (revealed at competition end) may land closer to or further
from 0.8087 than the public score did, by chance alone.

## Next step

None for now — this is a complete first pass. Possible future directions if revisited:
surname-derived family size, chained imputation beyond one level of group-backfill, and a
genuinely held-out final validation split for any future tuning to avoid the optimism bias
described above.
