"""Feature engineering for Spaceship Titanic, shared between EDA, baseline, and submission notebooks.

Imputation rules are derived from EDA findings (see notebooks/01_eda.ipynb):
- CryoSleep forces TotalSpend == 0 with no exceptions in train -> spend > 0 implies not in cryo.
- HomePlanet is 100% consistent within a travel group -> safe to backfill from groupmates.
- Cabin (deck/side) is ~70% consistent within a group -> backfill from groupmates, weaker signal.
- Destination is only ~49% consistent within a group -> group backfill is not reliable, use global mode.
"""

import pandas as pd

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _split_passenger_id(df: pd.DataFrame) -> pd.DataFrame:
    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("size")
    return df


def _split_cabin(df: pd.DataFrame) -> pd.DataFrame:
    cabin_parts = df["Cabin"].str.split("/", expand=True)
    cabin_parts.columns = ["Deck", "CabinNum", "Side"]
    df["Deck"] = cabin_parts["Deck"]
    df["CabinNum"] = pd.to_numeric(cabin_parts["CabinNum"])
    df["Side"] = cabin_parts["Side"]
    return df


def _group_backfill(df: pd.DataFrame, col: str) -> pd.Series:
    """Fill missing values using the first known value within the same travel group."""
    group_mode = df.groupby("Group")[col].transform(
        lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA
    )
    return df[col].fillna(group_mode)


def engineer_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply feature engineering + imputation jointly so train/test get consistent treatment.

    Group-based backfill uses only feature columns (PassengerId, Cabin, HomePlanet), never the
    target, so combining train+test for this step is not a leakage risk.
    """
    train = train.copy()
    test = test.copy()
    train["__is_train__"] = True
    test["__is_train__"] = False
    target = train["Transported"]
    combined = pd.concat([train.drop(columns=["Transported"]), test], axis=0, ignore_index=True)

    combined = _split_passenger_id(combined)
    combined = _split_cabin(combined)

    # CryoSleep: spend > 0 is incompatible with cryo (hard rule observed in EDA).
    raw_total_spend = combined[SPEND_COLS].sum(axis=1, skipna=True)
    inferred_not_cryo = raw_total_spend > 0
    combined["CryoSleep"] = combined["CryoSleep"].astype("boolean")
    combined.loc[combined["CryoSleep"].isna() & inferred_not_cryo, "CryoSleep"] = False
    combined["CryoSleep"] = combined["CryoSleep"].fillna(combined["CryoSleep"].mode().iat[0])

    # HomePlanet: near-deterministic within a group, then global mode.
    combined["HomePlanet"] = _group_backfill(combined, "HomePlanet")
    combined["HomePlanet"] = combined["HomePlanet"].fillna(combined["HomePlanet"].mode().iat[0])

    # Cabin deck/side: backfill from group (weaker, ~70% reliable), then global mode.
    combined["Deck"] = _group_backfill(combined, "Deck")
    combined["Side"] = _group_backfill(combined, "Side")
    combined["Deck"] = combined["Deck"].fillna(combined["Deck"].mode().iat[0])
    combined["Side"] = combined["Side"].fillna(combined["Side"].mode().iat[0])
    combined["CabinNum"] = combined["CabinNum"].fillna(combined["CabinNum"].median())

    # Destination: group consistency was weak in EDA (49%) -> global mode only.
    combined["Destination"] = combined["Destination"].fillna(combined["Destination"].mode().iat[0])

    # VIP: rare class (~3%), mode imputation is low-risk.
    combined["VIP"] = combined["VIP"].astype("boolean")
    combined["VIP"] = combined["VIP"].fillna(False)

    # Spend columns: 0 if in cryo (structural), else median of awake passengers.
    for col in SPEND_COLS:
        awake_median = combined.loc[~combined["CryoSleep"].astype(bool), col].median()
        is_cryo_missing = combined["CryoSleep"].astype(bool) & combined[col].isna()
        is_awake_missing = (~combined["CryoSleep"].astype(bool)) & combined[col].isna()
        combined.loc[is_cryo_missing, col] = 0.0
        combined.loc[is_awake_missing, col] = awake_median
    combined["TotalSpend"] = combined[SPEND_COLS].sum(axis=1)

    # Age: simple global median (no strong grouped pattern found in EDA).
    combined["Age"] = combined["Age"].fillna(combined["Age"].median())

    # Group size, binned per the trend break observed in EDA (noisy past ~6).
    combined["IsSolo"] = combined["GroupSize"] == 1
    combined["GroupSizeBin"] = pd.cut(
        combined["GroupSize"], bins=[0, 1, 4, 8], labels=["solo", "small", "large"]
    )

    combined["CryoSleep"] = combined["CryoSleep"].astype(bool)
    combined["VIP"] = combined["VIP"].astype(bool)

    train_out = combined[combined["__is_train__"]].drop(columns=["__is_train__"]).reset_index(drop=True)
    test_out = combined[~combined["__is_train__"]].drop(columns=["__is_train__"]).reset_index(drop=True)
    train_out["Transported"] = target.reset_index(drop=True)

    return train_out, test_out
