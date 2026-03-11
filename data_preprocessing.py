from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["apartment_size_sqm", "number_of_rooms", "location_score"]
TARGET_COLUMN = "price_eur"


def create_synthetic_dataset(n_samples: int = 800, random_state: int = 42) -> pd.DataFrame:
    """Creates a synthetic apartment dataset for model training and evaluation."""
    rng = np.random.default_rng(random_state)

    apartment_size = rng.normal(loc=75, scale=22, size=n_samples).clip(20, 220)
    number_of_rooms = np.round(apartment_size / 28 + rng.normal(0, 0.7, n_samples)).clip(1, 9)
    location_score = rng.uniform(1, 10, n_samples)

    base_price = 28000
    price = (
        base_price
        + apartment_size * 2500
        + number_of_rooms * 12000
        + location_score * 32000
        + (apartment_size * location_score) * 110
        + rng.normal(0, 20000, n_samples)
    )

    df = pd.DataFrame(
        {
            "apartment_size_sqm": apartment_size,
            "number_of_rooms": number_of_rooms,
            "location_score": location_score,
            "price_eur": price,
        }
    )

    return inject_missing_values(df, missing_rate=0.04, random_state=random_state)


def inject_missing_values(df: pd.DataFrame, missing_rate: float = 0.03, random_state: int = 42) -> pd.DataFrame:
    """Injects random missing values into feature columns to demonstrate imputation."""
    rng = np.random.default_rng(random_state)
    df_with_missing = df.copy()

    for col in FEATURE_COLUMNS:
        missing_mask = rng.random(len(df_with_missing)) < missing_rate
        df_with_missing.loc[missing_mask, col] = np.nan

    return df_with_missing


def remove_outliers_iqr(
    df: pd.DataFrame,
    feature_columns: list[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Removes outliers using IQR bounds across selected feature columns."""
    filtered = df.copy()

    for col in feature_columns:
        q1 = filtered[col].quantile(0.25)
        q3 = filtered[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        filtered = filtered[(filtered[col] >= lower_bound) & (filtered[col] <= upper_bound)]

    return filtered


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Splits a dataframe into feature matrix X and target y."""
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return x, y
