"""
Data Preprocessing Pipeline – Zurich Apartment Price Prediction
===============================================================
Provides a modular, step-by-step cleaning and feature-engineering pipeline
that transforms the enriched apartment listing data from
data/apartments_data_enriched_with_new_features.csv into a model-ready
dataset saved to data/apartments_enriched_clean.csv.

Engineered features
-------------------
* distance_to_zurich_center  – Haversine distance (km) to the city centre
* furnished                  – binary flag extracted from title/description
* parking                    – binary flag extracted from title/description
* rooms_per_sqm              – dimensionality ratio (number_of_rooms / size)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DATA_PATH = Path("data/apartments_data_enriched_with_new_features.csv")
CLEAN_DATA_PATH = Path("data/apartments_enriched_clean.csv")

# ── Geographic constants ──────────────────────────────────────────────────────
ZURICH_CENTER_LAT = 47.3769
ZURICH_CENTER_LON = 8.5417

# ── Keyword lists for feature extraction ─────────────────────────────────────
FURNISHED_KEYWORDS = ["möbliert", "moebliert", "furnished", "mobiliert"]
PARKING_KEYWORDS   = [
    "garage", "parking", "parkplatz", "carport",
    "tiefgarage", "einstellplatz",
]
WG_KEYWORDS = [
    "wg", "wg-zimmer", "roommate", "shared room",
    "mitbewohner", "zimmer in wg",
]

# ── Feature / target column names ────────────────────────────────────────────
TARGET_COLUMN     = "price_chf"
BASELINE_FEATURES = ["apartment_size_sqm", "number_of_rooms", "rooms_per_sqm"]
EXTENDED_FEATURES = BASELINE_FEATURES + [
    "distance_to_zurich_center",
    "furnished",
    "parking",
]


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def _standardize_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes external dataset schemas to the project's expected column names.

    This keeps the downstream pipeline unchanged and supports both the original
    assignment schema and enriched variants.
    """
    df = df.copy()

    # Some exports contain an unnamed index column that should not be model input.
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)

    rename_map = {
        "rooms": "number_of_rooms",
        "area": "apartment_size_sqm",
        "lat": "latitude",
        "lon": "longitude",
        "description_raw": "description",
        "postalcode": "zip_code",
        "town": "city",
    }
    df.rename(columns=rename_map, inplace=True)

    if "price" in df.columns and TARGET_COLUMN not in df.columns:
        df.rename(columns={"price": TARGET_COLUMN}, inplace=True)

    if "title" not in df.columns:
        if "description" in df.columns:
            df["title"] = df["description"].fillna("").astype(str).str.slice(0, 120)
        elif "address" in df.columns:
            df["title"] = df["address"].fillna("").astype(str)
        else:
            df["title"] = ""

    if "description" not in df.columns:
        df["description"] = df["title"].fillna("").astype(str)

    required = [TARGET_COLUMN, "number_of_rooms", "apartment_size_sqm", "latitude", "longitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns after schema mapping: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    return df


def load_raw_data(path: Path = INPUT_DATA_PATH) -> pd.DataFrame:
    """Loads enriched apartment listing data from CSV and normalizes schema names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input data not found at '{path}'. "
            "Provide 'data/apartments_data_enriched_with_new_features.csv' "
            "with the required columns."
        )

    df = pd.read_csv(path)
    df = _standardize_input_schema(df)
    return df


# ── Step 2: Filter WG listings ────────────────────────────────────────────────

def filter_wg_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Removes WG/shared-room listings detected by keywords in title or description."""
    text = (
        df["title"].fillna("").str.lower()
        + " "
        + df.get("description", pd.Series([""] * len(df), index=df.index)).fillna("").str.lower()
    )
    mask_wg = pd.Series(False, index=df.index)
    for kw in WG_KEYWORDS:
        mask_wg |= text.str.contains(kw, regex=False)
    before = len(df)
    df = df[~mask_wg].copy()
    print(f"  filter_wg_listings      : removed {before - len(df):>3} rows  → {len(df)} remaining")
    return df.reset_index(drop=True)


# ── Step 3: Filter invalid rows ───────────────────────────────────────────────

def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows with missing or implausible values in critical columns."""
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] >= 400].copy()           # min realistic rent CHF
    df = df.dropna(subset=["apartment_size_sqm"])
    df = df[df["apartment_size_sqm"] >= 15].copy()     # min 15 sqm
    df = df.dropna(subset=["number_of_rooms"])
    df = df[df["number_of_rooms"] >= 1.0].copy()
    print(f"  filter_invalid_rows     : removed {before - len(df):>3} rows  → {len(df)} remaining")
    return df.reset_index(drop=True)


# ── Step 4: Handle missing values ─────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing lat/lon with column medians (required for distance feature)."""
    df = df.copy()
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(
                    f"  handle_missing_values   : imputed {n_missing} '{col}' "
                    f"values with median ({median_val:.4f})"
                )
    return df


# ── Step 5: Feature – distance to Zürich centre ───────────────────────────────

def compute_distance_to_zurich_center(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds *distance_to_zurich_center* (km) using the Haversine formula.

    Engineering rationale
    ---------------------
    Proximity to the city centre is one of the strongest price drivers in the
    Zurich rental market.  Raw lat/lon values would require the model to learn
    a two-dimensional spatial relationship; a single distance scalar captures
    the dominant signal more efficiently.
    """
    R = 6371.0  # Earth radius in km
    lat1 = math.radians(ZURICH_CENTER_LAT)
    lon1 = math.radians(ZURICH_CENTER_LON)
    lat2 = np.radians(df["latitude"].values)
    lon2 = np.radians(df["longitude"].values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df = df.copy()
    df["distance_to_zurich_center"] = np.round(dist, 2)
    return df


# ── Step 6: Feature – furnished ───────────────────────────────────────────────

def extract_furnished(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary *furnished* (1 = furnished) extracted from title/description.

    Engineering rationale
    ---------------------
    Furnished apartments typically command a 10–20 % price premium.  The
    keyword search covers common German and English variants used on Swiss
    portals: möbliert, moebliert, furnished.
    """
    text = (
        df["title"].fillna("").str.lower()
        + " "
        + df.get("description", pd.Series([""] * len(df), index=df.index)).fillna("").str.lower()
    )
    mask = pd.Series(False, index=df.index)
    for kw in FURNISHED_KEYWORDS:
        mask |= text.str.contains(kw, regex=False)
    df = df.copy()
    df["furnished"] = mask.astype(int)
    return df


# ── Step 7: Feature – parking ─────────────────────────────────────────────────

def extract_parking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary *parking* (1 = parking available) extracted from title/description.

    Engineering rationale
    ---------------------
    Having a garage or parking space adds a measurable rent premium (~5–8 %)
    because central Zurich parking is scarce and expensive.  Keywords cover
    Garage, Tiefgarage, Parkplatz, Carport, Einstellplatz, parking.
    """
    text = (
        df["title"].fillna("").str.lower()
        + " "
        + df.get("description", pd.Series([""] * len(df), index=df.index)).fillna("").str.lower()
    )
    mask = pd.Series(False, index=df.index)
    for kw in PARKING_KEYWORDS:
        mask |= text.str.contains(kw, regex=False)
    df = df.copy()
    df["parking"] = mask.astype(int)
    return df


# ── Step 8: Feature – rooms_per_sqm ──────────────────────────────────────────

def compute_rooms_per_sqm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds *rooms_per_sqm* = number_of_rooms / apartment_size_sqm.

    Engineering rationale
    ---------------------
    This ratio captures apartment type (studio vs. large family flat) and
    corrects for multi-collinearity between room count and floor area.
    """
    df = df.copy()
    df["rooms_per_sqm"] = df["number_of_rooms"] / df["apartment_size_sqm"]
    return df


# ── Step 9: Remove outliers ───────────────────────────────────────────────────

def remove_outliers(df: pd.DataFrame, iqr_multiplier: float = 2.0) -> pd.DataFrame:
    """Removes price outliers using the IQR method on the target column."""
    q1 = df[TARGET_COLUMN].quantile(0.25)
    q3 = df[TARGET_COLUMN].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    before = len(df)
    df = df[(df[TARGET_COLUMN] >= lower) & (df[TARGET_COLUMN] <= upper)].copy()
    print(
        f"  remove_outliers         : removed {before - len(df):>3} rows  "
        f"(IQR bounds: {lower:.0f}–{upper:.0f} CHF) → {len(df)} remaining"
    )
    return df.reset_index(drop=True)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    raw_path: Path = INPUT_DATA_PATH,
    clean_path: Path = CLEAN_DATA_PATH,
    save: bool = True,
) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline and optionally persists the result.

    Steps
    -----
    1.  Load raw CSV
    2.  Filter WG / shared-room listings
    3.  Filter rows with missing or invalid price / size / rooms
    4.  Impute missing lat/lon with column medians
    5.  Compute distance_to_zurich_center (Haversine, km)
    6.  Extract *furnished* binary feature
    7.  Extract *parking* binary feature
    8.  Compute rooms_per_sqm
    9.  Remove price outliers (IQR ×2)
    10. Save cleaned CSV

    Returns
    -------
    pd.DataFrame  Clean, model-ready dataframe.
    """
    print("=" * 52)
    print("Preprocessing Pipeline")
    print("=" * 52)

    df = load_raw_data(raw_path)
    print(f"  load_raw_data           : {len(df)} rows loaded from '{raw_path}'")

    df = filter_wg_listings(df)
    df = filter_invalid_rows(df)
    df = handle_missing_values(df)
    df = compute_distance_to_zurich_center(df)
    df = extract_furnished(df)
    df = extract_parking(df)
    df = compute_rooms_per_sqm(df)
    df = remove_outliers(df)

    if save:
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(clean_path, index=False)
        print(f"  Saved → '{clean_path}'")

    print(f"{'=' * 52}")
    print(f"Pipeline complete: {len(df)} clean rows ready for modelling.")
    print(f"{'=' * 52}\n")
    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing_pipeline()

