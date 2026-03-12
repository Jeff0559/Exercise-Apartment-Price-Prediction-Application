"""
Iterative Model Training Pipeline – Zurich Apartment Price Prediction
======================================================================
Implements two modelling iterations with 5-fold cross-validation and
persists the best model plus a JSON report.

Iteration 1 – Baseline
-----------------------
Features : apartment_size_sqm, number_of_rooms, rooms_per_sqm
Models   : LinearRegression, RandomForestRegressor

Iteration 2 – Extended Features
--------------------------------
Features : + distance_to_zurich_center, furnished, parking
Models   : LinearRegression, RandomForestRegressor, GradientBoostingRegressor

Evaluation metrics (all via cross-validation):
    RMSE, MAE, R²

The model with the lowest cross-validated RMSE is selected, fitted on the
full dataset, and saved to trained_model.pkl.  A summary report is written
to model_report.json.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import (
    BASELINE_FEATURES,
    CLEAN_DATA_PATH,
    EXTENDED_FEATURES,
    TARGET_COLUMN,
    run_preprocessing_pipeline,
)

# ── Output paths ──────────────────────────────────────────────────────────────
MODEL_PATH  = Path("trained_model.pkl")
REPORT_PATH = Path("model_report.json")

RANDOM_STATE = 42


# ── Metric helpers ────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


SCORING = {
    "rmse": make_scorer(_rmse, greater_is_better=False),
    "mae":  make_scorer(mean_absolute_error, greater_is_better=False),
    "r2":   make_scorer(r2_score),
}


# ── Pipeline builder ──────────────────────────────────────────────────────────

def _build_pipeline(features: list[str], estimator) -> Pipeline:
    """Returns a sklearn Pipeline: median imputation → standard scaling → model."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale",  StandardScaler()),
                ]),
                features,
            )
        ],
        remainder="drop",
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        estimator),
    ])


# ── Cross-validated evaluation ────────────────────────────────────────────────

def evaluate_models(
    iteration_label: str,
    x: pd.DataFrame,
    y: pd.Series,
    models: dict[str, object],
    cv_folds: int = 5,
) -> list[dict]:
    """
    Runs 5-fold cross-validation for each model and returns evaluation records.

    Parameters
    ----------
    iteration_label : str
        Human-readable label stored in the result dictionary.
    x               : pd.DataFrame
        Feature matrix.
    y               : pd.Series
        Target vector.
    models          : dict[str, estimator]
        Mapping of {model_name: unfitted estimator}.
    cv_folds        : int
        Number of cross-validation folds (default: 5).

    Returns
    -------
    list[dict]  One record per model with RMSE, MAE, R² (mean ± std).
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results: list[dict] = []

    for model_name, estimator in models.items():
        pipeline = _build_pipeline(list(x.columns), estimator)
        scores   = cross_validate(pipeline, x, y, cv=cv, scoring=SCORING)

        record = {
            "iteration":       iteration_label,
            "model_name":      model_name,
            "hyperparameters": estimator.get_params(),
            "rmse_mean":       float(-scores["test_rmse"].mean()),
            "rmse_std":        float( scores["test_rmse"].std()),
            "mae_mean":        float(-scores["test_mae"].mean()),
            "mae_std":         float( scores["test_mae"].std()),
            "r2_mean":         float( scores["test_r2"].mean()),
            "r2_std":          float( scores["test_r2"].std()),
        }
        results.append(record)

        print(
            f"  [{iteration_label}] {model_name:<30} "
            f"RMSE={record['rmse_mean']:7.1f} ± {record['rmse_std']:5.1f}  "
            f"MAE={record['mae_mean']:7.1f}  "
            f"R²={record['r2_mean']:.3f}"
        )

    return results


# ── Persist best model ────────────────────────────────────────────────────────

def _persist_best_model(
    best_record: dict,
    features: list[str],
    x: pd.DataFrame,
    y: pd.Series,
    models_lookup: dict[str, object],
) -> None:
    """Fits the winning model on the full dataset and saves it with its feature list."""
    estimator = models_lookup[best_record["model_name"]]
    pipeline  = _build_pipeline(features, estimator)
    pipeline.fit(x, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as fh:
        pickle.dump({"model": pipeline, "features": features}, fh)
    print(f"\n  ✓ Best model saved → '{MODEL_PATH}'")


# ── Main training entry point ─────────────────────────────────────────────────

def train_and_persist_model(random_state: int = RANDOM_STATE) -> dict:
    """
    Runs both training iterations, selects the best model, and persists it.

    Returns
    -------
    dict  Full model report (mirrored in model_report.json).
    """
    # ── Load / preprocess data ────────────────────────────────────────────────
    print("Running preprocessing pipeline from enriched input…\n")
    df = run_preprocessing_pipeline()
    print(f"Loaded clean data: {len(df)} rows from '{CLEAN_DATA_PATH}'")

    y = df[TARGET_COLUMN]
    all_results: list[dict] = []

    # ── Iteration 1: Baseline features ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Iteration 1 – Baseline  |  features:", BASELINE_FEATURES)
    print("=" * 70)
    x1 = df[BASELINE_FEATURES]
    iter1_models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=4,
            random_state=random_state,
        ),
    }
    results_iter1 = evaluate_models(
        "Iteration 1 – Baseline", x1, y, iter1_models
    )
    all_results.extend(results_iter1)

    # ── Iteration 2: Extended features ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Iteration 2 – Extended  |  features:", EXTENDED_FEATURES)
    print("=" * 70)
    x2 = df[EXTENDED_FEATURES]
    iter2_models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=random_state,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=random_state,
        ),
    }
    results_iter2 = evaluate_models(
        "Iteration 2 – Extended", x2, y, iter2_models
    )
    all_results.extend(results_iter2)

    # ── Select best model ─────────────────────────────────────────────────────
    best_record = min(all_results, key=lambda r: r["rmse_mean"])
    print(
        f"\n  ✓ Best model: [{best_record['iteration']}] "
        f"{best_record['model_name']}  "
        f"(RMSE={best_record['rmse_mean']:.1f} CHF)"
    )

    # Determine which feature set and model dict belong to the winning iteration
    if "Iteration 1" in best_record["iteration"]:
        best_features      = BASELINE_FEATURES
        best_models_lookup = iter1_models
        x_best             = x1
    else:
        best_features      = EXTENDED_FEATURES
        best_models_lookup = iter2_models
        x_best             = x2

    _persist_best_model(best_record, best_features, x_best, y, best_models_lookup)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "best_model": {
            **best_record,
            "features": best_features,
        },
        "all_results":       all_results,
        "n_training_samples": int(len(df)),
        "target_column":      TARGET_COLUMN,
        "random_state":       random_state,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"  ✓ Model report saved  → '{REPORT_PATH}'\n")

    return report


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_persist_model()




