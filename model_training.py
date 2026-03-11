from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import (
    FEATURE_COLUMNS,
    create_synthetic_dataset,
    prepare_features_and_target,
    remove_outliers_iqr,
)

MODEL_PATH = Path("trained_model.pkl")
REPORT_PATH = Path("model_report.json")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, FEATURE_COLUMNS)],
        remainder="drop",
    )


def evaluate_iteration(
    iteration_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    models: dict[str, object],
    use_feature_selection: bool,
) -> list[dict]:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "rmse": make_scorer(rmse, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": make_scorer(r2_score),
    }

    results = []

    for model_name, estimator in models.items():
        selector_step = SelectKBest(score_func=f_regression, k=2) if use_feature_selection else "passthrough"
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("feature_selector", selector_step),
                ("model", estimator),
            ]
        )

        cv_scores = cross_validate(pipeline, x, y, cv=cv, scoring=scoring)

        result = {
            "iteration": iteration_name,
            "model_name": model_name,
            "hyperparameters": estimator.get_params(),
            "rmse_mean": float(-cv_scores["test_rmse"].mean()),
            "rmse_std": float(cv_scores["test_rmse"].std()),
            "mae_mean": float(-cv_scores["test_mae"].mean()),
            "mae_std": float(cv_scores["test_mae"].std()),
            "r2_mean": float(cv_scores["test_r2"].mean()),
            "r2_std": float(cv_scores["test_r2"].std()),
            "used_feature_selection": use_feature_selection,
        }
        results.append(result)

    return results


def train_and_persist_model(random_state: int = 42) -> dict:
    """Runs iterative model training, compares models, and persists the best model."""
    df = create_synthetic_dataset(n_samples=800, random_state=random_state)

    # Iteration 1: baseline models without outlier removal and without feature selection.
    x_iter1, y_iter1 = prepare_features_and_target(df)
    iteration1_models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_split=4,
            random_state=random_state,
        ),
    }
    iter1_results = evaluate_iteration(
        iteration_name="Iteration 1 - Baseline",
        x=x_iter1,
        y=y_iter1,
        models=iteration1_models,
        use_feature_selection=False,
    )

    # Iteration 2: outlier removal + feature selection + additional model.
    df_iter2 = remove_outliers_iqr(df, FEATURE_COLUMNS, multiplier=1.5)
    x_iter2, y_iter2 = prepare_features_and_target(df_iter2)
    iteration2_models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
            min_samples_split=3,
            random_state=random_state,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=random_state,
        ),
    }
    iter2_results = evaluate_iteration(
        iteration_name="Iteration 2 - Advanced",
        x=x_iter2,
        y=y_iter2,
        models=iteration2_models,
        use_feature_selection=True,
    )

    all_results = iter1_results + iter2_results
    best_result = min(all_results, key=lambda row: row["rmse_mean"])

    if best_result["iteration"] == "Iteration 1 - Baseline":
        final_x, final_y = x_iter1, y_iter1
        final_models = iteration1_models
    else:
        final_x, final_y = x_iter2, y_iter2
        final_models = iteration2_models

    final_selector = SelectKBest(score_func=f_regression, k=2) if best_result["used_feature_selection"] else "passthrough"
    best_estimator = final_models[best_result["model_name"]]

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("feature_selector", final_selector),
            ("model", best_estimator),
        ]
    )
    final_pipeline.fit(final_x, final_y)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(final_pipeline, model_file)

    report = {
        "best_model": {
            "iteration": best_result["iteration"],
            "model_name": best_result["model_name"],
            "hyperparameters": best_result["hyperparameters"],
            "rmse_mean": best_result["rmse_mean"],
            "mae_mean": best_result["mae_mean"],
            "r2_mean": best_result["r2_mean"],
        },
        "all_results": all_results,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    return report


if __name__ == "__main__":
    summary = train_and_persist_model()
    print("Best model:", summary["best_model"])
