from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import MODEL_FEATURES, training_frame


@dataclass
class ModelResult:
    models: dict[str, object]
    metrics: dict[str, dict]
    primary_name: str
    feature_columns: list[str]
    split_dates: dict[str, str]
    feature_importance: pd.DataFrame
    candidates: pd.DataFrame
    screening_summary: dict[str, int | float | str | None]

    @property
    def primary_model(self):
        return self.models[self.primary_name]


def _metrics(model, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "sensitivity_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "test_samples": int(len(y_test)),
    }


def _baseline_random_forest(random_state: int) -> RandomForestClassifier:
    """Build the pre-tuning random forest used for the baseline comparison."""
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=15,
        max_features="sqrt",
        class_weight=None,
        criterion="gini",
        n_jobs=-1,
        random_state=random_state,
    )


def _specificity_score(y_true, y_pred) -> float:
    """Measure the true-negative rate used to avoid one-class predictions."""
    return float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))


_RF_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "specificity": make_scorer(_specificity_score),
}


def _composite_refit(cv_results: dict) -> int:
    """Select the CV candidate with the best geometric mean across metrics."""
    metric_columns = [
        cv_results[f"mean_test_{name}"] for name in _RF_SCORING
    ]
    metric_matrix = np.clip(np.vstack(metric_columns), 1e-6, 1.0)
    composite_scores = np.exp(np.nanmean(np.log(metric_matrix), axis=0))
    return int(np.nanargmax(composite_scores))


def _random_forest_search(
    random_state: int,
    n_iter: int = 32,
    cv_splits: int = 4,
) -> RandomizedSearchCV:
    """Create a reproducible overall-performance search over RF parameters."""
    param_distributions = {
        "n_estimators": [200, 300, 400, 500, 700, 900],
        "max_depth": [None, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30],
        "min_samples_split": [2, 4, 6, 10, 15, 20, 30, 40],
        "min_samples_leaf": [1, 2, 4, 6, 8, 12, 15, 20, 30, 40, 60],
        "max_features": ["sqrt", "log2", None, 0.33, 0.5, 0.75, 1.0],
        "class_weight": [
            None,
            "balanced",
            "balanced_subsample",
            {0: 1.0, 1: 0.75},
            {0: 1.0, 1: 0.9},
            {0: 1.0, 1: 1.1},
            {0: 1.0, 1: 1.25},
            {0: 1.0, 1: 1.5},
        ],
    }
    estimator = RandomForestClassifier(
        criterion="gini",
        n_jobs=-1,
        random_state=random_state,
    )
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=_RF_SCORING,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        refit=_composite_refit,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
        return_train_score=False,
    )


def train_and_screen(
    features: pd.DataFrame,
    threshold: float = 0.55,
    random_state: int = 42,
    rf_search_iter: int = 32,
    rf_cv_splits: int = 4,
) -> ModelResult:
    frame = training_frame(features)
    if frame.empty:
        raise ValueError("No complete rows available for model training.")

    unique_dates = sorted(frame["Date"].dropna().unique())
    train_end = unique_dates[int(len(unique_dates) * 0.60)]
    valid_end = unique_dates[int(len(unique_dates) * 0.80)]
    train = frame[frame["Date"] <= train_end]
    valid = frame[(frame["Date"] > train_end) & (frame["Date"] <= valid_end)]
    test = frame[frame["Date"] > valid_end]
    x_train, y_train = train[MODEL_FEATURES], train["LabelUp5D"].astype(int)
    x_valid, y_valid = valid[MODEL_FEATURES], valid["LabelUp5D"].astype(int)
    x_test, y_test = test[MODEL_FEATURES], test["LabelUp5D"].astype(int)

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=random_state
                ),
            ),
        ]
    )
    baseline_rf = _baseline_random_forest(random_state)
    logistic.fit(x_train, y_train)
    baseline_rf.fit(x_train, y_train)

    # The time split is already kept outside the search.  Within the training
    # period, preserve chronological order for the expanding-window CV folds.
    cv_train = train.sort_values(["Date", "Ticker"])
    x_cv, y_cv = cv_train[MODEL_FEATURES], cv_train["LabelUp5D"].astype(int)
    search = _random_forest_search(
        random_state=random_state,
        n_iter=rf_search_iter,
        cv_splits=rf_cv_splits,
    )
    search.fit(x_cv, y_cv)
    tuned_rf = search.best_estimator_

    models = {
        "LogisticRegression": logistic,
        "RandomForestBaseline": baseline_rf,
        "RandomForest": tuned_rf,
    }
    metrics = {
        "LogisticRegression": {
            "validation": _metrics(logistic, x_valid, y_valid),
            "test": _metrics(logistic, x_test, y_test),
            "train_samples": int(len(train)),
            "validation_samples": int(len(valid)),
        },
        "RandomForestBaseline": {
            "validation": _metrics(baseline_rf, x_valid, y_valid),
            "test": _metrics(baseline_rf, x_test, y_test),
            "train_samples": int(len(train)),
            "validation_samples": int(len(valid)),
        },
        "RandomForest": {
            "validation": _metrics(tuned_rf, x_valid, y_valid),
            "test": _metrics(tuned_rf, x_test, y_test),
            "train_samples": int(len(train)),
            "validation_samples": int(len(valid)),
            "tuning": {
                "method": "RandomizedSearchCV",
                "scoring": (
                    "geometric_mean(accuracy, precision, recall, f1, "
                    "roc_auc, specificity)"
                ),
                "cv": f"TimeSeriesSplit(n_splits={rf_cv_splits})",
                "n_iter": int(rf_search_iter),
                "best_cv_composite": float(
                    np.exp(
                        np.mean(
                            np.log(
                                np.clip(
                                    [
                                        search.cv_results_[f"mean_test_{name}"][
                                            search.best_index_
                                        ]
                                        for name in _RF_SCORING
                                    ],
                                    1e-6,
                                    1.0,
                                )
                            )
                        )
                    )
                ),
                "best_cv_metrics": {
                    name: float(
                        search.cv_results_[f"mean_test_{name}"][search.best_index_]
                    )
                    for name in _RF_SCORING
                },
                "best_params": search.best_params_,
            },
        },
    }

    primary_name = "RandomForest"
    primary = models[primary_name]
    importance = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES,
            "Importance": primary.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    latest = (
        features.dropna(subset=MODEL_FEATURES)
        .sort_values(["Date", "Ticker"])
        .groupby("Ticker", as_index=False)
        .tail(1)
        .copy()
    )
    latest["ProbabilityUp5D"] = primary.predict_proba(latest[MODEL_FEATURES])[:, 1]
    latest["PredictedUp5D"] = (latest["ProbabilityUp5D"] >= threshold).astype(int)
    positive_mean = float(train.loc[train["LabelUp5D"].eq(1), "Future5Return"].mean())
    negative_mean = float(train.loc[train["LabelUp5D"].eq(0), "Future5Return"].mean())
    latest["Expected5DReturn"] = (
        latest["ProbabilityUp5D"] * positive_mean
        + (1 - latest["ProbabilityUp5D"]) * negative_mean
    )
    probability_pass = latest[latest["PredictedUp5D"].eq(1)].copy()
    limit_up_filtered = int(probability_pass["LimitUpFlagApprox"].eq(1).sum())
    candidates = probability_pass[
        probability_pass["LimitUpFlagApprox"].eq(0)
    ].copy()
    candidates = candidates.sort_values(
        ["ProbabilityUp5D", "Expected5DReturn"], ascending=False
    )
    screening_summary = {
        "selection_date": str(pd.Timestamp(latest["Date"].max()).date()),
        "stock_pool": int(len(latest)),
        "probability_pass": int(len(probability_pass)),
        "limit_up_filtered": limit_up_filtered,
        "effective_candidates": int(len(candidates)),
        "average_probability_up_5d": (
            float(candidates["ProbabilityUp5D"].mean())
            if not candidates.empty
            else None
        ),
        "average_expected_5d_return": (
            float(candidates["Expected5DReturn"].mean())
            if not candidates.empty
            else None
        ),
        "maximum_expected_5d_return": (
            float(candidates["Expected5DReturn"].max())
            if not candidates.empty
            else None
        ),
        "probability_threshold": float(threshold),
    }

    return ModelResult(
        models=models,
        metrics=metrics,
        primary_name=primary_name,
        feature_columns=MODEL_FEATURES.copy(),
        split_dates={
            "train_end": str(pd.Timestamp(train_end).date()),
            "validation_end": str(pd.Timestamp(valid_end).date()),
            "test_start": str(test["Date"].min().date()),
        },
        feature_importance=importance,
        screening_summary=screening_summary,
        candidates=candidates[
            [
                "Ticker",
                "StockNameCN",
                "Date",
                "Market",
                "Board",
                "Industry",
                "Close",
                "ProbabilityUp5D",
                "Expected5DReturn",
                "LimitUpFlagApprox",
            ]
        ],
    )
