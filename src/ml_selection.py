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
    precision_score,
    recall_score,
    roc_auc_score,
)
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


def train_and_screen(
    features: pd.DataFrame,
    threshold: float = 0.55,
    random_state: int = 42,
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

    models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=random_state
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics[name] = {
            "validation": _metrics(model, x_valid, y_valid),
            "test": _metrics(model, x_test, y_test),
            "train_samples": int(len(train)),
            "validation_samples": int(len(valid)),
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
    candidates = latest[
        latest["PredictedUp5D"].eq(1) & latest["LimitUpFlagApprox"].eq(0)
    ].copy()
    candidates = candidates.sort_values(
        ["ProbabilityUp5D", "Expected5DReturn"], ascending=False
    )

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
