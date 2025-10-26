from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import joblib
import os
from pathlib import Path
from typing import Union, List
from .model_def import get_catboost_model


def train_pipeline(
        X,
        y,
        cat_features: Union[List, None] = None,
        output_dir: Union[str, Path] = "artifacts",
        **model_kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = get_catboost_model(**model_kwargs)
    fit_params = {}
    if cat_features:
        fit_params["cat_features"] = cat_features

    model.fit(X_train, y_train, eval_set=(X_val, y_val), **fit_params)
    preds_proba = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, preds_proba)),
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1": float(f1_score(y_val, preds))
    }

    model_path = Path(output_dir) / "catboost_model.cbm"
    model.save_model(str(model_path))
    joblib.dump(metrics, Path(output_dir) / "metrics.pkl")
    joblib.dump(
        {"y_val": y_val, "proba": preds_proba, "preds": preds},
        Path(output_dir) / "val_predictions.pkl"
    )

    return model, metrics
