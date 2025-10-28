from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import os
from pathlib import Path
from typing import Union, List
from src.model.model_def import get_model
from src.utils.artifacts import save_artifacts


def train_pipeline(
        X,
        y,
        cat_features: Union[List, None],
        output_dir: Union[str, Path],
        catboost_dir: Union[str, Path],
        **model_kwargs
):
    os.makedirs(output_dir, exist_ok=True)

    model = get_model(catboost_dir, **model_kwargs)
    fit_params = {}

    if cat_features:
        fit_params["cat_features"] = cat_features

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), **fit_params)

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred))
    }

    save_artifacts(model, output_dir, metrics, y_val, y_proba, y_pred)

    return model, metrics
