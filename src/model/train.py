from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import os
from pathlib import Path
from typing import Union, List
from src.model.model_def import get_model
from src.utils.artifacts import save_artifacts

import mlflow
import mlflow.catboost


def train_pipeline(
        X, y,
        cat_features: Union[List, None],
        output_dir: Union[str, Path],
        catboost_dir: Union[str, Path],
        base_dir: Union[str, Path],
        experiment_name: str = "default",
        **model_kwargs
):
    os.makedirs(output_dir, exist_ok=True)

    model = get_model(catboost_dir, **model_kwargs)
    fit_params = {}

    if cat_features:
        fit_params["cat_features"] = cat_features

    # инициализация MLflow
    mlruns_dir = Path(base_dir) / "mlruns"
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)

    test_size = 0.2
    random_state = 42

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run():
        for k, v in model_kwargs.items():
            try:
                mlflow.log_param(k, v)
            except:
                pass

        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        model.fit(X_train, y_train, eval_set=(X_val, y_val), **fit_params)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        metrics = {
            "roc_auc": float(roc_auc_score(y_val, y_proba)),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1": float(f1_score(y_val, y_pred))
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        save_artifacts(model, output_dir, metrics, y_val, y_proba, y_pred)

        try:
            mlflow.catboost.log_model(model, artifact_path="model")
        except Exception:
            model_path = Path(output_dir) / "catboost_model.cbm"
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="model_files")

        metrics_path = Path(output_dir) / "metrics.pkl"
        val_pred_path = Path(output_dir) / "val_predictions.pkl"

        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        if val_pred_path.exists():
            mlflow.log_artifact(str(val_pred_path), artifact_path="val_predictions")

    return model, metrics
