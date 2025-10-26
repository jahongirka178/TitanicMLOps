import mlflow
import joblib
from pathlib import Path


def log_model_mlflow(artifacts_dir: str = "artifacts"):
    artifacts_dir = Path(artifacts_dir)
    model_path = artifacts_dir / "catboost_model.cbm"
    metrics_path = artifacts_dir / "metrics.pkl"
    metrics = joblib.load(metrics_path)
    with mlflow.start_run():
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        print("Logged to MLflow run id:", mlflow.active_run().info.run_id)


if __name__ == '__main__':
    log_model_mlflow()
