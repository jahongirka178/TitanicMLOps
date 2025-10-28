from pathlib import Path
import joblib


def save_artifacts(model, output_dir, metrics, y_val, preds_proba, preds):
    model_path = Path(output_dir) / "catboost_model.cbm"
    model.save_model(str(model_path))

    joblib.dump(metrics, Path(output_dir) / "metrics.pkl")
    joblib.dump(
        {"y_val": y_val, "proba": preds_proba, "preds": preds},
        Path(output_dir) / "val_predictions.pkl"
    )
