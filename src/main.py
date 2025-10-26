from pathlib import Path
from src.preprocessing.load_data import load_titanic
from src.preprocessing.preprocess import basic_preprocess, split_X_y
from src.preprocessing.features import add_features
from src.model.train import train_pipeline
from src.utils.metrics import plot_roc
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "train.csv"


def main(data_path: str = None, artifacts_dir: str = "artifacts"):
    if data_path is None:
        data_path = str(DATA_PATH)
    df = load_titanic(data_path)
    df_proc = basic_preprocess(df)
    df_feat = add_features(df_proc)

    X, y = split_X_y(df_feat, target_col="Survived")

    cat_feats = [c for c in X.columns if c.lower() in ["sex", "embarked", "title"] and c in X.columns]
    model, metrics = train_pipeline(
        X, y, cat_features=cat_feats, output_dir=artifacts_dir,
        random_seed=42, iterations=1000, learning_rate=0.03, depth=6
    )
    print("Metrics:", metrics)
    # рисуем ROC если есть val_predictions
    val_path = Path(artifacts_dir) / "val_predictions.pkl"
    if val_path.exists():
        vp = joblib.load(val_path)
        plot_roc(vp["y_val"], vp["proba"], out_path=Path(artifacts_dir) / "roc.png")


if __name__ == '__main__':
    main()
