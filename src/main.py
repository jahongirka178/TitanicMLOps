from pathlib import Path
from src.preprocessing.load_data import load_titanic
from src.preprocessing.preprocess import basic_preprocess, extract_X_y
from src.preprocessing.features import add_features
from src.model.train import train_pipeline
from src.utils.metrics import plot_roc
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data/train.csv"
ARTIFACT_PATH = BASE_DIR / "artifacts"
CATBOOST_INFO_PATH = BASE_DIR / "catboost_info"


def main(data_path: str = None, ):
    if data_path is None:
        data_path = str(DATA_PATH)
    artifacts_dir = str(ARTIFACT_PATH)
    catboost_dir = str(CATBOOST_INFO_PATH)

    df = load_titanic(data_path)
    df_proc = basic_preprocess(df)
    df_feat = add_features(df_proc)

    X, y = extract_X_y(df_feat, target_col="Survived")

    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    model, metrics = train_pipeline(
        X, y,
        cat_features=cat_features,
        output_dir=artifacts_dir,
        catboost_dir=catboost_dir,
        random_seed=42,
        iterations=1000,
        learning_rate=0.03,
        depth=6
    )

    print("Metrics:", metrics)
    # рисуем ROC если есть val_predictions
    val_path = Path(artifacts_dir) / "val_predictions.pkl"

    if val_path.exists():
        vp = joblib.load(val_path)
        plot_roc(vp["y_val"], vp["proba"], out_path=Path(artifacts_dir) / "roc.png")


if __name__ == '__main__':
    main()
