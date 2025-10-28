from pathlib import Path
from src.preprocessing.load_data import load_titanic
from src.preprocessing.preprocess import basic_preprocess, extract_X_y
from src.preprocessing.features import add_features
from src.model.train import train_pipeline
from src.utils.metrics import plot_roc, evaluate_and_plot
from utils.paths import ConfigPaths

paths = ConfigPaths()

DATA_PATH = paths.get_path_to_data()
ARTIFACT_PATH = paths.get_path_to_artifacts()
CATBOOST_INFO_PATH = paths.get_path_to_catboost_info()


def main():
    df = load_titanic(DATA_PATH)
    df_proc = basic_preprocess(df)
    df_feat = add_features(df_proc)

    X, y = extract_X_y(df_feat, target_col="Survived")

    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    model, metrics = train_pipeline(
        X, y,
        cat_features=cat_features,
        output_dir=ARTIFACT_PATH,
        catboost_dir=CATBOOST_INFO_PATH,
        random_seed=42,
        iterations=1000,
        learning_rate=0.03,
        depth=6
    )

    evaluate_and_plot(metrics, ARTIFACT_PATH)


if __name__ == '__main__':
    main()
