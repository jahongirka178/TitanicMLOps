from src.model.optuna_tune import run_optuna_study
from src.preprocessing.load_data import load_titanic
from src.preprocessing.preprocess import preprocess, extract_X_y
from src.model.train import train_pipeline
from utils.paths import ConfigPaths
from utils.vizualization import *
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

paths = ConfigPaths()

DATA_PATH = paths.get_path_to_data()
ARTIFACT_PATH = paths.get_path_to_artifacts()
CATBOOST_INFO_PATH = paths.get_path_to_catboost_info()
CATBOOST_INFO_OPTUNA_PATH = paths.get_path_to_catboost_info_optuna()


def main():
    df = load_titanic(DATA_PATH)
    df = preprocess(df)

    X, y = extract_X_y(df, target_col="Survived")
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

    show_metrics(metrics)
    plot_roc_from_artifacts(ARTIFACT_PATH)

    run_optuna_study(df, CATBOOST_INFO_OPTUNA_PATH, ARTIFACT_PATH)


if __name__ == '__main__':
    main()
