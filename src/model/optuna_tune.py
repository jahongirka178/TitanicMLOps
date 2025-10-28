from pathlib import Path

import optuna
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.preprocessing.preprocess import preprocess, extract_X_y
from src.utils.vizualization import plot_optuna_results


def objective(trial, df, catboost_dir):
    X, y = extract_X_y(df, target_col="Survived")
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Предлагаем параметры для набора
    params = {
        "iterations": trial.suggest_int("iterations", low=100, high=500, step=50),
        "depth": trial.suggest_int("depth", low=3, high=7, step=1),
        "learning_rate": trial.suggest_float("learning_rate", low=5e-3, high=0.5, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", low=1e-2, high=100.0, log=True),
        "random_seed": 42,
        "verbose": False,
    }

    model = CatBoostClassifier(train_dir=catboost_dir, **params)
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), verbose=False)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    return acc


def run_optuna_study(df: pd.DataFrame, catboost_dir_optuna: str, output_dir):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df, catboost_dir_optuna), n_trials=20)

    print(f"Лучшие параметры: {study.best_params}")

    joblib.dump(study.best_trial.params, Path(output_dir) / "best_params.joblib")

    print(f"Лучшая точность: {study.best_value:.4f}")

    plot_optuna_results(study, output_dir)
