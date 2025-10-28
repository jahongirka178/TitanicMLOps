from pathlib import Path
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.model.model_def import get_catboost_model

def objective(trial, X, y, catboost_dir):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "random_seed": 42
    }


