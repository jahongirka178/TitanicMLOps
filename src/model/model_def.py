from catboost import CatBoostClassifier


def get_model(catboost_dir,
              random_seed: int = 42,
              iterations: int = 1000,
              learning_rate: float = 0.01,
              depth: int = 6
              ):
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        verbose=100,
        train_dir=catboost_dir
    )
    return model
