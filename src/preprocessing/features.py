import pandas as pd


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
    X["AgeBand"] = pd.cut(X["Age"], bins=[0, 12, 20, 40, 60, 120], labels=False)
    X["FareBand"] = pd.qcut(X["Fare"], 4, labels=False, duplicates="drop")

    return X
