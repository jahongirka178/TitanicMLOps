import pandas as pd


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if set(["SibSp", "Parch"]).issubset(X.columns):
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    if "FamilySize" in X.columns:
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
    if "Age" in X.columns:
        X["AgeBand"] = pd.cut(X["Age"], bins=[0, 12, 20, 40, 60, 120], labels=False)
    if "Fare" in X.columns:
        X["FareBand"] = pd.qcut(X["Fare"], 4, labels=False, duplicates="drop")
    return X
