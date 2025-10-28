from typing import Tuple
import pandas as pd

CAT_FEATURES_DEFAULT = ["Sex", "Embarked"]


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode().iloc[0])
    X["Fare"] = X["Fare"].fillna(X["Fare"].median())

    X["Title"] = X["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    X["Title"] = X["Title"].str.strip()
    X["Title"] = X["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    X["Sex"] = X["Sex"].str.lower()

    X = X.drop(columns=["Ticket", "Cabin", "PassengerId", "Name"])

    return X


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
    X["AgeBand"] = pd.cut(X["Age"], bins=[0, 12, 20, 40, 60, 120], labels=False)
    X["FareBand"] = pd.qcut(X["Fare"], 4, labels=False, duplicates="drop")

    return X


def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    X = basic_preprocess(X)
    X = add_features(X)
    return X


def extract_X_y(df: pd.DataFrame, target_col: str = "Survived") -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y
