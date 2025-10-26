from typing import Tuple
import pandas as pd

CAT_FEATURES_DEFAULT = ["Sex", "Embarked"]


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    if "Age" in X.columns:
        X["Age"] = X["Age"].fillna(X["Age"].median())
    if "Embarked" in X.columns:
        X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode().iloc[0])
    if "Fare" in X.columns:
        X["Fare"] = X["Fare"].fillna(X["Fare"].median())
    if "Name" in X.columns:
        X["Title"] = X["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
        X["Title"] = X["Title"].str.strip()
        X["Title"] = X["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    if "Sex" in X.columns:
        X["Sex"] = X["Sex"].str.lower()
    drop_cols = [c for c in ["Ticket", "Cabin", "PassengerId", "Name"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
    return X


def split_X_y(df: pd.DataFrame, target_col: str = "Survived") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y
    else:
        return df, None
