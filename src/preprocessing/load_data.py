import pandas as pd
from pathlib import Path


def load_titanic(csv_path: str | Path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    return df
