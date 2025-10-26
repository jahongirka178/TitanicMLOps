import pandas as pd
from pathlib import Path
from typing import Union


def load_titanic(csv_path: Union[str, Path]):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    return df
