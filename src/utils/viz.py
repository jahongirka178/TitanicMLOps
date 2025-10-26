import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_feature_distribution(df: pd.DataFrame, col: str, out_path: str | None = None):
    plt.figure(figsize=(6, 4))
    df[col].hist(bins=30)
    plt.title(col)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()
