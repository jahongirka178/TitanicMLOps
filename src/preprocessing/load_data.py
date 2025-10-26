import pandas as pd


def load_data():
    train = pd.read_csv(f"train.csv")
    print(f"Data shape: {train.shape}")
    return train
