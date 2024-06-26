import pandas as pd
from os.path import join
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

y_name = "item_id"
cost_name = "click"
prob_name = "propensity_score"

data_path = "open_bandit_dataset"
bts_path = join(data_path, "bts")
bts_all_path = join(bts_path, "all")
bts_dat_path = join(bts_all_path, "all.dat")

bts_csv_path = join(data_path, "bts", "all", "all.csv")
random_csv_path = join(data_path, "random", "all", "all.csv")
bts_parquet_path = join(data_path, "bts", "all", "all.parquet")
random_parquet_path = join(data_path, "random", "all", "all.parquet")


def to_parquet(path="bts"):
    if path == "bts":
        pd.read_csv(bts_csv_path).to_parquet(bts_parquet_path)
    else:
        pd.read_csv(random_csv_path).to_parquet(random_parquet_path)


def preprocess(df: pd.DataFrame, x_encoder: OrdinalEncoder = None):
    dt = pd.to_datetime(df["timestamp"], format="mixed")
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df = df.drop(["timestamp", "Unnamed: 0"], axis=1)

    y, cost, prob = df[y_name], df[cost_name], df[prob_name]
    df = df.drop([y_name, cost_name, prob_name], axis=1)

    if x_encoder is None:
        x_encoder = OrdinalEncoder()
        x_encoder.set_output(transform="pandas")
        x_encoder.fit(df)
    df = x_encoder.transform(df)
    df = pd.concat([df, y, cost, prob], axis=1)
    return df, x_encoder


def downcast(df: pd.DataFrame):
    for name, col in df.items():
        if name != prob_name:
            df[name] = col.astype(np.int8)
    return df
