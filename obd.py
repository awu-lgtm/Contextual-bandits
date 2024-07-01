import pandas as pd
from os.path import join
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from tqdm import tqdm
from cb import to_vw_adf_format
from vowpalwabbit import LabelType, Workspace

y_name = "item_id"
cost_name = "click"
prob_name = "propensity_score"

data_path = "open_bandit_dataset"
bts_path = join(data_path, "bts")
bts_all_path = join(bts_path, "all")
bts_dat_path = join(bts_all_path, "all.dat")
bts_item_context_path = join(bts_all_path, "item_context.csv")

bts_csv_path = join(data_path, "bts", "all", "all.csv")
random_csv_path = join(data_path, "random", "all", "all.csv")
bts_parquet_path = join(data_path, "bts", "all", "all.parquet")
random_parquet_path = join(data_path, "random", "all", "all.parquet")


def to_parquet(path="bts"):
    if path == "bts":
        pd.read_csv(bts_csv_path).to_parquet(bts_parquet_path)
    else:
        pd.read_csv(random_csv_path).to_parquet(random_parquet_path)


def make_encoder_and_encode(df: pd.DataFrame, encoder: OrdinalEncoder):
    if encoder is None:
        encoder = OrdinalEncoder()
        encoder.set_output(transform="pandas")
        encoder.fit(df)
    df = encoder.transform(df)
    return df, encoder


def preprocess(df: pd.DataFrame, x_encoder: OrdinalEncoder = None):
    dt = pd.to_datetime(df["timestamp"], format="mixed")
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df = df.drop(["timestamp", "Unnamed: 0"], axis=1)

    y, cost, prob = df[y_name], df[cost_name], df[prob_name]
    cost = 1 - cost
    df = df.drop([y_name, cost_name, prob_name], axis=1)

    df, x_encoder = make_encoder_and_encode(df, x_encoder)
    df = pd.concat([df, y, cost, prob], axis=1)
    return df, x_encoder


def preprocess_item_context(df: pd.DataFrame, encoder: OrdinalEncoder = None):
    ids = df["item_id"]
    df = df.drop(["item_id"], axis=1)
    df, encoder = make_encoder_and_encode(df, encoder)
    df = pd.concat([ids, df], axis=1)
    return df, encoder


def downcast(df: pd.DataFrame):
    for name, col in df.items():
        if name != prob_name:
            df[name] = col.astype(np.int8)
    return df


def train_model(model: Workspace, df: pd.DataFrame, action_context: pd.DataFrame):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        action, cost, prob = row[y_name], row[cost_name], row[prob_name]
        context = row.drop([y_name, cost_name, prob_name])
        df[y_name], df[cost_name], df[prob_name]
        label = (action, cost, prob)
        example = model.parse(
            to_vw_adf_format(context, action_context, label),
            LabelType.CONTEXTUAL_BANDIT,
        )
        model.learn(example)
        model.finish_example(example)
