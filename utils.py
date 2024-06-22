import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from IPython.display import display
import math
from pandas.api.types import is_object_dtype
import seaborn as sns
import numpy as np

def preprocess(df: pd.DataFrame, x_encoder: OrdinalEncoder=None, y_encoder: LabelEncoder=None, test=False):
    if not test:
        y, y_encoder = encode_y(df["NObeyesdad"], y_encoder)
        y += 1
        df = df.drop(["NObeyesdad"], axis=1)

    X = df.set_index("id")
    X, x_encoder = encode_x(X, x_encoder)
    X = feature_engineering(X)

    # if x_scaler is None:
    #     x_scaler = RobustScaler()
    #     x_scaler.set_output(transform="pandas")
    #     x_scaler.fit(X)
    # X = x_scaler.transform(X)

    if not test:
        return X, y, x_encoder, y_encoder
    return X, x_encoder, y_encoder

def encode_y(y: pd.Series, y_encoder: LabelEncoder):
    if y_encoder is None:
        y_encoder = LabelEncoder()
        y_encoder.fit(y)
    y = y_encoder.transform(y)
    y = pd.Series(y, name="NObeyesdad")
    return y, y_encoder

def encode_x(X: pd.DataFrame, x_encoder: OrdinalEncoder):
    encoded_cols = []
    encoded_names = []
    for name, col in X.items():
        if is_object_dtype(col):
            encoded_names.append(name)
            encoded_cols.append(col)

    if x_encoder is None:
        x_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        x_encoder.set_output(transform="pandas")
        x_encoder.fit(pd.concat(encoded_cols, axis=1))
    X[encoded_names] = x_encoder.transform(pd.concat(encoded_cols, axis=1))
    return X, x_encoder

def feature_engineering(X: pd.DataFrame):
    X["BMI"] = X["Weight"]/X["Height"]**2
    X["BMI_group"] = group_series(X["BMI"], [18.5, 25, 30, 35, 40])
    X["FAVC-FCVC"] = X["FAVC"] - X["FCVC"]
    X["BMI*FAF"] = X["BMI"] * X["FAF"]
    X["FAF-TUE"] = X["FAF"] - X["TUE"]
    X["FCVC*NCP"] = X["FCVC"] * X["NCP"]
    X["BMI/NCP"] = X["BMI"]/X["NCP"]
    X["Age_group"] = group_series(X["Age"], [10, 20, 30, 40, 50, 60, 70])
    return X

def group_series(series: pd.Series, bounds):
    labels = list(range(len(bounds)))
    conditions = [series < bound for bound in bounds]
    return np.select(conditions, labels, len(labels))

def costs_to_moving_avg(costs: list):
    total_cost = 0
    avg_costs = []
    for i, cost in enumerate(costs):
        total_cost += cost
        avg_costs.append(-total_cost/(i+1))
    return avg_costs

def get_num_rows_cols(total: int):
    num_rows = int(math.ceil(math.sqrt(total)))
    while total % num_rows != 0:
        num_rows -= 1
    num_cols = total//num_rows
    return num_rows, num_cols

def get_ax(i: int, axes, num_rows, num_cols):
    if num_cols == 1 & num_rows == 1:
        ax = axes
    elif num_cols == 1:
        ax = axes[i%num_cols]
    elif num_rows == 1:
        ax = axes[i%num_cols]
    else:
        ax = axes[i//num_cols, i%num_cols]
    return ax

def plot_categorical_distr(df: pd.DataFrame, cols: list[str], figsize: tuple[int, int]):
    num_rows, num_cols = get_num_rows_cols(len(cols))
    _, axes = plt.subplots(num_rows, num_cols)
    for i, col in enumerate(cols):
        ax = get_ax(i, axes, num_rows, num_cols)
        counts = df[col].value_counts()
        counts.sort_index().plot.bar(ax=ax, figsize=figsize)

def plot_numerical_distr(df: pd.DataFrame, cols: list[str], figsize: tuple[int, int]):
    num_rows, num_cols = get_num_rows_cols(len(cols))
    df[cols].plot.kde(layout=(num_rows, num_cols), subplots=True, sharex=False, figsize=figsize)

def plot_average_costs(avg_costs: list):
    plt.plot(range(1, len(avg_costs) + 1), avg_costs)
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel("moving average cost", fontsize=14)
    plt.title("moving average cost vs iteration")
    plt.ylim([0, 1])

def print_stats(df: pd.DataFrame):
    print("head")
    display(df.head())
    print("description")
    display(df.describe())
    print("info")
    print(df.info())

def get_top_correlations(df: pd.DataFrame, threshold: 0.9):
    lower_corr = df.corr().reset_index().melt(id_vars="index", value_name="corr").loc[lambda df: df["index"] < df["variable"]]
    return lower_corr.loc[lambda df: df["corr"] > threshold]

def plot_correlation(df: pd.DataFrame):
    sns.heatmap(df.corr())