import pandas as pd
import matplotlib.pyplot as plt

def preprocess(df: pd.DataFrame):
    label_mapping = {}
    for col, dtype in df.dtypes.items():
        if dtype == pd.StringDtype:
            codes, uniques = df[col].factorize()
            df[col] = codes
            label_mapping[col] = uniques
    y = df["NObeyesdad"] + 1
    ids = df["id"]
    X = df.drop(["id", "NObeyesdad"], axis=1)
    X.set_index(ids)
    return X, y, label_mapping

def costs_to_ctr(costs: list):
    total_cost = 0
    ctr = []
    for i, cost in enumerate(costs):
        total_cost += cost
        ctr.append(-total_cost/(i+1))
    return ctr

def plot_ctr(ctr: list):
    plt.plot(range(1, len(ctr) + 1), ctr)
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel("ctr", fontsize=14)
    plt.title("iteration vs ctr")
    plt.ylim([0, 1])