import pandas as pd
from random import choices
from vowpalwabbit import LabelType, Workspace
import matplotlib.pyplot as plt
import json
from contextualbandits.evaluation import evaluateFullyLabeled
from sklearn.preprocessing import LabelBinarizer

def preprocess(df: pd.DataFrame):
    # label_mapping = {}
    # for col, dtype in df.dtypes.items():
    #     if dtype == pd.StringDtype:
    #         codes, uniques = df[col].factorize()
    #         df[col] = codes
    #         label_mapping[col] = uniques
    # ids = df["id"]
    # df = df.drop(["id"], axis=1)
    # df.set_index(ids)
    # return df, label_mapping
    lb = LabelBinarizer()
    label_mapping = {}
    for col, dtype in df.dtypes.items():
        if dtype == pd.StringDtype:
            codes, uniques = df[col].factorize()
            df[col] = codes
            label_mapping[col] = uniques
    y = lb.fit_transform(df["NObeyesdad"])
    ids = df["id"]
    X = df.drop(["id", "NObeyesdad"], axis=1)
    X.set_index(ids)
    return X, y, label_mapping

def get_cost(row, label, action):
    if row[label] == action:
        return -1
    return 0

def to_vw_format(context: pd.Series, actions, label=None):
    string = ""
    if label is not None:
        action, cost, prob = label
        string += f"{action}:{cost}:{prob}"
    string += " | "
    for k, feature in context.items():
        if k == "NObeyesdad":
            string += f"{k}:{feature} "
    return string

def get_action(vw, context, actions):
    context = to_vw_format(context, actions)
    # print(context)
    probs = vw.predict(context)
    a = choices(range(len(probs)), weights=probs, k=1)[0]
    return actions[a], probs[a]


def simulator_loop(vw: Workspace, df: pd.DataFrame, actions, label, learn):
    cost_sum = 0
    ctr = []

    for i, (_, row) in enumerate(df.iterrows()):
        a, p = get_action(vw, row, actions)
        cost = get_cost(row, label, a)
        cost_sum += cost

        if learn:
            vw_format = vw.parse(to_vw_format(row, actions, (a, cost, p)), LabelType.CONTEXTUAL_BANDIT)
            for j in vw_format.iter_features():
                print(j)
            print(to_vw_format(row, actions, (a, cost, p)))
            vw.learn(vw_format)
        ctr.append(-1 * cost_sum/(i + 1))
    return ctr

def plot_ctr(ctr):
    plt.plot(range(1, len(ctr) + 1), ctr)
    plt.xlabel("num_iterations", fontsize=14)
    plt.ylabel("ctr", fontsize=14)
    plt.ylim([0, 1])