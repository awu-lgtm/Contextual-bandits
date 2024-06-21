import pandas as pd
from random import choices
from vowpalwabbit import LabelType, Workspace
import matplotlib.pyplot as plt
import json
from contextualbandits.evaluation import evaluateFullyLabeled
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from tqdm import tqdm

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


def simulate_loop(model, X: pd.DataFrame, y, rewards, action_hist, start, end):
    actions = model.predict(X[start:end,:]).astype('uint8')
    rewards.append(y[np.arange(start, end), actions].sum())
    new_actions_hist = np.concatenate([action_hist, actions])
    model.fit(X[:end], new_actions_hist, y[np.arange(end), new_actions_hist])

    return rewards, new_actions_hist

def simulate(model, X, y, batch_size):
    first_batch = X[:batch_size, :]
    np.random.seed(1)
    action_chosen = np.random.randint(7, size=batch_size)
    rewards_received = y[np.arange(batch_size), action_chosen]

    model.fit(X=first_batch, a=action_chosen, r=rewards_received)
    act_hist = action_chosen
    rewards = []

    for i in tqdm(range(int(np.floor(X.shape[0]/batch_size)))):
        batch_st = (i + 1) * batch_size
        batch_end = (i + 2) * batch_size
        batch_end = np.min([batch_end, X.shape[0]])
        rewards, act_hist = simulate_loop(model, X, y, rewards, act_hist, batch_st, batch_end)
    return rewards, act_hist

def plot_ctr(ctr):
    plt.plot(range(1, len(ctr) + 1), ctr)
    plt.xlabel("num_iterations", fontsize=14)
    plt.ylabel("ctr", fontsize=14)
    plt.ylim([0, 1])