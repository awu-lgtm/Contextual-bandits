from random import choices
from vowpalwabbit import LabelType, Workspace
import pandas as pd
import numpy as np

def get_cost(y: int, action: int):
    if y == action:
        return -1
    return 0

def to_vw_format(context: pd.Series, label=None):
    string = ""
    if label is not None:
        action, cost, prob = label
        string += f"{action}:{cost}:{prob}"
    string += " | "
    for feature in context:
        string += f"{feature} "
    return string

def get_action(vw: Workspace, x: np.ndarray, A: list):
    context = to_vw_format(x)
    probs = vw.predict(context)
    a = choices(range(len(probs)), weights=probs, k=1)[0]
    return A[a], probs[a]

def simulate_once(model: Workspace, x: np.ndarray, y: int, A: list, learn: bool):
    a, p = get_action(model, x, A)
    cost = get_cost(y, a)

    if learn:
        label = (a, cost, p)
        vw_format = model.parse(to_vw_format(x, label), LabelType.CONTEXTUAL_BANDIT)
        model.learn(vw_format)
    return a, cost

def simulate(model: Workspace, Xs: np.ndarray, ys: np.ndarray, A: list, learn=True):
    actions = []
    costs = []
    for x, y in zip(Xs, ys):
        a, cost = simulate_once(model, x, y, A, learn)
        actions.append(a)
        costs.append(cost)
    return actions, costs