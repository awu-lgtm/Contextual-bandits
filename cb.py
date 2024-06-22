from random import choices
from vowpalwabbit import LabelType, Workspace
import pandas as pd
import numpy as np
from tqdm import tqdm


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


def shuffle(Xs: np.ndarray, ys: np.ndarray):
    indices = np.random.permutation(len(Xs))
    return Xs[indices], ys[indices]


def simulate(model: Workspace, Xs: np.ndarray, ys: np.ndarray, A: list, learn=True):
    actions = []
    costs = []

    if learn:
        Xs, ys = shuffle(Xs, ys)
    for x, y in tqdm(zip(Xs, ys), total=len(Xs)):
        a, cost = simulate_once(model, x, y, A, learn)
        actions.append(a)
        costs.append(cost)
    return actions, costs


def evaluate(model: Workspace, Xs: np.ndarray, A: list):
    actions = []
    for x in Xs:
        a, _ = get_action(model, x, A)
        actions.append(a)
    return actions
