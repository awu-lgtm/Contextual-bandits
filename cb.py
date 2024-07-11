from random import choices
from vowpalwabbit import LabelType, Workspace
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import TypeAlias

Action: TypeAlias = int
Cost: TypeAlias = int
Prob: TypeAlias = float


def get_cost(y: int, action: int, offset=-1):
    if y == action:
        cost = 0
    else:
        cost = 1
    cost += offset
    return cost


def to_vw_format(context: pd.Series, label: tuple[Action, Cost, Prob] = None):
    string = ""
    if label is not None:
        action, cost, prob = label
        string += f"{action}:{cost}:{prob}"
    string += " | "
    for feature in context:
        string += f"{feature} "
    return string


def to_vw_adf_format(
    context: pd.Series,
    action_context: pd.DataFrame = None,
    label: tuple[Action, Cost, Prob] = None,
):
    if label is not None:
        a, cost, prob = label
    string = "shared | "
    for k, v in context.items():
        string += f"{k}={v} "
    string += "\n"
    for i, row in action_context.iterrows():
        if i == a:
            string += f"0:{cost}:{prob}"
        string += " | "
        for k, v in row.items():
            string += f"{k}={v} "
        string += "\n"
    return string


def series_to_vw_format(series: pd.Series, y_name: str, cost_name: str, prob_name: str):
    y, cost, prob = series[y_name], series[cost_name], series[prob_name]
    context = series.drop(labels=[y_name, cost_name, prob_name])
    return to_vw_format(context, (y, cost, prob))


def series_to_vw_adf_format(
    series: pd.Series,
    action_context: pd.DataFrame,
    y_name: str,
    cost_name: str,
    prob_name: str,
):
    y, cost, prob = series[y_name], series[cost_name], series[prob_name]
    context = series.drop(labels=[y_name, cost_name, prob_name])
    return to_vw_adf_format(context, action_context, (y, cost, prob))


def df_to_vw_format(df: pd.DataFrame, y_name: str, cost_name: str, prob_name: str):
    format_fn = lambda series: series_to_vw_format(series, y_name, cost_name, prob_name)
    df = df.apply(format_fn, axis=1)
    return df


def df_to_vw_adf_format(
    df: pd.DataFrame,
    action_context: pd.Series,
    y_name: str,
    cost_name: str,
    prob_name: str,
):
    format_fn = lambda series: series_to_vw_adf_format(
        series, action_context, y_name, cost_name, prob_name
    )
    df = df.apply(format_fn, axis=1)
    return df


def df_to_dat(
    df: pd.DataFrame,
    y_name: str,
    cost_name: str,
    prob_name: str,
    save_path: str,
    mode="a",
):
    df: pd.Series = df_to_vw_format(df, y_name, cost_name, prob_name)
    with open(save_path, mode) as f:
        for _, v in df.items():
            f.write(v)
            f.write("\n")


def df_to_adf_dat(
    df: pd.DataFrame,
    action_context: pd.DataFrame,
    y_name: str,
    cost_name: str,
    prob_name: str,
    save_path: str,
    mode="a",
):
    df: pd.Series = df_to_vw_adf_format(
        df, action_context, y_name, cost_name, prob_name
    )
    with open(save_path, mode) as f:
        for _, v in df.items():
            f.write(v)
            f.write("\n")


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
        model.finish_example(vw_format)
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
