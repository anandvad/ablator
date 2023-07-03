import builtins
import copy
import enum
import random
import typing as ty
from collections import OrderedDict
from pathlib import Path

import numpy as np
import optuna
from hyperopt import hp
from optuna.distributions import (
    BaseDistribution,
    CategoricalChoiceType,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState
import pandas as pd
from sqlalchemy import Integer, PickleType, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from ablator.main.sampler import GridSampler, Sampler

import ablator.utils.base as butils
from ablator.config.main import ConfigBase
from ablator.config.utils import dict_hash, flatten_nested_dict
from ablator.main.configs import (
    SearchSpace,
)
from ablator.modules.loggers.file import FileLogger
from ablator.utils.file import nested_set

BUDGET = 100
REPETITIONS = 10


def mock_train(config):
    lr = config["train_config.optimizer_config"]["arguments"]["lr"]
    if lr != 0.1:
        perf = lr**2
    else:
        perf = config["b"] ** 2
    return {"loss": perf, "b": config["b"], "lr": lr}


def mock_train_optuna(trial: optuna.Trial):
    b = trial.suggest_float("b", -10, 10)
    if np.random.choice(2):
        return b**2
    lr = trial.suggest_float("lr", 0, 1)
    return lr**2


def search_space():
    return {
        "train_config.optimizer_config": SearchSpace(
            subspaces=[
                {"name": "sgd", "arguments": {"lr": 0.1}},
                {
                    "name": "adam",
                    "arguments": {"lr": {"value_range": (0, 1)}, "wd": 0.9},
                },
                {
                    "name": {"categorical_values": ["adam", "sgd"]},
                    "arguments": {"lr": {"value_range": (0, 1)}, "wd": 0.9},
                },
            ]
        ),
        "b": SearchSpace(value_range=(-10, 10)),
    }


def grid_search_space():
    return {
        "a": SearchSpace(
            subspaces=[
                {"d": {"value_range": (0, 1)}, "c": 0.9},
                {
                    "d": {"value_range": (0, 1)},
                    "c": {
                        "subspaces": [
                            {"i": {"value_range": (0, 1)}},
                            {"i": {"value_range": (0, 1)}},
                        ]
                    },
                },
            ]
        ),
        "b": SearchSpace(value_range=(-10, 10)),
        "c": SearchSpace(value_range=(-10, 10)),
    }


def _ablator_sampler(search_algo: ty.Literal["tpe", "random", "grid"], budget=None):
    budget = BUDGET if budget is None else budget
    space = search_space()
    s = Sampler(
        search_space=space, optim_metrics={"loss": "min"}, search_algo=search_algo
    )
    perfs = []
    for i in range(budget):
        try:
            trial_id, config = s.sample()
        except StopIteration:
            break
        perf = mock_train(config)
        s.update_trial(
            trial_id,
            {"loss": perf["loss"]},
            state="ok",
        )
        perfs.append(perf)
    return pd.DataFrame(perfs)


def _update_tpe():
    space = search_space()
    s = Sampler(search_space=space, optim_metrics={"loss": "min"}, search_algo="tpe")
    perfs = []
    for i in range(BUDGET):
        trial_id, config = s.sample()
        perf = mock_train(config)
        perfs.append(perf)
    for i, perf in enumerate(perfs):
        s.update_trial(
            i,
            {"loss": perf["loss"]},
            state="ok",
        )
    return pd.DataFrame(perfs)


def _optuna_sampler(sampler: ty.Literal["random", "tpe"]):
    if sampler == "random":
        sampler = optuna.samplers.RandomSampler()
    elif sampler == "tpe":
        sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler)  # Create a new study.
    study.optimize(mock_train_optuna, n_trials=BUDGET)
    return pd.DataFrame([{**{"loss": t.values[0]}, **t.params} for t in study.trials])


def _get_top_n(df: pd.DataFrame):
    top_n = int(BUDGET * 0.1)
    return df.sort_values("loss").iloc[:top_n].mean()["loss"].item()
    # return (
    #     df.abs().sort_values("b").iloc[:top_n].mean()['b'].item(),
    #     df.sort_values("lr").iloc[:top_n].mean()['lr'].item(),
    # )


def test_tpe():
    optuna_df = pd.concat([_optuna_sampler("tpe") for i in range(REPETITIONS)])
    tpe_df = pd.concat([_ablator_sampler("tpe") for i in range(REPETITIONS)])
    loss = _get_top_n(tpe_df)
    opt_loss = _get_top_n(optuna_df)
    assert abs(loss - opt_loss) < 0.0001


def test_random():
    optuna_rand_df = pd.concat([_optuna_sampler("random") for i in range(REPETITIONS)])
    rand_df = pd.concat([_ablator_sampler("random") for i in range(REPETITIONS)])
    loss = _get_top_n(rand_df)
    opt_loss = _get_top_n(optuna_rand_df)
    assert abs(loss - opt_loss) < 0.0001


def test_update_tpe():
    # Test whether lazy updates of TPE cause reduction in performance (Expected as it samples at random when not available) however not exactly random as it does not sample from approx close configurations
    update_tpe = pd.concat([_update_tpe() for i in range(REPETITIONS)])
    rand_df = pd.concat([_ablator_sampler("random") for i in range(REPETITIONS)])
    tpe_df = pd.concat([_ablator_sampler("tpe") for i in range(REPETITIONS)])
    tpe2_df = pd.concat([_ablator_sampler("tpe") for i in range(REPETITIONS)])
    loss = _get_top_n(rand_df)
    update_tpe_loss = _get_top_n(update_tpe)
    tpe_loss = _get_top_n(tpe_df)
    tpe2_loss = _get_top_n(tpe2_df)
    assert abs(loss - update_tpe_loss) < 0.0001 and abs(
        update_tpe_loss - tpe_loss
    ) > abs(tpe2_loss - tpe_loss)


def test_grid_sampler():
    space = {"b": SearchSpace(value_range=(-10, 10))}
    sampler = GridSampler(search_space=space, reset=True)
    for i in range(len(sampler.configs)*2):
        sampler.sample()
    assert True
    def _assert_stop_iter():
        sampler = GridSampler(search_space=space, reset=False)
        n_configs = len(sampler.configs)
        for i in range(n_configs*2):
            try:
                sampler.sample()
            except StopIteration:
                assert 0 == len(sampler.configs)
                return
        assert False

    _assert_stop_iter()
    grid_df = _ablator_sampler("grid", budget=BUDGET)
    grid_df = _ablator_sampler("discrete", budget=BUDGET)
    grid2_df = _ablator_sampler("grid", budget=BUDGET * 100)
    grid3_df = _ablator_sampler("grid", budget=BUDGET * 100)
    assert np.isclose(grid3_df["loss"].mean(), grid2_df["loss"].mean())
    assert _get_top_n(grid_df) < 0.1

    space = grid_search_space()
    sampler = GridSampler(space)
    assert len(sampler.configs) == 21000
    cfgs = []
    for i in range(100):
        i, cfg = sampler.sample()
        cfgs.append(cfg)
    sampler2 = GridSampler(space, trials=cfgs)
    assert len(sampler2.configs) == len(sampler.configs) and len(sampler.configs) == (21000 - 100)


if __name__ == "__main__":
    test_grid_sampler()
    breakpoint()
    print()
