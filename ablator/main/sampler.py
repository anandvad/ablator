import copy
import enum
import random
import typing as ty
from collections import OrderedDict
from pathlib import Path
import builtins

import numpy as np
import optuna
from sqlalchemy import Integer, PickleType, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from ablator.config.main import ConfigBase

import ablator.utils.base as butils
from ablator.main.configs import (
    Optim,
    ParallelConfig,
    SearchAlgo,
    SearchSpace,
    SearchType,
)
from ablator.modules.loggers.file import FileLogger
from ablator.utils.file import nested_set
from hyperopt import hp

from ablator.config.utils import dict_hash, flatten_nested_dict

from optuna.trial import TrialState
from optuna.study._study_direction import StudyDirection
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution


class _Trial:
    """
    Mock Trial object for the sake of using optuna
    """

    def __init__(self, id_: int, sampler: "Sampler" = None) -> None:
        self.state = TrialState.RUNNING

        self.values: np.ndarray | None = None
        self.id_ = id_
        self.params = {}
        self.distributions = {}
        if sampler is not None:
            self.relative_search_space = sampler.sampler.infer_relative_search_space(
                sampler, self
            )
            self.relative_params = sampler.sampler.sample_relative(
                sampler, self, self.relative_search_space
            )

    def update(self, values: np.ndarray, state):
        self.values = values
        assert state in {"ok", "fail"}
        self.state = TrialState.COMPLETE if state == "ok" else TrialState.FAIL

    def _is_relative_param(self, name: str, distribution: BaseDistribution) -> bool:
        if name not in self.relative_params:
            return False

        if name not in self.relative_search_space:
            raise ValueError(
                "The parameter '{}' was sampled by `sample_relative` method "
                "but it is not contained in the relative search space.".format(name)
            )

        param_value = self.relative_params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        return distribution._contains(param_value_in_internal_repr)


def _parse_value_range(value_range, fn):
    low_str, high_str = value_range
    low = fn(low_str)
    high = fn(high_str)
    assert min(low, high) == low, "`value_range` must be in the format of (min,max)"
    return low, high


def expand_search_space(search_space: dict[str, ty.Any]):
    configs = [{}]

    def _parse_search_space(space):
        if len(space._constant_values) > 0:
            return expand_search_space(space._constant_values)
        elif space.value_range is not None:
            low, high = _parse_value_range(
                space.value_range,
                float if space.value_type == SearchType.numerical else int,
            )
            step = (high - low) / space.n_bins if space.n_bins is not None else 10
            if space.value_type == SearchType.integer:
                step = max(int(step), 1)
            return np.linspace(low, high, step).tolist()
        elif space.categorical_values is not None:
            return space.categorical_values
        elif space.subspaces is not None:
            return [_ for _v in space.subspaces for _ in _parse_search_space(_v)]

    for k, v in search_space.items():
        if isinstance(v, dict):
            _configs = []
            for _config in configs:
                for _v in expand_search_space(v):
                    _config[k] = _v
                    _configs.append(copy.deepcopy(_config))

            configs = _configs
        elif isinstance(v, SearchSpace):
            _configs = []
            for _config in configs:
                for _v in _parse_search_space(v):
                    _config[k] = _v
                    _configs.append(copy.deepcopy(_config))
            configs = _configs
        else:
            for _config in configs:
                _config[k] = v
    return configs


class GridSampler:
    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        trials: list[_Trial] | None = None,
        reset: bool = True,
    ) -> None:
        self.search_space = search_space
        self.configs = expand_search_space(search_space)
        self._idx = 0
        if trials is not None:
            for t in trials:
                if t in self.configs:
                    del self.configs[self.configs.index(t)]

            self._idx = len(trials)
        self.reset = reset
        self.configs = np.random.permutation(self.configs).tolist()

    def sample(self):
        if len(self.configs) == 0 and self.reset:
            # reset
            self.configs = expand_search_space(self.search_space)
            self.configs = np.random.permutation(self.configs).tolist()
        elif len(self.configs) == 0:
            raise StopIteration

        return_tupple = self._idx, self.configs.pop()
        self._idx += 1
        return return_tupple


class Sampler:
    """
    A class to store the state of the Optuna study.

    Attributes
    ----------
    optim_metrics : OrderedDict
        The ordered dictionary containing the names of the metrics to optimize and their direction
        (minimize or maximize).
    search_space : dict of str to SearchSpace
        The search space containing the parameters to sample from.
    optuna_study : optuna.study.Study
        The Optuna study object.
    """

    def __init__(
        self,
        search_space: dict[str, SearchSpace],
        optim_metrics: dict[str, Optim] | None = None,
        trials: None = None,
        search_algo: SearchAlgo | None = None,
    ) -> None:
        """
        Initialize the Optuna state.

        Parameters
        ----------
        storage : str
            The path to the database URL or a database URL.
        study_name : str
            The name of the study.
        optim_metrics : dict[str, Optim]
            A dictionary of metric names and their optimization directions (either ``'max'`` or ``'min'``).
        search_algo : SearchAlgo
            The search algorithm to use (``'random'`` or ``'tpe'``).
        search_space : dict[str, SearchSpace]
            A dictionary of parameter names and their corresponding SearchSpace instances.

        Raises
        ------
        NotImplementedError
            If the specified search algorithm is not implemented.
        ValueError
            If ``optim_metrics`` is ``None``.

        Notes
        -----
        For tuning, add an attribute to the searchspace whose name is the name of the hyperparameter
        and whose value is the search space
        eg. ``search_space = {"train_config.optimizer_config.arguments.lr": SearchSpace(value_range=[0, 0.1], value_type="float")}``
        """

        if search_algo == SearchAlgo.random or search_algo is None:
            sampler = optuna.samplers.RandomSampler()
        elif search_algo == SearchAlgo.grid:
            sampler = GridSampler(search_space, trials, reset=False)
        elif search_algo == SearchAlgo.discrete:
            sampler = GridSampler(search_space, trials, reset=True)
        elif search_algo == SearchAlgo.tpe:
            sampler = optuna.samplers.TPESampler(constant_liar=True)
        else:
            raise NotImplementedError

        self.sampler = sampler
        if optim_metrics is None and search_algo == SearchAlgo.tpe:
            raise ValueError("Must specify optim_metrics.")
        self.trials: list[_Trial] = []
        self.optim_metrics = OrderedDict(optim_metrics)
        self.directions = [
            StudyDirection.MINIMIZE
            if v == Optim.min or v == "min"
            else StudyDirection.MAXIMIZE
            for v in self.optim_metrics.values()
        ]
        self.search_space = search_space

    def get_trials(self, deepcopy, states):
        assert deepcopy == False
        return [t for t in self.trials if t.state in states]

    def sample(self):
        """
        Sample a new set of trial parameters.

        Returns
        -------
        Tuple[int, dict[str, Any]]
            A tuple of the trial number and a dictionary of parameter names and their corresponding values.
        """
        if isinstance(self.sampler, GridSampler):
            idx, config = self.sampler.sample()
            trial = _Trial(id_=idx)
            trial.params = config
            trial.state = TrialState.RUNNING
        else:
            trial = _Trial(id_=len(self.trials), sampler=self)
            config = self._sample_trial_params(trial, self.search_space)
            trial.state = TrialState.RUNNING
        self.trials.append(trial)
        return trial.id_, config

    def update_trial(
        self, trial_id, result: dict[str, float], state: ty.Literal["ok", "fail"]
    ):
        assert result.keys() == self.optim_metrics.keys()
        values = [result[k] for k in self.optim_metrics]
        self.trials[trial_id].update(values, state)

    def _discretize_search_space(self, search_space):
        pass

    def _is_multi_objective(self):
        return len(self.directions) > 1

    def _suggest(self, trial: _Trial, name, dist):
        if trial._is_relative_param(name, dist):
            val = trial.relative_params[name]
        else:
            val = self.sampler.sample_independent(self, trial, name, dist)
        trial.params[name] = val
        trial.distributions[name] = dist
        return val

    def _suggest_int(self, trial: _Trial, name, low, high, log=False, n_bins=None):
        if n_bins is None:
            step = 1
        else:
            step = max((high - low) // n_bins, 1)
        dist = IntDistribution(low, high, log=log, step=step)
        return self._suggest(trial, name, dist)

    def _suggest_float(self, trial: _Trial, name, low, high, log=False, n_bins=None):
        if n_bins is None:
            step = n_bins
        else:
            step = (high - low) / n_bins
        dist = FloatDistribution(low, high, log=log, step=step)
        return self._suggest(trial, name, dist)

    def _suggest_categorical(self, trial: _Trial, name, vals: list[str]):
        dist = CategoricalDistribution(choices=vals)
        return self._suggest(trial, name, dist)

    def _sample_trial_params(
        self, trial: _Trial, search_space: dict[str, SearchSpace], _prefix: str = ""
    ) -> dict[str, ty.Any]:
        parameter: dict[str, ty.Any] = {}

        for k, v in search_space.items():
            if isinstance(v, dict):
                parameter[k] = self._sample_trial_params(
                    trial, v, _prefix=_prefix + "_" + k
                )
            elif not isinstance(v, SearchSpace):
                parameter[k] = v
            elif v.value_range is not None and v.value_type == SearchType.integer:
                low, high = _parse_value_range(v.value_range, int)
                parameter[k] = self._suggest_int(trial, _prefix + "_" + k, low, high)
            elif v.value_range is not None and v.value_type == SearchType.numerical:
                low, high = _parse_value_range(v.value_range, float)
                parameter[k] = self._suggest_float(trial, _prefix + "_" + k, low, high)
            elif v.categorical_values is not None:
                parameter[k] = self._suggest_categorical(
                    trial, _prefix + "_" + k, v.categorical_values
                )
            elif v.subspaces is not None:
                idx = np.random.choice(len(v.subspaces))
                parameter[k] = self._sample_trial_params(
                    trial,
                    {k: v.subspaces[idx]},
                    _prefix=_prefix + f"{_prefix}_{idx}",
                )[k]
            elif getattr(v, "_constant_values", None) is not None:
                parameter[k] = self._sample_trial_params(
                    trial, v._constant_values, _prefix=_prefix + "_" + k
                )
            else:
                raise ValueError(f"Invalid SearchSpace {v}.")

        return parameter
