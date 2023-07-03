from ablator.main.configs import SearchSpace


def test_search_space():
    optim_config = SearchSpace(
        subspaces=[
            {"name": "sgd", "arguments": {"lr": 0.1}},
            {
                "name": "adam",
                "arguments": {"lr": {"value_range": (0, 1)}, "wd": 0.9},
            },
            {
                "name": "adam",
                "arguments": {
                    "lr": {
                        "subspaces": [
                            {"value_range": (0, 1)},
                            {"value_range": (0, 1)},
                        ]
                    },
                    "wd": 0.9,
                },
            },
        ]
    )
    space = {
        "train_config.optimizer_config": optim_config,
        "b": SearchSpace(value_range=(-10, 10)),
    }
    lr_sp = (
        space["train_config.optimizer_config"]
        .subspaces[1]
        ._constant_values["arguments"]["lr"]
    )
    assert (
        isinstance(
            lr_sp,
            SearchSpace,
        )
        and lr_sp.value_range
        == ["0", "1"]  # this is because we cast to str for float safety
        and lr_sp.categorical_values is None
    )
    assert isinstance(
        space["train_config.optimizer_config"]
        .subspaces[1]
        ._constant_values["arguments"]["lr"],
        SearchSpace,
    )
    assert (
        space["train_config.optimizer_config"]
        .subspaces[1]
        ._constant_values["arguments"]["wd"]
        == 0.9
    )
    space["train_config.optimizer_config"].make_dict(
        space["train_config.optimizer_config"].annotations
    )


if __name__ == "__main__":
    test_search_space()
