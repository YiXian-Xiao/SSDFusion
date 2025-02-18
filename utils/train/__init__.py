import itertools
from typing import Iterable, Union
from torch import nn


def collect_params(*models: nn.Module) -> Iterable:
    params = []

    for model in models:
        params.append(model.parameters())

    return itertools.chain(*params)
