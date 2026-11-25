"""Exception-safe deterministic analysis boundaries."""
from functools import wraps
from inspect import signature

import torch


def deterministic_analysis(function):
    """Use the complete deterministic map and restore all caller model flags."""
    sig = signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        model = bound.arguments['model']
        with model.evaluation_context(noise=False, strict_state=True):
            initial = bound.arguments.get('init_states')
            if initial is not None:
                model.validate_state(torch.as_tensor(initial))
            return function(*args, **kwargs)
    return wrapped
