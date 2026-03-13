"""Shared random-number utilities for tinyTorch.

The module centralizes stochastic behavior so initialization, dropout, and data
shuffling follow the same RNG source by default.
"""

import random as _py_random
from typing import MutableSequence, Optional


_GLOBAL_RNG = _py_random.Random()


def seed(value: int) -> None:
    """Seed the shared tinyTorch RNG."""
    _GLOBAL_RNG.seed(value)


def generator(seed_value: Optional[int] = None) -> _py_random.Random:
    """Create an independent RNG, optionally seeded."""
    rng = _py_random.Random()
    if seed_value is not None:
        rng.seed(seed_value)
    return rng


def random() -> float:
    """Draw a uniform random float in ``[0, 1)``."""
    return _GLOBAL_RNG.random()


def uniform(a: float, b: float) -> float:
    """Draw a uniform random value from ``[a, b]``."""
    return _GLOBAL_RNG.uniform(a, b)


def gauss(mean: float, std: float) -> float:
    """Draw a Gaussian random value."""
    return _GLOBAL_RNG.gauss(mean, std)


def shuffle(values: MutableSequence) -> None:
    """Shuffle a mutable sequence in place."""
    _GLOBAL_RNG.shuffle(values)
