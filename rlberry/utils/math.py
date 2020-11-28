import numpy as np
from typing import Union, Tuple

Interval = Union[
    np.ndarray,
    Tuple[float, float],
    Tuple[np.ndarray, np.ndarray]
]


def lmap(v: np.ndarray, x: Interval, y: Interval) -> np.ndarray:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
