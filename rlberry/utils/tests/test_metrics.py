import pytest
import numpy as np
from rlberry.utils.metrics import metric_lp


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_metrics(dim):
    y = np.zeros(dim)
    x = np.ones(dim)
    scaling_1 = np.ones(dim)
    scaling_2 = 0.5 * np.ones(dim)

    for p in range(1, 10):
        assert np.abs(metric_lp(x, y, p, scaling_1)
                      - np.power(dim, 1.0 / p)) < 1e-15
        assert np.abs(metric_lp(x, y, p, scaling_2)
                      - 2 * np.power(dim, 1.0 / p)) < 1e-15
