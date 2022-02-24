import numpy as np
import logging
from rlberry.utils.jit_setup import numba_jit


logger = logging.getLogger(__name__)


def polynomial_schedule(
    init_value: float,
    end_value: float,
    power: float,
    transition_steps: int,
    transition_begin: int = 0,
):
    """Constructs a schedule with polynomial transition from init to end value.

    Notes
    -----

    Function taken from: https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py,
    which is licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

    Modifications with respect to source:

    * Remove chex typing from the arguments.
    * Log using a logger :code:`logging.getLogger(__name__)` instead of :code:`logging.info()`.
    * Changed documentation style.

    Parameters
    ----------
    init_value: float
        Initial value for the scalar to be annealed.
    end_value: float
        End value of the scalar to be annealed.
    power: float
        The power of the polynomial used to transition from init to end.
    transition_steps: float
        Number of steps over which annealing takes place,
        the scalar starts changing at `transition_begin` steps and completes
        the transition by `transition_begin + transition_steps` steps.
        If `transition_steps <= 0`, then the entire annealing process is disabled
        and the value is held fixed at `init_value`.
    transition_begin: float
        Must be positive. After how many steps to start annealing
        (before this many steps the scalar value is held fixed at `init_value`).

    Returns
    -------
    schedule: Callable[[int], float]
        A function that maps step counts to values.
    """
    if transition_steps <= 0:
        logger.info(
            "A polynomial schedule was set with a non-positive `transition_steps` "
            "value; this results in a constant schedule with value `init_value`."
        )
        return lambda count: init_value

    if transition_begin < 0:
        logger.info(
            "An exponential schedule was set with a negative `transition_begin` "
            "value; this will result in `transition_begin` falling back to `0`."
        )
        transition_begin = 0

    def schedule(count):
        count = np.clip(count - transition_begin, 0, transition_steps)
        frac = 1 - count / transition_steps
        return (init_value - end_value) * (frac**power) + end_value

    return schedule


@numba_jit
def lambda_returns(r_t, discount_t, v_tp1, lambda_):
    """
    Computer lambda returns

    Parameters
    ----------
    r_t: array
        Array of shape (batch_dim, time_dim) containing the rewards.
    discount_t: array
        Array of shape (batch_dim, time_dim) containing the discounts (0.0 if terminal state).
    v_tp1: array
        Array of shape (batch_dim, time_dim) containing the values at timestep t+1
    lambda_ : float in [0, 1]
        Lambda-returns parameter.
    """
    returns = np.zeros_like(r_t)
    aux = v_tp1[:, -1]
    time_dim = v_tp1.shape[1]
    for tt in range(time_dim):
        i = time_dim - tt - 1
        aux = r_t[:, i] + discount_t[:, i] * (
            (1 - lambda_) * v_tp1[:, i] + lambda_ * aux
        )
        returns[:, i] = aux
    return returns
