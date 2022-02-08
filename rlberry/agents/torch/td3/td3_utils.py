import logging
import numpy as np
from rlberry.utils.jit_setup import numba_jit


logger = logging.getLogger(__name__)



@numba_jit
def lambda_returns(r_t, discount_t, v_t, lambda_):
  """
  Computer lambda returns
  
  r_t, discount_t, v_t: numpy arrays of shape (batch_size, time).
  lambda_ : float in [0, 1]
  """
  # If scalar make into vector.
  lambda_ = np.ones_like(discount_t) * lambda_

  # Work backwards to compute `G_{T-1}`, ..., `G_0`.
  returns = np.zeros_like(r_t)
  g = v_t[:, -1]
  T = v_t.shape[1]
  for tt in range(T):
    i = T - tt - 1
    g = r_t[:, i] + discount_t[:, i] * ((1 - lambda_[:, i]) * v_t[:, i] + lambda_[:, i] * g)
    returns[:, i] = g
  return returns


@numba_jit
def lambda_returns_no_batch(r_t, discount_t, v_t, lambda_):
  """
  Computer lambda returns
  
  r_t, discount_t, v_t: numpy arrays of shape (time,).
  lambda_ : float in [0, 1]
  """
  # If scalar make into vector.
  lambda_ = np.ones_like(discount_t) * lambda_

  # Work backwards to compute `G_{T-1}`, ..., `G_0`.
  returns = np.zeros_like(r_t)
  g = v_t[-1]
  T = v_t.shape[0]
  for tt in range(T):
    i = T - tt - 1
    g = r_t[i] + discount_t[i] * ((1 - lambda_[i]) * v_t[i] + lambda_[i] * g)
    returns[i] = g
  return returns


def scale_action(action_space, action):
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_space, scaled_action):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))