import torch

from rlberry.envs import Wrapper
import logging
import numpy as np
from rlberry.utils.factory import load

logger = logging.getLogger(__name__)


class UncertaintyEstimatorWrapper(Wrapper):
    """
    Adds exploration bonuses to the info output of env.step(), according to an
    instance of UncertaintyEstimator.

    Example
    -------

    ```
    observation, reward, done, info = env.step(action)
    bonus = info['exploration_bonus']
    ```

    Parameters
    ----------
    uncertainty_estimator_fn : function(observation_space,
                                        action_space, **kwargs)
        Function that gives an instance of UncertaintyEstimator,
        used to compute bonus.
    uncertainty_estimator_kwargs:
        kwargs for uncertainty_estimator_fn
    bonus_scale_factor : double
            Scale factor for the bonus.
    """

    def __init__(self,
                 env,
                 uncertainty_estimator_fn,
                 uncertainty_estimator_kwargs=None,
                 bonus_scale_factor=1.0,
                 bonus_max=np.inf):
        Wrapper.__init__(self, env)

        self.bonus_scale_factor = bonus_scale_factor
        self.bonus_max = bonus_max
        uncertainty_estimator_kwargs = uncertainty_estimator_kwargs or {}

        uncertainty_estimator_fn = load(uncertainty_estimator_fn) if isinstance(uncertainty_estimator_fn, str) else \
            uncertainty_estimator_fn
        self.uncertainty_estimator = uncertainty_estimator_fn(
            env.observation_space,
            env.action_space,
            **uncertainty_estimator_kwargs)
        self.previous_obs = None

    def reset(self):
        self.previous_obs = self.env.reset()
        return self.previous_obs

    def _update_and_get_bonus(self, state, action, next_state, reward):
        if self.previous_obs is None:
            return 0.0
        #
        self.uncertainty_estimator.update(state,
                                          action,
                                          next_state,
                                          reward)
        return self.bonus(state, action)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        # update uncertainty and compute bonus
        bonus = self._update_and_get_bonus(self.previous_obs,
                                           action,
                                           observation,
                                           reward)
        #
        self.previous_obs = observation

        # add bonus to info
        if info is None:
            info = {}
        else:
            if 'exploration_bonus' in info:
                logger.error("UncertaintyEstimatorWrapper Error: info has" +
                             "  already a key named exploration_bonus!")

        info['exploration_bonus'] = bonus

        return observation, reward, done, info

    def sample(self, state, action):
        logger.warning(
            '[UncertaintyEstimatorWrapper]: sample()'
            + ' method does not consider nor update bonuses.')
        return self.env.sample(state, action)

    def bonus(self, state, action=None):
        bonus = self.bonus_scale_factor * self.uncertainty_estimator.measure(state, action)
        return np.clip(bonus, 0, self.bonus_max)

    def bonus_batch(self, states, actions=None):
        bonus = self.bonus_scale_factor * self.uncertainty_estimator.measure_batch(states, actions)
        return np.clip(bonus, 0, self.bonus_max) if isinstance(bonus, np.ndarray) else torch.clamp(bonus, 0,
                                                                                                   self.bonus_max)
