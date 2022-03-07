import numpy as np
from rlberry.agents.bandits import BanditWithSimplePolicy
from rlberry.wrappers import WriterWrapper
import logging

logger = logging.getLogger(__name__)


class SeqHalvAgent(BanditWithSimplePolicy):
    """
    Sequential Halving Agent
    """

    name = "SeqHalvAgent"

    def __init__(self, env, mean_est_function=None, **kwargs):
        BanditWithSimplePolicy.__init__(self, env, **kwargs)
        if mean_est_function is None:
            mean_est_function = np.mean
        else:
            self.mean_est_function = mean_est_function
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )

    def fit(self, budget=None, **kwargs):
        horizon = budget
        rewards = []
        actions = []
        active_set = np.arange(self.n_arms)

        logk = int(np.ceil(np.log2(self.n_arms)))
        ep = 0

        for r in range(logk):
            tr = np.floor(horizon / (len(active_set) * logk))
            for _ in range(int(tr)):
                for k in active_set:
                    action = k
                    actions += [action]
                    _, reward, _, _ = self.env.step(action)
                    rewards += [reward]
                    ep += 1
            mean_est = [
                self.mean_est_function(np.array(rewards)[actions == k])
                for k in active_set
            ]
            half_len = int(np.ceil(len(active_set) / 2))
            active_set = active_set[np.argsort(mean_est)[-half_len:]]

        self.optimal_action = active_set[0]
        self.writer.add_scalar("optimal_action", self.optimal_action, ep)

        return actions
