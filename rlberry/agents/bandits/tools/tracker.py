import logging
from rlberry import metadata_utils
from rlberry.utils.writers import DefaultWriter

logger = logging.getLogger(__name__)


class BanditTracker(DefaultWriter):
    """
    Container class for rewards and various statistics (means...) collected
    during the run of a bandit algorithm.

    Parameters
    ----------
    agent: rlberry bandit agent
        See :class:`~rlberry.agents.bandits`.

    params: dict
        Other parameters to condition what to store and compute.
    """

    name = "BanditTracker"

    def __init__(self, agent, params={}):
        self.n_arms = agent.n_arms
        self.arms = agent.arms
        self.seeder = agent.seeder

        # Store rewards for each arm or not
        self.store_rewards = params.get("store_rewards", False)
        # Add importance weighted rewards or not
        self.do_iwr = params.get("do_iwr", False)

        maxlen = None if self.store_rewards else 1
        _tracker_kwargs = dict(
            name="BanditTracker",
            execution_metadata=metadata_utils.ExecutionMetadata(),
            maxlen=maxlen,
        )
        DefaultWriter.__init__(self, **_tracker_kwargs)

        self.reset_tracker()

    def reset_tracker(self):
        self.add_scalar("t", 0)

        tag_scalar_dict = dict()
        for arm in self.arms:
            tag_scalar_dict["n_pulls"] = 0
            tag_scalar_dict["total_reward"] = 0.0
            if self.do_iwr:
                tag_scalar_dict["iw_total_reward"] = 0.0

            self.add_scalars(arm, tag_scalar_dict)

    def update(self, arm, reward, params={}):
        self.add_scalar("t", self.read_last_tag_value("t") + 1)

        # Total number of pulls for current arm
        n_pulls_arm = self.read_last_tag_value("n_pulls", arm) + 1
        # Sum of rewards for current arm
        total_reward_arm = self.read_last_tag_value("total_reward", arm) + reward

        tag_scalar_dict = {
            "n_pulls": n_pulls_arm,
            "reward": reward,
            "total_reward": total_reward_arm,
            "mu_hat": total_reward_arm / n_pulls_arm,
        }
        if self.do_iwr:
            p = params.get("p", 1.0)
            iw_total_reward_arm = self.read_last_tag_value("iw_total_reward", arm)
            tag_scalar_dict["iw_total_reward"] = (
                iw_total_reward_arm + 1 - (1 - reward) / p
            )

        self.add_scalars(arm, tag_scalar_dict)
