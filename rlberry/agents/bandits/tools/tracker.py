import logging
from rlberry import metadata_utils
from rlberry.utils.writers import DefaultWriter

logger = logging.getLogger(__name__)


class BanditTracker(DefaultWriter):
    """
    Container class for rewards and various statistics (means...) collected
    during the run of a bandit algorithm.

    Used mainly to compute indices for bandits algorithms. A default tracker
    is automatically constructed in :class:`~rlberry.agents.bandits.BanditWithSimplePolicy`
    and can then be used as entry for an index function. See the example.

    Parameters
    ----------
    agent: rlberry bandit agent
        See :class:`~rlberry.agents.bandits`.

    params: dict
        Other parameters to condition what to store and compute.
        In particuler if params contains store_rewards=True, the
        rewards will be saved for each arm at each step.

    Examples
    --------
    >>>  def index(tr):
         ''' Compute UCB index for rewards in [0,1]'''
         return [
            tr.read_last_tag_value("mu_hat", arm)
            +  np.sqrt( 0.5 * np.log(1 / delta(tr.read_last_tag_value("t")))
                / tr.read_last_tag_value("n_pulls", arm)
            )
            for arm in tr.arms
          ]

    """

    name = "BanditTracker"

    def __init__(self, agent, params={}):
        self.n_arms = agent.n_arms
        self.arms = agent.arms
        self.rng = agent.rng

        # Store rewards for each arm or not
        self.store_rewards = params.get("store_rewards", False)
        # Add importance weighted rewards or not
        self.do_iwr = params.get("do_iwr", False)

        # By default, store a single attribute (the most recent)
        maxlen = 1
        # To store all rewards, override the maxlen for the corresponding tags
        maxlen_by_tag = dict()
        if self.store_rewards:
            for arm in self.arms:
                maxlen_by_tag[str(arm) + "_reward"] = None

        _tracker_kwargs = dict(
            name="BanditTracker",
            execution_metadata=metadata_utils.ExecutionMetadata(),
            maxlen=maxlen,
            maxlen_by_tag=maxlen_by_tag,
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
