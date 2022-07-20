from rlberry import metadata_utils
from rlberry.utils.writers import DefaultWriter

import rlberry

logger = rlberry.logger


class BanditTracker(DefaultWriter):
    """
    Container class for rewards and various statistics (means...) collected
    during the run of a bandit algorithm.

    BanditTracker is a companion class for
    :class:`~rlberry.agents.bandits.BanditWithSimplePolicy` (and other agents
    based on it), where a default tracker is automatically constructed, and can
    then be used e.g as an entry for an index function.

    It inherits the logic of DefaultWriter to write/store/read
    various data of interest for the execution of a bandit agent.

    Data are stored in the data attribute and indexed by a specific tag.
    Except for the tag "t" (corresponding to the running total number of time
    steps played by the agent), all tags are arm-specific (n_pulls,
    total_reward...). Each tag entry is stored as a deque with fixed maximum
    length (FIFO). By default, this maximum length is set to 1, i.e each new
    update to the tag erases the previously stored entry. The maximum length
    can be changed on a tag-by-tag basis with the dict maxlen_by_tag.

    Data can be interacted with by using the following DefaultWriter accessors:
        * Read:
            * read_last_tag_value(tag, arm)): returns the last entry of the
            deque corresponding to arm-specific tag.
            * read_tag_value(tag, arm)): returns the full deque corresponding
            to the arm-specific tag.
        * Write:
            * add_scalar(tag, value): add a single scalar value to the deque
            corresponding to the tag.
            * add_scalars(arm, {tags: values}): add multiple arm-specific
            tagged values to each corresponding deque.

    For ease of use, wrapper methods are provided to access common tag such as
    t, n_pulls, total_reward... without explicitly calling the
    read_last_tag_value/read_tag_value methods.

    Parameters
    ----------
    agent: rlberry bandit agent
        See :class:`~rlberry.agents.bandits`.

    params: dict
        Other parameters to condition what to store and compute.
        In particuler if params contains store_rewards=True, the
        rewards will be saved for each arm at each step and if
        store_actions=True, the actions are saved.
        It can also contain a function named "update" that will
        be called at the end of the update phase. def update(tr, arm): ...


    Examples
    --------
    >>>  def index(tr):
         ''' Compute UCB index for rewards in [0,1]'''
         return [
            tr.mu_hat(arm) +  np.sqrt(
                0.5 * np.log(1 / delta(tr.t))) / tr.n_pulls(arm)
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
        # Store the actions for each arm or not
        self.store_actions = params.get("store_actions", False)
        # Additional update function
        self.additional_update = params.get("update", None)

        # Add importance weighted rewards or not
        self.do_iwr = params.get("do_iwr", False)

        # By default, store a single attribute (the most recent)
        maxlen = 1
        # To store all rewards, override the maxlen for the corresponding tags
        maxlen_by_tag = dict()
        if self.store_rewards:
            for arm in self.arms:
                maxlen_by_tag[str(arm) + "_reward"] = None
        if self.store_actions:
            maxlen_by_tag["action"] = None

        _tracker_kwargs = dict(
            name="BanditTracker",
            execution_metadata=metadata_utils.ExecutionMetadata(),
            maxlen=maxlen,
            maxlen_by_tag=maxlen_by_tag,
        )
        DefaultWriter.__init__(self, print_log=False, **_tracker_kwargs)

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

    @property
    def t(self):
        """
        Current running time of the bandit algorithm played by the associated
        bandit agent.
        """
        return self.read_last_tag_value("t")

    def n_pulls(self, arm):
        """
        Current number of pulls by the associated bandit agent to a given arm.
        """
        return self.read_last_tag_value("n_pulls", arm)

    def rewards(self, arm):
        """
        All rewards collected so far by the associated bandit agent for a given
        arm and currently stored. If maxlen_by_tag[str(arm) + "_reward"] is None
        or maxlen is None, all the reward history is stored at anytime.
        """
        return self.read_tag_value("reward", arm)

    def reward(self, arm):
        """
        Last collected reward for a given arm.
        """
        return self.read_last_tag_value("reward", arm)

    def actions(self, arm):
        """
        All actions collected so far by the associated bandit agent for a given
        arm and currently stored. If maxlen_by_tag["action"] is None
        or maxlen is None, all the action history is stored at anytime.
        """
        return self.read_tag_value("action")

    def action(self, arm):
        """
        Last collected action for a given arm.
        """
        return self.read_last_tag_value("action")

    def total_reward(self, arm):
        """
        Current total reward collected so far by the associated bandit agent
        for a given arm.
        """
        return self.read_last_tag_value("total_reward", arm)

    def mu_hat(self, arm):
        """
        Current empirical mean reward for a given arm estimated by the
        associated bandit agent.
        """
        return self.read_last_tag_value("mu_hat", arm)

    def iw_total_reward(self, arm):
        """
        Empirical Importance weighted total reward collected so far by the
        associated bandit agent for a given arm. Used by randomized algorithms.
        The IW total reward is the sum of rewards for a given arm inversely
        weighted by the arm sampling probabilities at each pull.
        In this implementation, we update the loss-based estimator, i.e for
        a reward r in [0, 1], we weight 1 - r instead of r
        (see Note 9, Chapter 11 of [1]).

        .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
                Cambridge University Press, 2020.
        """
        return self.read_last_tag_value("iw_total_reward", arm)

    def update(self, arm, reward, params={}):
        """
        After the associated bandit agent played a given arm and collected a
        given reward, update the stored data.
        By default, only standard statistics are calculated and stored (number
        of pulls, current reward, total reward and current empirical mean
        reward). Special parameters can be passed in params, e.g the sampling
        probability for randomized algorithms (to update the importance
        weighted total reward).
        """
        # Update current running time
        self.add_scalar("t", self.t + 1)

        # Total number of pulls for current arm
        n_pulls_arm = self.n_pulls(arm) + 1
        # Sum of rewards for current arm
        total_reward_arm = self.total_reward(arm) + reward

        tag_scalar_dict = {
            "n_pulls": n_pulls_arm,
            "reward": reward,
            "total_reward": total_reward_arm,
            "mu_hat": total_reward_arm / n_pulls_arm,
        }

        # Importance weighted total rewards for randomized algorithns
        if self.do_iwr:
            p = params.get("p", 1.0)
            iw_total_reward_arm = self.iw_total_reward(arm)
            tag_scalar_dict["iw_total_reward"] = (
                iw_total_reward_arm + 1 - (1 - reward) / p
            )

        # Write all tracked statistics
        self.add_scalars(arm, tag_scalar_dict)
        self.add_scalar("action", arm)

        # Do the additional update
        if self.additional_update is not None:
            self.additional_update(self, arm)
