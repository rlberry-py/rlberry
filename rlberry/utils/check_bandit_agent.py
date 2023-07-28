from rlberry.envs.bandits import BernoulliBandit
from rlberry.manager import ExperimentManager


def check_bandit_agent(Agent, environment=BernoulliBandit, seed=42):
    """
    Function used to check a bandit agent in rlberry on a Gaussian bandit problem.

    Parameters
    ----------
    Agent: rlberry agent module
        Agent class that we want to test.

    environment: rlberry env module
        Environment (i.e bandit instance) on which to test the agent.

    seed : Seed sequence from which to spawn the random number generator.


    Returns
    -------
    result : bool
        Whether the agent is a valid/compatible bandit agent.

    Examples
    --------
    >>> from rlberry.agents.bandits import IndexAgent
    >>> from rlberry.utils import check_bandit_agent
    >>> import numpy as np
    >>> class UCBAgent(IndexAgent):
    >>>     name = "UCB"
    >>>     def __init__(self, env, **kwargs):
    >>>         def index(r, t):
    >>>             return np.mean(r) + np.sqrt(np.log(t**2) / (2 * len(r)))
    >>>         IndexAgent.__init__(self, env, index, **kwargs)
    >>> check_bandit_agent(UCBAgent)
    True

    """
    env_ctor = environment
    env_kwargs = {}

    xp_manager1 = ExperimentManager(
        Agent, (env_ctor, env_kwargs), fit_budget=10, n_fit=1, seed=seed
    )
    xp_manager2 = ExperimentManager(
        Agent, (env_ctor, env_kwargs), fit_budget=10, n_fit=1, seed=seed
    )

    xp_manager1.fit()
    xp_manager2.fit()
    env = env_ctor(**env_kwargs)
    state, info = env.reset()
    result = True
    for _ in range(5):
        # test reproducibility on 5 actions
        action1 = xp_manager1.agent_handlers[0].policy(state)
        action2 = xp_manager2.agent_handlers[0].policy(state)
        if action1 != action2:
            result = False

    return result
