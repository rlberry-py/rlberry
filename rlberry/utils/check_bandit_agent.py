from rlberry.envs.bandits import NormalBandit
from rlberry.manager import AgentManager


def check_bandit_agent(Agent):
    """
    Function used to check a bandit agent in rlberry.

    Parameter
    ---------

    Agent: rlberry agent module
        Agent class that we want to test.

    """
    env_ctor = NormalBandit
    env_kwargs = {}

    agent1 = AgentManager(
        Agent, (env_ctor, env_kwargs), fit_budget=10, n_fit=1, seed=42
    )
    agent2 = AgentManager(
        Agent, (env_ctor, env_kwargs), fit_budget=10, n_fit=1, seed=42
    )

    agent1.fit()
    agent2.fit()
    env = env_ctor(**env_kwargs)
    state = env.reset()
    result = True
    for _ in range(5):
        # test reproducibility on 5 actions
        action1 = agent1.agent_handlers[0].policy(state)
        action2 = agent2.agent_handlers[0].policy(state)
        if action1 != action2:
            result = False

    return result
