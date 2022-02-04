from rlberry.envs import Chain
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.manager import AgentManager
import numpy as np
from rlberry.seeding import set_external_seed


def check_finiteMDP_agent(Agent):
    env_ctor = Chain
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
    for f in range(5):
        # test reproducibility on 5 actions
        set_external_seed(42)  # needed for torch policy
        action1 = agent1.agent_handlers[0].policy(state)
        set_external_seed(42)
        action2 = agent2.agent_handlers[0].policy(state)
        if np.mean(action1 == action2) != 1:
            result = False

    return result


def check_continuous_state_agent(Agent):
    env_ctor = PBall2D
    env_kwargs = {}

    agent1 = AgentManager(
        Agent,
        (env_ctor, env_kwargs),
        fit_budget=2,
        n_fit=1,
        init_kwargs={"horizon": 3},
        seed=42,
    )
    agent2 = AgentManager(
        Agent,
        (env_ctor, env_kwargs),
        fit_budget=2,
        n_fit=1,
        init_kwargs={"horizon": 3},
        seed=42,
    )

    agent1.fit()
    agent2.fit()
    env = env_ctor(**env_kwargs)
    state = env.reset()
    result = True
    for f in range(5):
        # test reproducibility on 5 actions
        set_external_seed(42)  # needed for torch policy
        action1 = agent1.agent_handlers[0].policy(state)
        set_external_seed(42)
        action2 = agent2.agent_handlers[0].policy(state)
        if np.mean(action1 == action2) != 1:
            result = False

    return result
