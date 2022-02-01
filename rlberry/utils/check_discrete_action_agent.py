from rlberry.envs import Chain
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.manager import AgentManager
import numpy as np


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
    action1 = agent1.agent_handlers[0].policy(state)
    action2 = agent2.agent_handlers[0].policy(state)

    return action1 == action2


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

    action1 = agent1.agent_handlers[0].policy(state)
    action2 = agent2.agent_handlers[0].policy(state)

    return np.mean(action1 == action2) == 1
