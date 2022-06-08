from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.envs.finite import GridWorld
from rlberry.manager import AgentManager, evaluate_agents
import numpy as np


def test_vi_gridworld():
    env = GridWorld(7, 10, walls=((2, 2), (3, 3)))
    agent = ValueIterationAgent(env, gamma=0.95)
    rbagent = AgentManager(
        ValueIterationAgent,
        (GridWorld, {"nrows": 7, "ncols": 10, "walls": ((2, 2), (3, 3))}),
        fit_budget=1e3,
        init_kwargs=dict(gamma=0.95),
        seed=42,
        eval_kwargs=dict(eval_horizon=100, n_simulations=20),
    )

    rbagent.fit()
    assert np.median(evaluate_agents([rbagent], plot=False)) > 83
