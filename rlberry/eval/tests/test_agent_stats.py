import numpy as np
import pytest
import rlberry.seeding as seeding
from rlberry.envs import GridWorld
from rlberry.agents import Agent
from rlberry.eval.agent_stats import AgentStats, plot_episode_rewards, compare_policies


# global seed
seeding.set_global_seed(1234)


class DummyAgent(Agent):
    fit_info = ("episode_rewards",)

    def __init__(self, env, n_episodes, **kwargs):
        Agent.__init__(self, env, **kwargs)
        self.name = "DummyAgent"
        self.n_episodes = n_episodes
        self.fitted = False

    def fit(self, **kwargs):
        info = {}
        info["episode_rewards"] = np.arange(self.n_episodes)
        self.fitted = True
        return info 
    
    def policy(self, observation, time=0, **kwargs):
        return self.env.action_space.sample()



def test_agent_stats():

    # Define train and evaluation envs
    train_env = GridWorld()
    eval_env  = GridWorld()

    # Parameters
    params = {"n_episodes": 500, "horizon":20}

    # Check DummyAgent
    agent = DummyAgent(train_env, **params)
    agent.fit()
    agent.policy(None)

    # Run AgentStats
    stats_agent1 = AgentStats(DummyAgent, train_env, init_kwargs=params, nfit=4) # fit 4 agents
    stats_agent2 = AgentStats(DummyAgent, train_env, init_kwargs=params, nfit=4)
    agent_stats_list = [stats_agent1, stats_agent2]

    # learning curves
    plot_episode_rewards(agent_stats_list, cumulative=True, show=False)

    # compare final policies
    compare_policies(agent_stats_list, eval_env, eval_horizon=params["horizon"], nsim=10, show=False)
    compare_policies(agent_stats_list, eval_env, eval_horizon=params["horizon"], nsim=10, show=False, stationary_policy=False)

    # check if fitted 
    for agent_stats in agent_stats_list:
        assert len(agent_stats.fitted_agents) == 4
        for agent in agent_stats.fitted_agents:
            assert agent.fitted 