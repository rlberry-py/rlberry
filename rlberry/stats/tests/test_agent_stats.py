import numpy as np
import os
import rlberry.seeding as seeding
from rlberry.envs import GridWorld
from rlberry.agents import IncrementalAgent
from rlberry.stats import AgentStats, plot_episode_rewards, compare_policies


# global seed
seeding.set_global_seed(1234)


class DummyAgent(IncrementalAgent):
    fit_info = ("episode_rewards",)

    def __init__(self, env, n_episodes, hyperparameter=0, **kwargs):
        IncrementalAgent.__init__(self, env, **kwargs)
        self.name = "DummyAgent"
        self.n_episodes = n_episodes
        self.fitted = False
        self.hyperparameter = hyperparameter

        self.fraction_fitted = 0.0

    def fit(self, **kwargs):
        info = {}
        info["episode_rewards"] = np.arange(self.n_episodes)
        self.fitted = True
        return info

    def partial_fit(self, fraction, **kwargs):
        assert fraction > 0.0 and fraction <= 1.0
        self.fraction_fitted = min(1.0, self.fraction_fitted + fraction)
        info = {}
        nn = int(np.ceil(fraction*self.n_episodes))
        info["episode_rewards"] = np.arange(nn)
        return info

    def policy(self, observation, time=0, **kwargs):
        return self.env.action_space.sample()

    @classmethod
    def sample_parameters(cls, trial):
        hyperparameter = trial.suggest_categorical('hyperparameter', [1, 2, 3])
        return {'hyperparameter': hyperparameter}


def test_agent_stats():

    # Define train and evaluation envs
    train_env = GridWorld()
    eval_env = GridWorld()

    # Parameters
    params = {"n_episodes": 500, "horizon": 20}

    # Check DummyAgent
    agent = DummyAgent(train_env, **params)
    agent.fit()
    agent.policy(None)

    # Run AgentStats
    stats_agent1 = AgentStats(DummyAgent, train_env,
                              init_kwargs=params, n_fit=4, eval_horizon=10)
    stats_agent2 = AgentStats(DummyAgent, train_env,
                              init_kwargs=params, n_fit=4, eval_horizon=10)
    agent_stats_list = [stats_agent1, stats_agent2]

    # learning curves
    plot_episode_rewards(agent_stats_list, cumulative=True, show=False)

    # compare final policies
    compare_policies(agent_stats_list, eval_env,
                     eval_horizon=params["horizon"], n_sim=10, show=False)
    compare_policies(agent_stats_list, eval_env,
                     eval_horizon=params["horizon"],
                     n_sim=10, show=False, stationary_policy=False)

    # check if fitted
    for agent_stats in agent_stats_list:
        assert len(agent_stats.fitted_agents) == 4
        for agent in agent_stats.fitted_agents:
            assert agent.fitted

    # test saving/loading
    stats_agent1.save('test_agent_stats_file.pickle')
    loaded_stats = AgentStats.load('test_agent_stats_file.pickle')
    assert stats_agent1.identifier == loaded_stats.identifier

    # delete file
    os.remove('test_agent_stats_file.pickle')

    # test hyperparemeter optimization
    loaded_stats.optimize_hyperparams()


def test_agent_stats_partial_fit():
    # Define train and evaluation envs
    train_env = GridWorld()
    eval_env = GridWorld()

    # Parameters
    params = {"n_episodes": 500, "horizon": 20}

    # Check DummyAgent
    agent = DummyAgent(train_env, **params)
    agent.fit()
    agent.policy(None)

    # Run AgentStats
    stats = AgentStats(DummyAgent, train_env,
                       init_kwargs=params, n_fit=4, eval_horizon=10)

    # Run partial fit
    stats.partial_fit(0.1)
    stats.partial_fit(0.5)
    for agent in stats.fitted_agents:
        assert agent.fraction_fitted == 0.6
    # for _ in range(2):
    #     stats.partial_fit(0.5)
    #     for agent in stats.fitted_agents:
    #         assert agent.fraction_fitted == 1.0

    # learning curves
    plot_episode_rewards([stats], cumulative=True, show=False)

    # compare final policies
    compare_policies([stats], eval_env,
                     eval_horizon=params["horizon"], n_sim=10, show=False)
