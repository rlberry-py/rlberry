import numpy as np
import os
from rlberry.envs import GridWorld
from rlberry.agents import IncrementalAgent
from rlberry.stats import AgentStats, plot_writer_data, compare_policies
from rlberry.utils.writers import DefaultWriter


class DummyAgent(IncrementalAgent):

    name = 'DummyAgent'

    def __init__(self, env, n_episodes, hyperparameter=0, **kwargs):
        IncrementalAgent.__init__(self, env, **kwargs)
        self.name = "DummyAgent"
        self.n_episodes = n_episodes
        self.fitted = False
        self.hyperparameter = hyperparameter
        self.fraction_fitted = 0.0
        self.writer = DefaultWriter(name='DummyAgent')

    def fit(self, **kwargs):
        info = {}
        for ii in range(self.n_episodes):
            if self.writer is not None:
                self.writer.add_scalar('episode_reward', 1.0 * ii)
        self.fitted = True
        self.env.reset()
        self.env.step(self.env.action_space.sample())
        return info

    def partial_fit(self, fraction, **kwargs):
        assert fraction > 0.0 and fraction <= 1.0
        self.fraction_fitted = min(1.0, self.fraction_fitted + fraction)
        info = {}
        nn = int(np.ceil(fraction * self.n_episodes))
        for ii in range(nn):
            if self.writer is not None:
                self.writer.add_scalar('episode_reward', 1.0 * ii)
        return info

    def policy(self, observation, time=0, **kwargs):
        return self.env.action_space.sample()

    @classmethod
    def sample_parameters(cls, trial):
        hyperparameter = trial.suggest_categorical('hyperparameter', [1, 2, 3])
        return {'hyperparameter': hyperparameter}


def test_agent_stats_1():
    # Define train and evaluation envs
    train_env = GridWorld()
    eval_env = GridWorld()

    # Parameters
    params = {"n_episodes": 5}
    horizon = 20

    # Check DummyAgent
    agent = DummyAgent(train_env, **params)
    agent.fit()
    agent.policy(None)

    # Run AgentStats
    stats_agent1 = AgentStats(DummyAgent, train_env,
                              init_kwargs=params, n_fit=4, eval_horizon=10, seed=123)
    stats_agent2 = AgentStats(DummyAgent, train_env,
                              init_kwargs=params, n_fit=4, eval_horizon=10, seed=123)
    agent_stats_list = [stats_agent1, stats_agent2]

    # learning curves
    plot_writer_data(agent_stats_list, tag='episode_rewards', show=False)

    # compare final policies
    compare_policies(agent_stats_list, eval_env,
                     eval_horizon=horizon, n_sim=10, show=False)
    compare_policies(agent_stats_list, eval_env,
                     eval_horizon=horizon,
                     n_sim=10, show=False, stationary_policy=False)

    # check if fitted
    for agent_stats in agent_stats_list:
        assert len(agent_stats.fitted_agents) == 4
        for agent in agent_stats.fitted_agents:
            assert agent.fitted

    # test saving/loading
    dirname = stats_agent1.output_dir
    fname = dirname / 'stats'
    stats_agent1.save()
    loaded_stats = AgentStats.load(fname)
    assert stats_agent1.identifier == loaded_stats.identifier

    # delete file
    os.remove(fname.with_suffix('.pickle'))
    dirname.rmdir()

    # test hyperparameter optimization call
    loaded_stats.optimize_hyperparams()
    loaded_stats.optimize_hyperparams(continue_previous=True)


def test_agent_stats_2():
    # Define train and evaluation envs
    train_env = GridWorld()
    eval_env = GridWorld()

    # Parameters
    params = {"n_episodes": 5}

    # Run AgentStats
    stats_agent1 = AgentStats(DummyAgent, train_env, eval_env=eval_env,
                              init_kwargs=params, n_fit=4, eval_horizon=10,
                              n_jobs=1, seed=123)
    stats_agent2 = AgentStats(DummyAgent, train_env, eval_env=eval_env,
                              init_kwargs=params, n_fit=4, eval_horizon=10,
                              n_jobs=1, seed=123)
    agent_stats_list = [stats_agent1, stats_agent2]

    # compare final policies
    compare_policies(agent_stats_list, n_sim=10, show=False)
    compare_policies(agent_stats_list,
                     n_sim=10, show=False, stationary_policy=False)

    # learning curves
    plot_writer_data(agent_stats_list, tag='episode_rewards', show=False)

    # check if fitted
    for agent_stats in agent_stats_list:
        assert len(agent_stats.fitted_agents) == 4
        for agent in agent_stats.fitted_agents:
            assert agent.fitted

    # test saving/loading
    dirname = stats_agent1.output_dir
    fname = dirname / 'stats'
    stats_agent1.save()
    loaded_stats = AgentStats.load(fname)
    assert stats_agent1.identifier == loaded_stats.identifier

    # delete file
    os.remove(fname.with_suffix('.pickle'))
    dirname.rmdir()

    # test hyperparemeter optimization
    loaded_stats.optimize_hyperparams()

    # delete some writers
    stats_agent1.set_writer(1, None)
    stats_agent1.set_writer(2, None)


def test_agent_stats_partial_fit_and_tuple_env():
    # Define train and evaluation envs
    train_env = (GridWorld, None)  # tuple (constructor, kwargs) must also work in AgentStats

    # Parameters
    params = {"n_episodes": 5}
    horizon = 20

    # Run AgentStats
    stats = AgentStats(DummyAgent, train_env,
                       init_kwargs=params, n_fit=4, eval_horizon=10, seed=123)
    stats2 = AgentStats(DummyAgent, train_env,
                        init_kwargs=params, n_fit=4, eval_horizon=10, seed=123)

    # Run partial fit
    stats.partial_fit(0.1)
    stats.partial_fit(0.5)
    for agent in stats.fitted_agents:
        assert agent.fraction_fitted == 0.6
    for _ in range(2):
        stats.partial_fit(0.5)
        for agent in stats.fitted_agents:
            assert agent.fraction_fitted == 1.0

    # Run fit
    stats2.fit()

    # learning curves
    plot_writer_data([stats], tag='episode_rewards', show=False, preprocess_func=np.cumsum)

    # compare final policies
    compare_policies([stats],
                     eval_horizon=horizon, n_sim=10, show=False)

    # delete some writers
    stats.set_writer(0, None)
    stats.set_writer(3, None)
