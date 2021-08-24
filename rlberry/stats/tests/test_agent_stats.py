import numpy as np
from rlberry.envs import GridWorld
from rlberry.agents import AgentWithSimplePolicy
from rlberry.stats import AgentStats, plot_writer_data, evaluate_agents
from rlberry.utils.writers import DefaultWriter


class DummyAgent(AgentWithSimplePolicy):
    def __init__(self,
                 env,
                 hyperparameter1=0,
                 hyperparameter2=0,
                 **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.name = "DummyAgent"
        self.fitted = False
        self.hyperparameter1 = hyperparameter1
        self.hyperparameter2 = hyperparameter2

        self.total_budget = 0.0
        self.writer = DefaultWriter(self.name)

    def fit(self, budget):
        self.fitted = True
        self.total_budget += budget
        for ii in range(budget):
            if self.writer is not None:
                self.writer.add_scalar('a', 5)
                self.writer.add_scalar('b', 6, ii)
        return None

    def policy(self, observation):
        return 0

    @classmethod
    def sample_parameters(cls, trial):
        hyperparameter1 \
            = trial.suggest_categorical('hyperparameter1', [1, 2, 3])
        hyperparameter2 \
            = trial.suggest_uniform('hyperparameter2', -10, 10)
        return {'hyperparameter1': hyperparameter1,
                'hyperparameter2': hyperparameter2}


def test_agent_stats_1():
    # Define train and evaluation envs
    train_env = (GridWorld, {})

    # Parameters
    params = {}
    eval_kwargs = dict(eval_horizon=10)

    # Check DummyAgent
    agent = DummyAgent(train_env[0](**train_env[1]), **params)
    agent.fit(10)
    agent.policy(None)

    # Run AgentStats
    stats_agent1 = AgentStats(DummyAgent, train_env, fit_budget=5, eval_kwargs=eval_kwargs,
                              init_kwargs=params, n_fit=4, seed=123)
    stats_agent2 = AgentStats(DummyAgent, train_env, fit_budget=5, eval_kwargs=eval_kwargs,
                              init_kwargs=params, n_fit=4, seed=123)
    agent_stats_list = [stats_agent1, stats_agent2]
    for st in agent_stats_list:
        st.fit()

    # learning curves
    plot_writer_data(agent_stats_list, tag='episode_rewards', show=False)

    # compare final policies
    evaluate_agents(agent_stats_list, show=False)

    # check if fitted
    for agent_stats in agent_stats_list:
        assert len(agent_stats.agent_handlers) == 4
        for agent in agent_stats.agent_handlers:
            assert agent.fitted

    # test saving/loading
    dirname = stats_agent1.output_dir
    fname = dirname / 'stats'
    stats_agent1.save()
    loaded_stats = AgentStats.load(fname)
    assert stats_agent1.identifier == loaded_stats.identifier

    # test hyperparameter optimization call
    loaded_stats.optimize_hyperparams(n_trials=5)
    loaded_stats.optimize_hyperparams(n_trials=5, continue_previous=True)

    for st in agent_stats_list:
        st.clear_output_dir()


def test_agent_stats_2():
    # Define train and evaluation envs
    train_env = (GridWorld, {})
    eval_env = (GridWorld, {})

    # Parameters
    params = {}
    eval_kwargs = dict(eval_horizon=10)

    # Run AgentStats
    stats_agent1 = AgentStats(DummyAgent, train_env, eval_env=eval_env,
                              fit_budget=5, eval_kwargs=eval_kwargs,
                              init_kwargs=params, n_fit=4,
                              n_jobs=1, seed=123)
    stats_agent2 = AgentStats(DummyAgent, train_env, eval_env=eval_env,
                              fit_budget=5, eval_kwargs=eval_kwargs,
                              init_kwargs=params, n_fit=4,
                              n_jobs=1, seed=123)
    agent_stats_list = [stats_agent1, stats_agent2]
    for st in agent_stats_list:
        st.fit()

    # compare final policies
    evaluate_agents(agent_stats_list, show=False)
    evaluate_agents(agent_stats_list, show=False)

    # learning curves
    plot_writer_data(agent_stats_list, tag='episode_rewards', show=False)

    # check if fitted
    for agent_stats in agent_stats_list:
        assert len(agent_stats.agent_handlers) == 4
        for agent in agent_stats.agent_handlers:
            assert agent.fitted

    # test saving/loading
    dirname = stats_agent1.output_dir
    fname = dirname / 'stats'
    stats_agent1.save()
    loaded_stats = AgentStats.load(fname)
    assert stats_agent1.identifier == loaded_stats.identifier

    # test hyperparemeter optimization
    loaded_stats.optimize_hyperparams(n_trials=5)

    # delete some writers
    stats_agent1.set_writer(1, None)
    stats_agent1.set_writer(2, None)

    stats_agent1.clear_output_dir()
    stats_agent2.clear_output_dir()


def test_agent_stats_partial_fit_and_tuple_env():
    # Define train and evaluation envs
    train_env = (GridWorld, None)  # tuple (constructor, kwargs) must also work in AgentStats

    # Parameters
    params = {}
    eval_kwargs = dict(eval_horizon=10)

    # Run AgentStats
    stats = AgentStats(DummyAgent, train_env,
                       init_kwargs=params, n_fit=4,
                       fit_budget=5, eval_kwargs=eval_kwargs,
                       seed=123)
    stats2 = AgentStats(DummyAgent, train_env,
                        init_kwargs=params, n_fit=4,
                        fit_budget=5, eval_kwargs=eval_kwargs,
                        seed=123)

    # Run partial fit
    stats.fit(10)
    stats.fit(20)
    for agent in stats.agent_handlers:
        assert agent.total_budget == 30

    # Run fit
    stats2.fit()

    # learning curves
    plot_writer_data([stats], tag='episode_rewards', show=False, preprocess_func=np.cumsum)

    # compare final policies
    evaluate_agents([stats], show=False)

    # delete some writers
    stats.set_writer(0, None)
    stats.set_writer(3, None)

    stats.clear_output_dir()
    stats2.clear_output_dir()
