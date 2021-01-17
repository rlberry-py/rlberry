import numpy as np
import rlberry.seeding as seeding
from rlberry.envs import GridWorld
from rlberry.agents import IncrementalAgent
from rlberry.agents.dynprog.value_iteration import ValueIterationAgent
from rlberry.stats import AgentStats
from optuna.samplers import TPESampler


# global seed
seeding.set_global_seed(1234)


class DummyAgent(IncrementalAgent):
    def __init__(self,
                 env,
                 n_episodes,
                 hyperparameter1=0,
                 hyperparameter2=0,
                 **kwargs):
        IncrementalAgent.__init__(self, env, **kwargs)
        self.name = "DummyAgent"
        self.n_episodes = n_episodes
        self.fitted = False
        self.hyperparameter1 = hyperparameter1
        self.hyperparameter2 = hyperparameter2

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
        hyperparameter1 \
            = trial.suggest_categorical('hyperparameter1', [1, 2, 3])
        hyperparameter2 \
            = trial.suggest_uniform('hyperparameter2', -10, 10)
        return {'hyperparameter1': hyperparameter1,
                'hyperparameter2': hyperparameter2}


def test_hyperparam_optim_tpe():
    # Define trainenv
    train_env = GridWorld()

    # Parameters
    params = {"n_episodes": 500}

    # Run AgentStats
    stats_agent = AgentStats(DummyAgent, train_env, init_kwargs=params,
                             n_fit=4, eval_horizon=10, n_jobs=1)

    # test hyperparameter optimization with TPE sampler
    # using hyperopt default values
    sampler_kwargs = TPESampler.hyperopt_parameters()
    stats_agent.optimize_hyperparams(sampler_kwargs=sampler_kwargs)


def test_hyperparam_optim_random():
    # Define train env
    train_env = GridWorld()

    # Parameters
    params = {"n_episodes": 500}

    # Run AgentStats
    stats_agent = AgentStats(DummyAgent, train_env, init_kwargs=params,
                             n_fit=4, eval_horizon=10, n_jobs=1)

    # test hyperparameter optimization with random sampler
    stats_agent.optimize_hyperparams(sampler_method="random")


def test_hyperparam_optim_grid():
    # Define train env
    train_env = GridWorld()

    # Parameters
    params = {"n_episodes": 500}

    # Run AgentStats
    stats_agent = AgentStats(DummyAgent, train_env, init_kwargs=params,
                             n_fit=4, eval_horizon=10, n_jobs=1)

    # test hyperparameter optimization with grid sampler
    search_space = {"hyperparameter1": [1, 2, 3],
                    "hyperparameter2": [-5, 0, 5]}
    sampler_kwargs = {"search_space": search_space}
    stats_agent.optimize_hyperparams(n_trials=3*3,
                                     sampler_method="grid",
                                     sampler_kwargs=sampler_kwargs)


def test_hyperparam_optim_cmaes():
    # Define train env
    train_env = GridWorld()

    # Parameters
    params = {"n_episodes": 500}

    # Run AgentStats
    stats_agent = AgentStats(DummyAgent, train_env, init_kwargs=params,
                             n_fit=4, eval_horizon=10, n_jobs=1)

    # test hyperparameter optimization with CMA-ES sampler
    stats_agent.optimize_hyperparams(sampler_method="cmaes")


def test_discount_optimization():
    seeding.set_global_seed(42)

    class ValueIterationAgentToOptimize(ValueIterationAgent):
        @classmethod
        def sample_parameters(cls, trial):
            """
            Sample hyperparameters for hyperparam optimization using Optuna (https://optuna.org/)
            """
            gamma = trial.suggest_categorical('gamma', [0.1, 0.99])
            return {'gamma': gamma}

    env = GridWorld(nrows=3, ncols=10,
                    reward_at={(1, 1): 0.1, (2, 9): 1.0},
                    walls=((1, 4), (2, 4), (1, 5)),
                    success_probability=0.9)

    vi_params = {'gamma': 0.1, 'epsilon': 1e-3}

    vi_stats = AgentStats(ValueIterationAgentToOptimize, env, eval_horizon=20, init_kwargs=vi_params, n_fit=4, n_jobs=1)

    vi_stats.optimize_hyperparams(n_trials=5, timeout=30, n_sim=5, n_fit=1, n_jobs=1,
                                  sampler_method='random', pruner_method='none')

    assert vi_stats.best_hyperparams['gamma'] == 0.99
