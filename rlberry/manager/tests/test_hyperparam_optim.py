from rlberry.envs import GridWorld
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.dynprog.value_iteration import ValueIterationAgent
from rlberry.manager import AgentManager
from optuna.samplers import TPESampler


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

        self.fraction_fitted = 0.0

    def fit(self, budget, **kwargs):
        del kwargs
        self.fitted = True
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


def test_hyperparam_optim_tpe():
    # Define trainenv
    train_env = (GridWorld, {})

    # Run AgentManager
    stats_agent = AgentManager(DummyAgent,
                             train_env,
                             fit_budget=1,
                             init_kwargs={},
                             eval_kwargs={'eval_horizon': 5},
                             n_fit=4)

    # test hyperparameter optimization with TPE sampler
    # using hyperopt default values
    sampler_kwargs = TPESampler.hyperopt_parameters()
    stats_agent.optimize_hyperparams(sampler_kwargs=sampler_kwargs, n_trials=5)
    stats_agent.clear_output_dir()


def test_hyperparam_optim_random():
    # Define train env
    train_env = (GridWorld, {})

    # Run AgentManager
    stats_agent = AgentManager(DummyAgent,
                             train_env,
                             init_kwargs={},
                             fit_budget=1,
                             eval_kwargs={'eval_horizon': 5},
                             n_fit=4)

    # test hyperparameter optimization with random sampler
    stats_agent.optimize_hyperparams(sampler_method="random", n_trials=5)
    stats_agent.clear_output_dir()


def test_hyperparam_optim_grid():
    # Define train env
    train_env = (GridWorld, {})

    # Run AgentManager
    stats_agent = AgentManager(DummyAgent,
                             train_env,
                             init_kwargs={},
                             fit_budget=1,
                             eval_kwargs={'eval_horizon': 5},
                             n_fit=4)

    # test hyperparameter optimization with grid sampler
    search_space = {"hyperparameter1": [1, 2, 3],
                    "hyperparameter2": [-5, 0, 5]}
    sampler_kwargs = {"search_space": search_space}
    stats_agent.optimize_hyperparams(n_trials=3 * 3,
                                     sampler_method="grid",
                                     sampler_kwargs=sampler_kwargs)
    stats_agent.clear_output_dir()


def test_hyperparam_optim_cmaes():
    # Define train env
    train_env = (GridWorld, {})

    # Run AgentManager
    stats_agent = AgentManager(DummyAgent,
                             train_env,
                             init_kwargs={},
                             fit_budget=1,
                             eval_kwargs={'eval_horizon': 5},
                             n_fit=4)

    # test hyperparameter optimization with CMA-ES sampler
    stats_agent.optimize_hyperparams(sampler_method="cmaes", n_trials=5)
    stats_agent.clear_output_dir()


def test_discount_optimization():
    class ValueIterationAgentToOptimize(ValueIterationAgent):
        @classmethod
        def sample_parameters(cls, trial):
            """
            Sample hyperparameters for hyperparam optimization using Optuna (https://optuna.org/)
            """
            gamma = trial.suggest_categorical('gamma', [0.1, 0.99])
            return {'gamma': gamma}

    env = (GridWorld, dict(
        nrows=3, ncols=10,
        reward_at={(1, 1): 0.1, (2, 9): 1.0},
        walls=((1, 4), (2, 4), (1, 5)),
        success_probability=0.9))

    vi_params = {'gamma': 0.1, 'epsilon': 1e-3}

    vi_stats = AgentManager(ValueIterationAgentToOptimize,
                          env,
                          fit_budget=0,
                          eval_kwargs=dict(eval_horizon=20),
                          init_kwargs=vi_params,
                          n_fit=4,
                          seed=123)

    vi_stats.optimize_hyperparams(n_trials=5, n_fit=1,
                                  sampler_method='random', pruner_method='none')

    assert vi_stats.optuna_study
    vi_stats.clear_output_dir()
