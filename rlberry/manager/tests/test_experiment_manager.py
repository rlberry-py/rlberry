import shutil
import tempfile
import pytest
import numpy as np
import sys
import os
from rlberry_research.envs import GridWorld
from rlberry.agents import AgentWithSimplePolicy, AgentTorch, Agent
from rlberry.manager import (
    ExperimentManager,
    plot_writer_data,
    evaluate_agents,
    preset_manager,
    read_writer_data,
)
from rlberry.wrappers import WriterWrapper
from rlberry.utils.check_agent import check_save_load


class DummyAgent(AgentWithSimplePolicy):
    def __init__(self, env, hyperparameter1=0, hyperparameter2=0, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.name = "DummyAgent"
        self.fitted = False
        self.hyperparameter1 = hyperparameter1
        self.hyperparameter2 = hyperparameter2

        self.total_budget = 0.0

    def fit(self, budget, **kwargs):
        del kwargs
        self.fitted = True
        self.total_budget += budget
        for ii in range(budget):
            if self.writer is not None:
                self.writer.add_scalar("a", 5)
                self.writer.add_scalar("b", 6, ii)
        return None

    def policy(self, observation):
        return 0

    @classmethod
    def sample_parameters(cls, trial):
        hyperparameter1 = trial.suggest_categorical("hyperparameter1", [1, 2, 3])
        hyperparameter2 = trial.suggest_float("hyperparameter2", -10, 10)
        return {"hyperparameter1": hyperparameter1, "hyperparameter2": hyperparameter2}


class DummyAgent2(Agent):
    def __init__(self, env, hyperparameter1=0, hyperparameter2=0, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.name = "DummyAgent"
        self.fitted = False
        self.hyperparameter1 = hyperparameter1
        self.hyperparameter2 = hyperparameter2

        self.total_budget = 0.0

    def fit(self, budget, **kwargs):
        del kwargs
        self.fitted = True
        self.total_budget += budget
        for ii in range(budget):
            if self.writer is not None:
                self.writer.add_scalar("a", 5)
                self.writer.add_scalar("b", 6, ii)
        return None

    def policy(self, observation):
        return 0

    # 3 - function to evaluate a model
    def eval(self, n_simulations: int = 2, eval_horizon=3):
        obs = self.eval_env.reset()
        action = self.policy(obs)
        episode_returns = [
            0,
        ] * n_simulations
        return np.mean(episode_returns)


@pytest.mark.xfail(sys.platform == "win32", reason="bug with windows???")
def test_experiment_manager_1():
    # Define train and evaluation envs
    train_env = (GridWorld, {})

    # Parameters
    params = dict(hyperparameter1=-1, hyperparameter2=100)
    eval_kwargs = dict(eval_horizon=10)

    # Check DummyAgent
    agent = DummyAgent(train_env[0](**train_env[1]), **params)
    agent.fit(10)
    agent.policy(None)

    # Run ExperimentManager
    params_per_instance = [dict(hyperparameter2=ii) for ii in range(2)]
    stats_agent1 = ExperimentManager(
        DummyAgent,
        train_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=2,
        seed=123,
        init_kwargs_per_instance=params_per_instance,
    )
    stats_agent2 = ExperimentManager(
        DummyAgent2,
        train_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=2,
        seed=123,
    )
    experiment_manager_list = [stats_agent1, stats_agent2]
    for st in experiment_manager_list:
        st.fit()

    for ii, instance in enumerate(stats_agent1.agent_handlers):
        assert instance.hyperparameter1 == -1
        assert instance.hyperparameter2 == ii

    for ii, instance in enumerate(stats_agent2.agent_handlers):
        assert instance.hyperparameter1 == -1
        assert instance.hyperparameter2 == 100

    # learning curves
    plot_writer_data(experiment_manager_list, tag="episode_rewards", show=False)

    # compare final policies
    evaluate_agents(experiment_manager_list, show=False)

    # check if fitted
    for experiment_manager in experiment_manager_list:
        assert len(experiment_manager.agent_handlers) == 2
        for agent in experiment_manager.agent_handlers:
            assert agent.fitted

    # test saving/loading
    fname = stats_agent1.save()
    loaded_stats = ExperimentManager.load(fname)
    assert stats_agent1.unique_id == loaded_stats.unique_id

    # test hyperparameter optimization call
    loaded_stats.optimize_hyperparams(n_trials=3)
    loaded_stats.optimize_hyperparams(n_trials=3, continue_previous=True)

    for st in experiment_manager_list:
        st.clear_output_dir()


@pytest.mark.xfail(sys.platform == "win32", reason="bug with windows???")
def test_experiment_manager_2():
    # Define train and evaluation envs
    train_env = (GridWorld, {})
    eval_env = (GridWorld, {})

    # Parameters
    params = {}
    eval_kwargs = dict(eval_horizon=10)

    # Run ExperimentManager
    stats_agent1 = ExperimentManager(
        DummyAgent,
        train_env,
        eval_env=eval_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=4,
        seed=123,
    )
    stats_agent2 = ExperimentManager(
        DummyAgent2,
        train_env,
        eval_env=eval_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=4,
        seed=123,
    )
    experiment_manager_list = [stats_agent1, stats_agent2]
    for st in experiment_manager_list:
        st.fit()

    # compare final policies
    outputs = evaluate_agents(experiment_manager_list, n_simulations=5, show=False)
    assert len(outputs) == 5
    outputs = evaluate_agents(
        experiment_manager_list, n_simulations=5, show=False, choose_random_agents=False
    )
    assert len(outputs) == 4 * 5
    # learning curves
    plot_writer_data(experiment_manager_list, tag="episode_rewards", show=False)

    # check if fitted
    for experiment_manager in experiment_manager_list:
        assert len(experiment_manager.agent_handlers) == 4
        for agent in experiment_manager.agent_handlers:
            assert agent.fitted

    # test saving/loading
    fname = stats_agent1.save()
    loaded_stats = ExperimentManager.load(fname)
    assert stats_agent1.unique_id == loaded_stats.unique_id

    # test hyperparemeter optimization
    loaded_stats.optimize_hyperparams(n_trials=5)

    # delete some writers
    stats_agent1.set_writer(1, None)
    stats_agent1.set_writer(2, None)

    stats_agent1.clear_output_dir()
    stats_agent2.clear_output_dir()


@pytest.mark.parametrize("train_env", [(GridWorld, None), (None, None)])
def test_experiment_manager_partial_fit_and_tuple_env(train_env):
    # Define train and evaluation envs
    train_env = (
        GridWorld,
        None,
    )  # tuple (constructor, kwargs) must also work in ExperimentManager

    # Parameters
    params = {"compress_pickle": False}
    eval_kwargs = dict(eval_horizon=10)

    # Run ExperimentManager
    stats = ExperimentManager(
        DummyAgent,
        train_env,
        init_kwargs=params,
        n_fit=4,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        seed=123,
    )
    stats2 = ExperimentManager(
        DummyAgent2,
        train_env,
        init_kwargs=params,
        n_fit=4,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        seed=123,
    )

    # Run partial fit
    stats.fit(10)
    stats.fit(20)
    for agent in stats.agent_handlers:
        assert agent.total_budget == 30

    # Run fit
    stats2.fit()

    # learning curves
    plot_writer_data(
        [stats], tag="episode_rewards", show=False, preprocess_func=np.cumsum
    )

    # compare final policies
    evaluate_agents([stats], show=False)

    # delete some writers
    stats.set_writer(0, None)
    stats.set_writer(3, None)

    stats.clear_output_dir()
    stats2.clear_output_dir()


def test_equality():
    # Define train and evaluation envs
    train_env = (GridWorld, {})

    # Parameters
    params = dict(hyperparameter1=-1, hyperparameter2=100)
    eval_kwargs = dict(eval_horizon=10)

    # Run ExperimentManager
    params_per_instance = [dict(hyperparameter2=ii) for ii in range(4)]
    stats_agent1 = ExperimentManager(
        DummyAgent,
        train_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=4,
        seed=123,
        init_kwargs_per_instance=params_per_instance,
    )

    stats_agent2 = ExperimentManager(
        DummyAgent,
        train_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=4,
        seed=123,
        init_kwargs_per_instance=params_per_instance,
    )

    stats_agent3 = ExperimentManager(
        DummyAgent,
        train_env,
        fit_budget=42,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=4,
        seed=123,
        init_kwargs_per_instance=params_per_instance,
    )

    assert stats_agent1 == stats_agent2
    assert stats_agent1 != stats_agent3


def test_version():
    # Define train and evaluation envs
    train_env = (GridWorld, {})

    # Parameters
    params = dict(hyperparameter1=-1, hyperparameter2=100)
    eval_kwargs = dict(eval_horizon=10)

    # Run ExperimentManager
    params_per_instance = [dict(hyperparameter2=ii) for ii in range(4)]
    stats_agent1 = ExperimentManager(
        DummyAgent,
        train_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=4,
        seed=123,
        init_kwargs_per_instance=params_per_instance,
    )
    version = stats_agent1.rlberry_version
    assert (version is not None) and (len(version) > 0)


def test_profile():
    # Define train and evaluation envs
    train_env = (GridWorld, {})

    # Parameters
    params = dict(hyperparameter1=-1, hyperparameter2=100)
    eval_kwargs = dict(eval_horizon=10)

    # Run ExperimentManager
    params_per_instance = [dict(hyperparameter2=ii) for ii in range(4)]
    stats_agent1 = ExperimentManager(
        DummyAgent,
        train_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=4,
        seed=123,
        init_kwargs_per_instance=params_per_instance,
    )
    stats_agent1.generate_profile(fname="profile.prof")
    assert (
        os.path.getsize("profile.prof") > 100
    ), "experiment manager saved an empty profile"


def test_preset():
    # Define train and evaluation envs
    train_env = (GridWorld, {})

    # Parameters
    params = dict(hyperparameter1=-1, hyperparameter2=100)
    eval_kwargs = dict(eval_horizon=10)

    # Run ExperimentManager
    params_per_instance = [dict(hyperparameter2=ii) for ii in range(4)]

    manager_maker = preset_manager(
        train_env=train_env,
        fit_budget=4,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=4,
        seed=123,
        init_kwargs_per_instance=params_per_instance,
    )
    manager = manager_maker(DummyAgent)
    manager.fit()


def test_compress():
    # Define train and evaluation envs
    train_env = (GridWorld, {})

    # Parameters
    params = dict(
        hyperparameter1=-1, hyperparameter2=lambda x: 42, compress_pickle=True
    )
    eval_kwargs = dict(eval_horizon=10)

    # Run ExperimentManager
    stats = ExperimentManager(
        DummyAgent,
        train_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=3,
        seed=123,
    )
    stats.fit()
    evaluate_agents([stats], show=False)


@pytest.mark.parametrize("style_log", ["multi_line", "one_line", "progressbar"])
def test_logs(style_log):
    # Define train and evaluation envs
    train_env = (GridWorld, {})

    # Parameters
    params = dict(hyperparameter1=-1, hyperparameter2=lambda x: 42)
    eval_kwargs = dict(eval_horizon=10)

    # Run ExperimentManager
    stats = ExperimentManager(
        DummyAgent,
        train_env,
        fit_budget=15,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        default_writer_kwargs={"style_log": style_log, "log_interval": 0},
        n_fit=3,
        seed=123,
    )
    stats.fit()
    evaluate_agents([stats], show=False)


class DummyAgentNotPickleable(AgentWithSimplePolicy):
    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")

        self.name = "DummyAgent"
        self.fitted = False
        file = open("profile.prof", "w")
        self.not_pickleable = file
        self.total_budget = 0.0

    def fit(self, budget, **kwargs):
        del kwargs
        self.fitted = True
        self.total_budget += budget
        self.env.step(0)

        return

    def policy(self, observation):
        return 0


class DummyAgentNotPickleableTorch(AgentTorch):
    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")

        self.name = "DummyAgent"
        self.fitted = False
        file = open("profile.prof", "w")
        self.not_pickleable = file
        self.total_budget = 0.0

    def fit(self, budget, **kwargs):
        del kwargs
        self.fitted = True
        self.total_budget += budget
        self.env.step(0)
        return

    def policy(self, observation):
        return 0

    def eval(self, eval_horizon=10**5, n_simulations=10, gamma=1.0, **kwargs):
        return 0


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize(
    "agent", [DummyAgentNotPickleable, DummyAgentNotPickleableTorch]
)
def test_not_pickle(compress, agent):
    check_save_load(agent)
    train_env = (GridWorld, None)
    # Define train and evaluation envs
    train_env = (
        GridWorld,
        None,
    )  # tuple (constructor, kwargs) must also work in ExperimentManager

    # Parameters
    params = {"compress_pickle": compress}
    eval_kwargs = dict(eval_horizon=10)

    # Run ExperimentManager
    stats = ExperimentManager(
        agent,
        train_env,
        init_kwargs=params,
        n_fit=4,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        seed=123,
    )

    # Run partial fit
    stats.fit(10)
    stats.load(stats.save())

    # learning curves
    plot_writer_data(
        [stats], tag="episode_rewards", show=False, preprocess_func=np.cumsum
    )
    # compare final policies
    evaluate_agents([stats], show=False)
    data = read_writer_data(stats)

    stats.clear_output_dir()


def test_fitbudget_exception():
    msg = "\[ExperimentManager\] fit_budget missing in __init__\(\)\."  # /!\ regex : need to escape some char.
    with pytest.raises(ValueError, match=msg):
        # Define train and evaluation envs
        train_env = (GridWorld, {})

        # Run ExperimentManager
        stats = ExperimentManager(
            DummyAgent,
            train_env,
            n_fit=3,
            seed=123,
        )


def test_save_logger_and_warning(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_test_name = "test_agent"

        path_to_save = tmpdir + "/" + agent_test_name
        if os.path.isdir(path_to_save):  # pragma: no cover
            shutil.rmtree(path_to_save)  # pragma: no cover

        # Define train and evaluation envs
        train_env = (GridWorld, {})

        # Run ExperimentManager
        stats_agent1 = ExperimentManager(
            DummyAgent,
            train_env,
            fit_budget=5,
            n_fit=4,
            seed=123,
            agent_name=agent_test_name,
            output_dir=path_to_save,
            outdir_id_style=None,
        )

        ExperimentManager_message_begin = "The ExperimentManager was saved in : '"
        assert not ExperimentManager_message_begin in caplog.text
        stats_agent1.fit()
        assert ExperimentManager_message_begin in caplog.text

        path = stats_agent1.save()
        assert str(ExperimentManager_message_begin + str(path) + "'") in caplog.text

        warning_overwrite_message = "This output directory already exists, the save may overwrite the previous Experiment."
        assert not warning_overwrite_message in caplog.text
        stats_agent2 = ExperimentManager(
            DummyAgent,
            train_env,
            fit_budget=5,
            n_fit=4,
            seed=456,
            agent_name=agent_test_name,
            output_dir=path_to_save,
            outdir_id_style=None,
        )

        stats_agent2.fit()
        assert warning_overwrite_message in caplog.text
