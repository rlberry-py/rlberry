import time
from rlberry_research.envs import GridWorld
from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import ExperimentManager


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
                self.writer.add_scalar("a", ii, ii)
                scalar_dict = dict(multi1=1, multi2=2)
                self.writer.add_scalars("multi_scalar_test", scalar_dict)
            time.sleep(1)

        return None

    def policy(self, observation):
        return 0


def test_myoutput(capsys):  # or use "capfd" for fd-level
    env_ctor = GridWorld
    env_kwargs = dict()

    env = env_ctor(**env_kwargs)
    xp_manager = ExperimentManager(
        DummyAgent,
        (env_ctor, env_kwargs),
        fit_budget=3,
        n_fit=1,
        default_writer_kwargs={"log_interval": 1},
    )
    budget_size = 22
    xp_manager.fit(budget=budget_size)

    assert xp_manager.agent_handlers[0].writer.summary_writer == None
    assert list(xp_manager.agent_handlers[0].writer.read_tag_value("a")) == list(
        range(budget_size)
    )
    assert xp_manager.agent_handlers[0].writer.read_first_tag_value("a") == 0
    assert (
        xp_manager.agent_handlers[0].writer.read_last_tag_value("a") == budget_size - 1
    )  # start at 0

    captured = capsys.readouterr()
    # test that what is written to stderr is longer than 50 char,
    assert (
        len(captured.err) + len(captured.out) > 50
    ), "the logging did not print the info to stderr"
