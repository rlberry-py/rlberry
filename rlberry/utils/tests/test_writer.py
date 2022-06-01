import time
from rlberry.envs import GridWorld
from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import AgentManager


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
                self.writer.add_scalar("a", 42, ii)
            time.sleep(1)
        return None

    def policy(self, observation):
        return 0


def test_myoutput(capsys):  # or use "capfd" for fd-level
    env_ctor = GridWorld
    env_kwargs = dict()

    env = env_ctor(**env_kwargs)
    agent = AgentManager(
        DummyAgent,
        (env_ctor, env_kwargs),
        fit_budget=3,
        n_fit=1,
        default_writer_kwargs={"log_interval": 1},
    )
    agent.fit(budget=3)

    captured = capsys.readouterr()
    # test that what is written to stderr is longer than 50 char,
    assert (
        len(captured.err) + len(captured.out) > 50
    ), "the logging did not print the info to stderr"
