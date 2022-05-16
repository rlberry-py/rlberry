import numpy as np
from rlberry.envs import GridWorld
from rlberry.agents import UCBVIAgent
from rlberry.manager import AgentManager, read_writer_data
from rlberry.wrappers import WriterWrapper


def test_myoutput(capsys):  # or use "capfd" for fd-level
    env_ctor = GridWorld
    env_kwargs = dict()

    env = env_ctor(**env_kwargs)
    agent = AgentManager(UCBVIAgent, (env_ctor, env_kwargs), fit_budget=100, n_fit=1)
    agent.fit(budget=100)

    captured = capsys.readouterr()
    # test that what is written to stderr is longer than 100 char,
    assert len(captured.err) > 100, "the logging did not print the info to stderr"
