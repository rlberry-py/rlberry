import pytest
import numpy as np
from rlberry.agents import Agent
from rlberry.manager import AgentManager


class DummyAgent(Agent):
    def __init__(self, **kwargs):
        Agent.__init__(self, **kwargs)
        self.name = "DummyAgent"
        self.shared_data_id = id(self.thread_shared_data)

    def fit(self, budget, **kwargs):
        del budget, kwargs

    def eval(self, **kwargs):
        del kwargs
        return self.shared_data_id


@pytest.mark.parametrize("paralellization", ["thread", "process"])
def test_data_sharing(paralellization):
    shared_data = dict(X=np.arange(10))
    manager = AgentManager(
        agent_class=DummyAgent,
        fit_budget=-1,
        n_fit=4,
        parallelization=paralellization,
        thread_shared_data=shared_data,
    )
    manager.fit()
    data_ids = [agent.eval() for agent in manager.get_agent_instances()]
    unique_data_ids = list(set(data_ids))
    if paralellization == "thread":
        # id() is unique for each object: make sure that shared data have same id
        assert len(unique_data_ids) == 1
    else:
        # when using processes, make sure that data is copied and each instance
        # has its own data id
        assert len(unique_data_ids) == manager.n_fit
