import pytest
from rlberry.envs import GridWorld
from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import AgentManager
from rlberry.manager import compare_agents


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
        return None

    def policy(self, observation):
        return 0


@pytest.mark.parametrize("method", ["tukey_hsd", "permutation"])
def test_compare(method):
    # Define train and evaluation envs
    train_env = (GridWorld, {})
    eval_env = (GridWorld, {})

    # Parameters
    params = {}
    eval_kwargs = dict(eval_horizon=10)

    # Run AgentManager
    agent1 = AgentManager(
        DummyAgent,
        train_env,
        agent_name="Dummy1",
        eval_env=eval_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=4,
        seed=123,
    )
    agent2 = AgentManager(
        DummyAgent,
        train_env,
        eval_env=eval_env,
        agent_name="Dummy2",
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs=params,
        n_fit=4,
        seed=123,
    )
    agent1.fit()
    agent2.fit()

    df = compare_agents([agent1, agent2], method=method, B=10, n_simulations=5)
    assert len(df) > 0

    agent1_pickle = agent1.save()
    agent2_pickle = agent2.save()

    df = compare_agents(
        [agent1_pickle, agent2_pickle], method=method, B=10, n_simulations=5
    )
    assert len(df) > 0


test_compare("tukey_hsd")
