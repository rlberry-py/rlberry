import pytest
from rlberry_research.envs import GridWorld
from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import AgentManager
from rlberry.manager import compare_agents


class DummyAgent(AgentWithSimplePolicy):
    def __init__(self, env, eval_val=0, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.name = "DummyAgent"
        self.fitted = False
        self.eval_val = eval_val

        self.total_budget = 0.0

    def fit(self, budget, **kwargs):
        del kwargs
        self.fitted = True
        self.total_budget += budget
        return None

    def policy(self, observation):
        return 0

    def eval(self, eval_horizon=None, **kwargs):
        return self.eval_val


@pytest.mark.parametrize("method", ["tukey_hsd", "permutation"])
def test_compare(method):
    # Define train and evaluation envs
    train_env = (GridWorld, {})
    eval_env = (GridWorld, {})

    # Parameters
    eval_kwargs = dict(eval_horizon=10)

    # Run AgentManager
    agent1 = AgentManager(
        DummyAgent,
        train_env,
        agent_name="Dummy1",
        eval_env=eval_env,
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs={"eval_val": 0},
        n_fit=10,
        seed=123,
    )
    agent2 = AgentManager(
        DummyAgent,
        train_env,
        eval_env=eval_env,
        agent_name="Dummy2",
        fit_budget=5,
        eval_kwargs=eval_kwargs,
        init_kwargs={"eval_val": 10},
        n_fit=10,
        seed=123,
    )
    agent1.fit()
    agent2.fit()

    df = compare_agents([agent1, agent2], method=method, B=20, n_simulations=5, seed=42)
    assert len(df) > 0
    if method == "tukey_hsd":
        assert df["p-val"].item() < 0.05
    assert df["decisions"].iloc[0] == "reject"
    agent1_pickle = agent1.save()
    agent2_pickle = agent2.save()

    df = compare_agents(
        [agent1_pickle, agent2_pickle], method=method, B=10, n_simulations=5
    )
    assert len(df) > 0
