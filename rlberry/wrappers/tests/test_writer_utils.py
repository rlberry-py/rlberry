import pytest

from rlberry_scool.envs import GridWorld

from rlberry_scool.agents import UCBVIAgent


@pytest.mark.parametrize(
    "write_scalar", [None, "action", "reward", "action_and_reward"]
)
def test_wrapper(write_scalar):
    """Test that the wrapper record data"""

    class MyAgent(UCBVIAgent):
        def __init__(self, env, **kwargs):
            UCBVIAgent.__init__(self, env, writer_extra=write_scalar, **kwargs)

    env = GridWorld()
    agent = MyAgent(env)
    agent.fit(budget=10)
    assert len(agent.writer.data) > 0


def test_invalid_wrapper_name():
    """Test that warning message is thrown when unsupported write_scalar."""

    class MyAgent(UCBVIAgent):
        def __init__(self, env, **kwargs):
            UCBVIAgent.__init__(self, env, writer_extra="invalid", **kwargs)

    msg = "write_scalar invalid is not known"
    env = GridWorld()
    agent = MyAgent(env)
    with pytest.raises(ValueError, match=msg):
        agent.fit(budget=10)
