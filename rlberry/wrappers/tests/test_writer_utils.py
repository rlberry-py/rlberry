import numpy as np
import pytest

from rlberry.wrappers import WriterWrapper
from rlberry.envs import GridWorld

from rlberry.agents import UCBVIAgent

@pytest.mark.parametrize("write_scalar", ['action','reward', "action_and_reward"])
def test_wrapper(write_scalar):
    """ Test that the wrapper record data"""
    class MyAgent(UCBVIAgent):
        def __init__(self, env, **kwargs):
            UCBVIAgent.__init__(self, env, **kwargs)
            self.env = WriterWrapper(self.env, self.writer, write_scalar = write_scalar)
    env = GridWorld()
    agent = MyAgent(env)
    agent.fit(budget=10)
    assert len(agent.writer.data)>0

def test_invalid_wrapper_name():
    """Test that warning message is thrown when unsupported write_scalar."""
    class MyAgent(UCBVIAgent):
        def __init__(self, env, **kwargs):
            UCBVIAgent.__init__(self, env, **kwargs)
            self.env = WriterWrapper(self.env, self.writer, write_scalar = "invalid")
    msg = "write_scalar invalid is not known"
    env = GridWorld()
    agent = MyAgent(env)
    with pytest.raises(ValueError, match=msg):
        agent.fit(budget=10)
