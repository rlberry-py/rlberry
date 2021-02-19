from rlberry.agents import KOVIAgent
from rlberry.envs.finite import GridWorld
import numpy as np


# Create pd kernel
def pd_kernel(state1, action1, state2=None, action2=None, type='gaussian'):
    if state2 is None and action2 is None:
        state2, action2 = state1, action1

    if type == 'gaussian':
        sigma = 1.0
        return np.exp(- (state1 - state2) ** 2 / (2 * sigma)) * (action1 == action2)
    else:
        return


def test_kovi_agent():
    env = GridWorld(nrows=2, ncols=4, walls=(), success_probability=1.0)
    n_episodes = 50
    horizon = 30

    agent = KOVIAgent(env,
                      n_episodes=n_episodes,
                      horizon=horizon,
                      pd_kernel_fn=pd_kernel,
                      gamma=0.99
                      )
    agent._log_interval = 0
    agent.fit()
    agent.policy(env.observation_space.sample())
