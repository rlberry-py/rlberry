import numpy as np
import pytest
from rlberry.envs import GridWorld, Chain
from rlberry.utils.check_env import check_env
from rlberry.utils.check_agent import (
    check_rl_agent,
    _fit_agent_manager,
    check_agents_almost_equal,
)
from rlberry.spaces import Box, Dict, Discrete
import gymnasium as gym
from rlberry.agents import ValueIterationAgent, UCBVIAgent


class ActionDictTestEnv(gym.Env):
    action_space = Dict({"position": Discrete(1), "velocity": Discrete(1)})
    observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)

    def step(self, action):
        observation = np.array([1.0, 1.5, 0.5])
        reward = 1
        done = True
        return observation, reward, done

    def reset(self, seed=None, options=None):
        return np.array([1.0, 1.5, 0.5]), {}


class DummyAgent:
    def __init__(self, **kwargs):
        pass


class ReferenceAgent(ValueIterationAgent):
    def __init__(self, **kwargs):
        ValueIterationAgent.__init__(self, **kwargs)

    def eval(self, eval_horizon=5, n_simulations=2, gamma=1.0, **kwargs):
        """
        rewrite the eval function to have a smaller eval_horizon
        """
        del kwargs  # unused
        episode_rewards = np.zeros(n_simulations)
        for sim in range(n_simulations):
            observation, info = self.eval_env.reset()
            tt = 0
            while tt < eval_horizon:
                action = self.policy(observation)
                observation, reward, terminated, truncated, info = self.eval_env.step(
                    action
                )
                done = terminated or truncated
                episode_rewards[sim] += reward * np.power(gamma, tt)
                tt += 1

        return episode_rewards.mean()


def test_check_env_dict_action():
    # Environment.step() only returns 3 values: obs, reward, done. Not info!
    test_env = ActionDictTestEnv()

    with pytest.raises(AssertionError) as errorinfo:
        check_env(env=test_env)
        assert (
            str(errorinfo.value)
            == "The `step()` method must return four values: obs, reward, done, info"
        )


def test_check_agent():
    check_rl_agent(ValueIterationAgent, (Chain, {}))


def test_check_agent_manager_almost_equal():
    env = GridWorld
    env_kwargs = {}
    agent1 = _fit_agent_manager(ReferenceAgent, (env, env_kwargs)).agent_handlers[0]
    agent2 = _fit_agent_manager(ReferenceAgent, (env, env_kwargs)).agent_handlers[0]
    agent3 = _fit_agent_manager(UCBVIAgent, (env, env_kwargs)).agent_handlers[0]
    assert check_agents_almost_equal(agent1, agent2, compare_using="eval")
    assert not check_agents_almost_equal(agent1, agent3)


def test_error_message_check_agent():
    msg = "The env given in parameter is not implemented"
    with pytest.raises(ValueError, match=msg):
        check_rl_agent(ValueIterationAgent, "not_implemented")
    with pytest.raises(ValueError, match=msg):
        check_rl_agent(ValueIterationAgent, 42)
    msg = "Agent not compatible with Agent Manager"
    with pytest.raises(RuntimeError, match=msg):
        check_rl_agent(DummyAgent)
