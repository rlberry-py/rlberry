import numpy as np
import pytest
from rlberry.envs import Chain
from gym.utils.env_checker import check_env
from rlberry.utils.check_agent import (
    check_rl_agent,
    _fit_agent_manager,
    check_agents_almost_equal,
)
from rlberry.agents import ValueIterationAgent


class TestEnv(Chain):
    def step(self, action):
        observation = np.array([1.0, 1.5, 0.5])
        reward = 1
        done = True
        return observation, reward, done


class TestAgent(ValueIterationAgent):
    def __init__(self, **kwargs):
        ValueIterationAgent.__init__(self, **kwargs)

    def eval(self, eval_horizon=5, n_simulations=2, gamma=1.0, **kwargs):
        """
        rewrite the eval function to have a smaller eval_horizon
        """
        del kwargs  # unused
        episode_rewards = np.zeros(n_simulations)
        for sim in range(n_simulations):
            observation = self.eval_env.reset()
            tt = 0
            while tt < eval_horizon:
                action = self.policy(observation)
                observation, reward, done, _ = self.eval_env.step(action)
                episode_rewards[sim] += reward * np.power(gamma, tt)
                tt += 1
                if done:
                    break
        return episode_rewards.mean()


def test_check_env_dict_action():
    # Environment.step() only returns 3 values: obs, reward, done. Not info!
    test_env = TestEnv()

    with pytest.raises(AssertionError) as errorinfo:
        check_env(env=test_env, warn=True)
        assert (
            str(errorinfo.value)
            == "The `step()` method must return four values: obs, reward, done, info"
        )


def test_check_agent():
    check_rl_agent(ValueIterationAgent, (Chain, {}))


def test_check_agent_manager_almost_equal():
    agent1 = _fit_agent_manager(TestAgent, "discrete_state").agent_handlers[0]
    agent2 = _fit_agent_manager(TestAgent, "discrete_state").agent_handlers[0]
    check_agents_almost_equal(agent1, agent2, compare_using="eval")
