"""
Premade connectors for stable-baselines3
"""
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
from probing_environments.utils.type_hints import AgentType

from probing_environments.checks import (
    check_actor_and_critic_coupling,
    check_advantage_policy,
    check_backprop_value_net,
    check_loss_or_optimizer_value_net,
    check_reward_discounting,
)
from rlberry.agents.torch.a2c.a2c import A2CAgent


def init_agent(
    agent: type[A2CAgent],
    env: gym.Env,
    run_name: str,  # pylint: disable=W0613
    gamma: Optional[float] = 0.5,
    learning_rate: Optional[float] = 1e-3,
    num_envs: Optional[int] = None,
    seed: Optional[int] = 42,
) -> A2CAgent:
    """
    Initialize your agent on a given env while also setting the discount factor.

    Args:
        agent (AgentType) : The agent to be used
        env (gym.Env): The env to use with your agent.
        gamma (float, optional): The discount factor to use. Defaults to 0.5.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        AgentType: Your agent with the right settings.
    """
    agent_instance = agent(env=env(), gamma=gamma, learning_rate=learning_rate)
    return agent_instance


def train_agent(agent: A2CAgent, budget: Optional[int] = int(1e3)) -> AgentType:
    """
    Train your agent for a given budget/number of timesteps.

    Args:
        agent (AgentType): Your agent (created by init_agent)
        budget (int, optional): The number of timesteps to train the agent on. Defaults\
              to int(1e3).

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        AgentType: Your trained agents.
    """
    agent.fit(budget=budget)
    return agent


def get_value(agent: A2CAgent, obs: np.ndarray) -> np.ndarray:
    """
    Predict the value of a given obs (in numpy array format) using your current value \
        net.

    Args:
        agent (AgentType): Your agent to make the prediction.
        obs (np.ndarray): The observation to make the prediction on.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        np.ndarray: The predicted value of the given observation.
    """
    return agent.value_net(torch.from_numpy(obs).float()).detach().numpy()


def get_policy(agent: AgentType, obs: np.ndarray) -> List[float]:
    """
    Predict the probabilitie of actions in a given obs (in numpy array format) using\
          your current policy net.

    Args:
        agent (AgentType): Your agent to make the prediction.
        obs (np.ndarray): The observation to make the prediction on.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        List[float]: The probabilities of taking every actions.
    """
    obs_torch = torch.from_numpy(obs).float()
    action_dist = agent._policy_old(obs_torch)
    return action_dist.probs


def get_gamma(agent: AgentType) -> float:
    """
    Fetch the gamma/discount factor value from your agent (to use it in tests)

    Args:
        agent (AgentType): Your agent.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        float: The gamma/discount factor value of your agent
    """
    return agent.gamma


####################


AGENT = A2CAgent
LEARNING_RATE = 1e-3
BUDGET = 4e4


def test_check_loss_or_optimizer_value_net():
    """
    Test that check_loss_or_optimizer_value_net works on failproof sb3.
    """
    check_loss_or_optimizer_value_net(
        AGENT,
        init_agent,
        train_agent,
        get_value,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
    )


def test_check_backprop_value_net_1_env():
    """
    Test that check_backprop_value_net works on failproof sb3.
    """
    check_backprop_value_net(
        AGENT,
        init_agent,
        train_agent,
        get_value,
        num_envs=1,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
    )


def test_check_reward_discounting_1_env():
    """
    Test that check_reward_discounting works on failproof sb3.
    """
    check_reward_discounting(
        AGENT,
        init_agent,
        train_agent,
        get_value,
        get_gamma,
        num_envs=1,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
    )


def test_check_advantage_policy():
    """
    Test that check_advantage_policy works on failproof sb3.
    """
    check_advantage_policy(
        AGENT,
        init_agent,
        train_agent,
        get_policy,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
    )


def test_check_actor_and_critic_coupling_1_env():
    """
    Test that check_actor_and_critic_coupling works on failproof sb3.
    """
    check_actor_and_critic_coupling(
        AGENT,
        init_agent,
        train_agent,
        get_policy,
        get_value,
        num_envs=1,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
    )
