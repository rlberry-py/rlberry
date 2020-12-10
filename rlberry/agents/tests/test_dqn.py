from copy import deepcopy
from rlberry.envs import gym_make
from rlberry.agents.dqn import DQNAgent
from rlberry.exploration_tools.online_discretization_counter import \
    OnlineDiscretizationCounter
from rlberry.wrappers.uncertainty_estimator_wrapper import \
    UncertaintyEstimatorWrapper


def test_dqn_agent():
    _env = gym_make("CartPole-v0")

    #
    def uncertainty_estimator_fn(observation_space, action_space):
        counter = OnlineDiscretizationCounter(
                                  observation_space,
                                  action_space,
                                  min_dist=0.25)
        return counter

    env = UncertaintyEstimatorWrapper(_env,
                                      uncertainty_estimator_fn,
                                      bonus_scale_factor=1.0)
    #
    params = {"n_episodes": 10, 'use_bonus_if_available': True}
    agent = DQNAgent(env, **params)
    agent.fit()
    agent.policy(env.observation_space.sample())
