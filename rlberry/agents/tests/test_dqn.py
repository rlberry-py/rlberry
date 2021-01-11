from copy import deepcopy
from rlberry.envs import MountainCar
from rlberry.agents.dqn import DQNAgent
from rlberry.exploration_tools.online_discretization_counter import \
    OnlineDiscretizationCounter
from rlberry.wrappers.uncertainty_estimator_wrapper import \
    UncertaintyEstimatorWrapper


def test_dqn_agent():
    env = MountainCar()

    def uncertainty_estimator_fn(observation_space, action_space):
        counter = OnlineDiscretizationCounter(
                                  observation_space,
                                  action_space,
                                  min_dist=0.25)
        return counter

    agent = DQNAgent(env,
                     n_episodes=10,
                     use_bonus=True,
                     uncertainty_estimator_kwargs=dict(
                         uncertainty_estimator_fn=uncertainty_estimator_fn,
                         bonus_scale_factor=1.0
                     ))
    agent.fit()
    agent.policy(env.observation_space.sample())
