import numpy as np
from rlberry.agents.features import FeatureMap
from rlberry.envs.finite import GridWorld
from rlberry.stats import AgentStats, plot_episode_rewards,\
    compare_policies
from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.agents.linear import LSVIUCBAgent


# Define environment
env = GridWorld(nrows=2, ncols=3, walls=(), success_probability=1.0)


# Create feature map
class OneHotFeatureMap(FeatureMap):
    def __init__(self, S, A):
        self.S = env.observation_space.n
        self.A = env.action_space.n
        self.shape = (S*A,)

    def map(self, observation, action):
        feat = np.zeros((self.S, self.A))
        feat[observation, action] = 1.0
        return feat.flatten()


# Function that returns an instance of a feature map
def feature_map_fn():
    return OneHotFeatureMap(env.observation_space.n, env.action_space.n)


params = {'n_episodes': 200,
          'feature_map_fn': feature_map_fn,
          'horizon': 10,
          'bonus_scale_factor': 0.1,
          'gamma': 0.99
          }

params_greedy = {'n_episodes': 200,
                 'feature_map_fn': feature_map_fn,
                 'horizon': 10,
                 'bonus_scale_factor': 0.0,
                 'gamma': 0.99
                 }

params_oracle = {
                    'horizon': 10,
                    'gamma': 0.99
                    }

stats = AgentStats(LSVIUCBAgent,
                   env,
                   eval_horizon=10,
                   init_kwargs=params,
                   n_fit=1)

stats_random = AgentStats(LSVIUCBAgent,
                          env,
                          eval_horizon=10,
                          init_kwargs=params_greedy,
                          n_fit=2,
                          agent_name='LSVI-random-expl')

oracle_stats = AgentStats(ValueIterationAgent,
                          env,
                          eval_horizon=10,
                          init_kwargs=params_oracle,
                          n_fit=1)


plot_episode_rewards([stats, stats_random], cumulative=True, show=False)
compare_policies([stats, stats_random, oracle_stats], show=True)


# stats.fit()
# agent_eval = stats.fitted_agents[0]

# # visualize
# agent_eval = stats.fitted_agents[0]
# state = env.reset()
