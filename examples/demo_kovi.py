import numpy as np
from rlberry.agents.features import FeatureMap
from rlberry.envs.finite import GridWorld
from rlberry.stats import AgentStats, plot_episode_rewards,\
    compare_policies
from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.agents.linear import LSVIUCBAgent
from rlberry.agents.kovi.kovi2 import KOVIAgent

from rlberry.utils.jit_setup import numba_jit

# Define environment
env = GridWorld(nrows=2, ncols=4, walls=(), success_probability=1.0)


# Create pd kernel
# @numba_jit
def pd_kernel(state1, action1, state2=None, action2=None, type='gaussian'):
    if state2 is None and action2 is None:
        state2, action2 = state1, action1

    if type == 'gaussian':
        sigma = 1.0
        # return np.exp(- np.linalg.norm(state1 - state2)**2 / (2 * sigma)) * (action1 == action2)
        return np.exp(- (state1 - state2) ** 2 / (2 * sigma)) * (action1 == action2)
    else:
        return


params = {'n_episodes': 500,
          'pd_kernel_fn': pd_kernel,
          'horizon': 10,
          'bonus_scale_factor': 0.01,
          'gamma': 0.99
          }

params_greedy = {'n_episodes': 500,
                 'pd_kernel_fn': pd_kernel,
                 'horizon': 10,
                 'bonus_scale_factor': 0.0,
                 'gamma': 0.99
                 }

params_oracle = {
                    'horizon': 10,
                    'gamma': 0.99
                    }

stats = AgentStats(KOVIAgent,
                   env,
                   eval_horizon=10,
                   init_kwargs=params,
                   n_fit=1)

stats_random = AgentStats(KOVIAgent,
                          env,
                          eval_horizon=10,
                          init_kwargs=params_greedy,
                          n_fit=1,
                          agent_name='LSVI-random-expl')

oracle_stats = AgentStats(ValueIterationAgent,
                          env,
                          eval_horizon=10,
                          init_kwargs=params_oracle,
                          n_fit=1)

plot_episode_rewards([stats, stats_random], cumulative=True, show=False)
compare_policies([stats, stats_random, oracle_stats], show=True)

