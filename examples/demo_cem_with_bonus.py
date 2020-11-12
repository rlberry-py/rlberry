"""
TODO: check reproducibility issue! Maybe set_global_seed is not working for pytorch.
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import rlberry.seeding as seeding

seeding.set_global_seed(123)

from rlberry.agents.cem import CEMAgent
from rlberry.envs.toy_exploration import PBall2D
from rlberry.eval.agent_stats import AgentStats, plot_episode_rewards, compare_policies
from rlberry.exploration_tools.discrete_counter import DiscreteCounter


env = PBall2D(p=np.inf, reward_smoothness=np.array([0.2]), reward_centers=[np.array([-0.8, -0.8])])
train_env = env 
eval_env = env

counter = DiscreteCounter(env.observation_space, env.action_space)


cem_params = {'batch_size':16,
              'horizon':50,
              'gamma':1.0,
              'n_episodes':2000,
              'percentile':70,
              'learning_rate':0.01,
              'uncertainty_estimator':counter}

cem_params_no_bonus = cem_params.copy()
cem_params_no_bonus['uncertainty_estimator'] = None


# -----------------------------
# Run AgentStats
# -----------------------------
cem_stats          = AgentStats(CEMAgent, train_env, init_kwargs=cem_params,          nfit=4)
cem_no_bonus_stats = AgentStats(CEMAgent, train_env, init_kwargs=cem_params_no_bonus, nfit=4)
agent_stats_list   = [cem_stats, cem_no_bonus_stats]

# learning curves
plot_episode_rewards(agent_stats_list, cumulative=True, show=False)

# compare final policies
output = compare_policies(agent_stats_list, eval_env, eval_horizon=cem_params['horizon'], nsim=50)
print(output)


# #
# if counter is not None:
#     plt.scatter( counter.state_discretizer.discretized_elements[0, :], counter.state_discretizer.discretized_elements[1, :], c=counter.N_sa.sum(axis=1) )
#     plt.show()
