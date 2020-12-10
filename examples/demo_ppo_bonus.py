
import rlberry.seeding as seeding
from copy import deepcopy
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.agents.ppo import PPOAgent
from rlberry.stats import AgentStats, plot_episode_rewards, compare_policies
from rlberry.exploration_tools.online_discretization_counter import OnlineDiscretizationCounter
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.wrappers.uncertainty_estimator_wrapper import UncertaintyEstimatorWrapper


# global seed
seeding.set_global_seed(12345)

# --------------------------------
# Define train env
# --------------------------------
_env = get_benchmark_env(level=4)
eval_env = get_benchmark_env(level=4)


def uncertainty_estimator_fn(obs_space, act_space):
    counter = DiscreteCounter(obs_space,
                              act_space,
                              n_bins_obs=20)
    return counter


env = UncertaintyEstimatorWrapper(_env,
                                  uncertainty_estimator_fn,
                                  bonus_scale_factor=1.0)

# -----------------------------
# Parameters
# -----------------------------
N_EPISODES = 2000
GAMMA = 0.99
HORIZON = 30
BONUS_SCALE_FACTOR = 0.1
MIN_DIST = 0.1

params_ppo = {'n_episodes': N_EPISODES,
              'gamma': GAMMA,
              'horizon': HORIZON,
              'batch_size': 16,
              'entr_coef': 8e-7,
              'k_epochs': 10,
              'eps_clip': 0.2,
              'learning_rate': 0.03
              }

params_ppo_bonus = {
              'n_episodes': N_EPISODES,
              'gamma': GAMMA,
              'horizon': HORIZON,
              'batch_size': 16,
              'entr_coef': 8e-7,
              'k_epochs': 10,
              'eps_clip': 0.2,
              'learning_rate': 0.03,
              'use_bonus_if_available': True,
              }


# -----------------------------
# Run AgentStats
# -----------------------------
ppo_stats = AgentStats(PPOAgent, env, eval_env=eval_env, init_kwargs=params_ppo, n_fit=4, agent_name='PPO')
ppo_bonus_stats = AgentStats(PPOAgent, env, eval_env=eval_env, init_kwargs=params_ppo_bonus, n_fit=4, agent_name='PPO-Bonus')

agent_stats_list = [ppo_bonus_stats, ppo_stats]

# learning curves
plot_episode_rewards(agent_stats_list, cumulative=True, show=False)

# compare final policies
output = compare_policies(agent_stats_list, eval_horizon=HORIZON, n_sim=20)
print(output)
