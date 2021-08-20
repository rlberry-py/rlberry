import numpy as np
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.agents.torch.ppo import PPOAgent
from rlberry.stats import AgentStats, plot_writer_data, compare_policies
from rlberry.exploration_tools.discrete_counter import DiscreteCounter


# --------------------------------
# Define train env
# --------------------------------
env = get_benchmark_env(level=4)
eval_env = get_benchmark_env(level=4)


def uncertainty_estimator_fn(obs_space, act_space):
    counter = DiscreteCounter(obs_space,
                              act_space,
                              n_bins_obs=20)
    return counter


# -----------------------------
# Parameters
# -----------------------------
N_EPISODES = 200
GAMMA = 0.99
HORIZON = 30
BONUS_SCALE_FACTOR = 0.1
MIN_DIST = 0.1

params_ppo = {
    'n_episodes': N_EPISODES,
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
    'use_bonus': True,
    'uncertainty_estimator_kwargs': {
        'uncertainty_estimator_fn': uncertainty_estimator_fn}
}


# -----------------------------
# Run AgentStats
# -----------------------------
ppo_stats = AgentStats(PPOAgent, env, eval_env=eval_env, init_kwargs=params_ppo, n_fit=4, agent_name='PPO')
ppo_bonus_stats = AgentStats(
    PPOAgent, env, eval_env=eval_env, init_kwargs=params_ppo_bonus, n_fit=4, agent_name='PPO-Bonus')

agent_stats_list = [ppo_bonus_stats, ppo_stats]

# learning curves
plot_writer_data(agent_stats_list, tag='episode_rewards',
                 preprocess_func=np.cumsum,
                 title='Cumulative Rewards', show=False)

# compare final policies
output = compare_policies(agent_stats_list, eval_horizon=HORIZON, n_sim=20)
print(output)
