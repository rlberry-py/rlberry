import numpy as np
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.agents import MBQVIAgent
from rlberry.agents.torch.ppo import PPOAgent
from rlberry.wrappers import DiscretizeStateWrapper
from rlberry.stats import AgentStats, plot_writer_data, compare_policies


# --------------------------------
# Define train and evaluation envs
# --------------------------------
train_env = get_benchmark_env(level=1)
d_train_env = DiscretizeStateWrapper(train_env, 20)


# -----------------------------
# Parameters
# -----------------------------
N_EPISODES = 1000
GAMMA = 0.99
HORIZON = 30

params_oracle = {
    "n_samples": 20,  # samples per state-action
    "gamma": GAMMA,
    "horizon": HORIZON
}

params_ppo = {"n_episodes": N_EPISODES,
              "gamma": GAMMA,
              "horizon": HORIZON,
              "learning_rate": 0.0003}

# -----------------------------
# Run AgentStats
# -----------------------------
oracle_stats = AgentStats(MBQVIAgent, d_train_env, init_kwargs=params_oracle,
                          n_fit=4, agent_name="Oracle")
ppo_stats = AgentStats(PPOAgent, train_env, init_kwargs=params_ppo,
                       n_fit=4, agent_name="PPO")

agent_stats_list = [oracle_stats, ppo_stats]

# learning curves
plot_writer_data(agent_stats_list, tag='episode_rewards',
                 preprocess_func=np.cumsum,
                 title='Cumulative Rewards', show=False)


# compare final policies
output = compare_policies(agent_stats_list, eval_horizon=HORIZON, n_sim=10)
print(output)
