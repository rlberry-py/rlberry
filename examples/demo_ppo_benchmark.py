import numpy as np
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.agents import MBQVIAgent
from rlberry.agents.torch.ppo import PPOAgent
from rlberry.wrappers import DiscretizeStateWrapper
from rlberry.stats import AgentStats, plot_writer_data, evaluate_policies


# --------------------------------
# Define train and evaluation envs
# --------------------------------
env_ctor = get_benchmark_env
env_kwargs = dict(level=1)
discrete_env_ctor = lambda level: DiscretizeStateWrapper(env_ctor(level), 20)

train_env = (env_ctor, env_kwargs)
d_train_env = (discrete_env_ctor, env_kwargs)

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
output = evaluate_policies(agent_stats_list, eval_horizon=HORIZON, n_sim=10)
print(output)
