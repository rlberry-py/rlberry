import numpy as np
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents.torch.ppo import PPOAgent
from rlberry.stats import AgentStats, plot_writer_data, compare_policies


# --------------------------------
# Define train and evaluation envs
# --------------------------------
train_env = PBall2D()
eval_env = PBall2D()


# -----------------------------
# Parameters
# -----------------------------
N_EPISODES = 100
GAMMA = 0.99
HORIZON = 50
BONUS_SCALE_FACTOR = 0.1
MIN_DIST = 0.1


params_ppo = {"n_episodes": N_EPISODES,
              "gamma": GAMMA,
              "horizon": HORIZON,
              "learning_rate": 0.0003}

# -------------------------------
# Run AgentStats and save results
# --------------------------------
ppo_stats = AgentStats(PPOAgent, train_env, init_kwargs=params_ppo, n_fit=4, output_dir='ppo_stats')
ppo_stats.fit()  # fit the 4 agents
ppo_stats.save()
del ppo_stats

# -------------------------------
# Load and plot results
# --------------------------------
ppo_stats = AgentStats.load('ppo_stats/stats.pickle')

# learning curves
plot_writer_data(ppo_stats, tag='episode_rewards',
                 preprocess_func=np.cumsum,
                 title='Cumulative Rewards', show=False)

# compare final policies
output = compare_policies([ppo_stats], eval_env,
                          eval_horizon=HORIZON, n_sim=10)
print(output)
