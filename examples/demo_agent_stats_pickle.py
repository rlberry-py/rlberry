from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents.ppo import PPOAgent
from rlberry.stats import AgentStats, plot_episode_rewards, compare_policies


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
agent_stats_list = [ppo_stats]

# learning curves
plot_episode_rewards(agent_stats_list, cumulative=True, show=False)

# compare final policies
output = compare_policies(agent_stats_list, eval_env,
                          eval_horizon=HORIZON, n_sim=10)
print(output)
