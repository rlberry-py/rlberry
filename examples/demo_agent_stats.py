import rlberry.seeding as seeding
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents import RSKernelUCBVIAgent, RSUCBVIAgent
from rlberry.agents.ppo import PPOAgent
from rlberry.stats import AgentStats, plot_episode_rewards, compare_policies


# global seed
seeding.set_global_seed(1234)

# --------------------------------
# Define train and evaluation envs
# --------------------------------
train_env = PBall2D()
eval_env = PBall2D()


# -----------------------------
# Parameters
# -----------------------------
N_EPISODES = 500
GAMMA = 0.99
HORIZON = 50
BONUS_SCALE_FACTOR = 0.1
MIN_DIST = 0.1
VERBOSE = 4

params = {
    "n_episodes": N_EPISODES,
    "gamma": GAMMA,
    "horizon": HORIZON,
    "bonus_scale_factor": BONUS_SCALE_FACTOR,
    "min_dist": MIN_DIST,
    "bandwidth": 0.1,
    "beta": 1.0,
    "kernel_type": "gaussian",
    "verbose": VERBOSE
}

params_ppo = {"n_episodes": N_EPISODES,
              "gamma": GAMMA,
              "horizon": HORIZON,
              "learning_rate": 0.0003}

# -----------------------------
# Run AgentStats
# -----------------------------
rsucbvi_stats = AgentStats(RSUCBVIAgent, train_env,
                           init_kwargs=params, n_fit=4)
rskernel_stats = AgentStats(RSKernelUCBVIAgent, train_env,
                            init_kwargs=params, n_fit=4)
ppo_stats = AgentStats(PPOAgent, train_env, init_kwargs=params_ppo, n_fit=4)

agent_stats_list = [rsucbvi_stats, rskernel_stats, ppo_stats]

# learning curves
plot_episode_rewards(agent_stats_list, cumulative=True, show=False)

# compare final policies
output = compare_policies(agent_stats_list, eval_env,
                          eval_horizon=HORIZON, n_sim=10)
print(output)
