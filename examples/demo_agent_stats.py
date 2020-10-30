import rlberry.seeding as seeding
from rlberry.envs import PBall2D
from rlberry.agents import RSKernelUCBVIAgent, RSUCBVIAgent
from rlberry.wrappers import RescaleRewardWrapper
from rlberry.eval.agent_stats import AgentStats, plot_episode_rewards


# global seed
seeding.set_global_seed(93487698723)

# --------------------------------
# Define train env
# --------------------------------
train_env = PBall2D()


# -----------------------------
# Parameters
# -----------------------------
N_EPISODES = 1500
GAMMA = 0.99
HORIZON = 25
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

# -----------------------------
# Run AgentStats
# -----------------------------
rsucbvi_stats = AgentStats(RSUCBVIAgent, train_env, init_kwargs=params, nfit=4)
rsucbvi_stats.fit()

rskernel_stats = AgentStats(RSKernelUCBVIAgent, train_env, init_kwargs=params, nfit=4)
rskernel_stats.fit()


plot_episode_rewards([rsucbvi_stats, rskernel_stats], cumulative=True,  show=False)
plot_episode_rewards([rsucbvi_stats, rskernel_stats], cumulative=False, show=True)
