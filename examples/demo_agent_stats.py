
import numpy as np
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents import RSKernelUCBVIAgent, RSUCBVIAgent
from rlberry.agents.torch.ppo import PPOAgent
from rlberry.stats import AgentStats, plot_writer_data, evaluate_policies


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

params = {
    "n_episodes": N_EPISODES,
    "gamma": GAMMA,
    "horizon": HORIZON,
    "bonus_scale_factor": BONUS_SCALE_FACTOR,
    "min_dist": MIN_DIST,
}

params_kernel = {
    "n_episodes": N_EPISODES,
    "gamma": GAMMA,
    "horizon": HORIZON,
    "bonus_scale_factor": BONUS_SCALE_FACTOR,
    "min_dist": MIN_DIST,
    "bandwidth": 0.1,
    "beta": 1.0,
    "kernel_type": "gaussian",
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
                            init_kwargs=params_kernel, n_fit=4)
ppo_stats = AgentStats(PPOAgent, train_env, init_kwargs=params_ppo, n_fit=4)

agent_stats_list = [rsucbvi_stats, rskernel_stats, ppo_stats]

# learning curves
plot_writer_data(agent_stats_list,
                 tag='episode_rewards',
                 preprocess_func=np.cumsum,
                 title='cumulative rewards',
                 show=False)

# compare final policies
output = evaluate_policies(agent_stats_list, eval_env,
                           eval_horizon=HORIZON, n_sim=10)
print(output)
