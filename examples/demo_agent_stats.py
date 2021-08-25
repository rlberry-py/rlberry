
import numpy as np
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents import RSKernelUCBVIAgent, RSUCBVIAgent
from rlberry.agents.torch.ppo import PPOAgent
from rlberry.stats import AgentStats, plot_writer_data, evaluate_agents


# --------------------------------
# Define train and evaluation envs
# --------------------------------
train_env = (PBall2D, dict())
eval_env = (PBall2D, dict())


# -----------------------------
# Parameters
# -----------------------------
N_EPISODES = 100
GAMMA = 0.99
HORIZON = 50
BONUS_SCALE_FACTOR = 0.1
MIN_DIST = 0.1

params = {
    "gamma": GAMMA,
    "horizon": HORIZON,
    "bonus_scale_factor": BONUS_SCALE_FACTOR,
    "min_dist": MIN_DIST,
}

params_kernel = {
    "gamma": GAMMA,
    "horizon": HORIZON,
    "bonus_scale_factor": BONUS_SCALE_FACTOR,
    "min_dist": MIN_DIST,
    "bandwidth": 0.1,
    "beta": 1.0,
    "kernel_type": "gaussian",
}

params_ppo = {"gamma": GAMMA,
              "horizon": HORIZON,
              "learning_rate": 0.0003}

eval_kwargs = dict(eval_horizon=HORIZON, n_simulations=20)

# -----------------------------
# Run AgentStats
# -----------------------------
rsucbvi_stats = AgentStats(
    RSUCBVIAgent,
    train_env,
    fit_budget=N_EPISODES,
    init_kwargs=params,
    eval_kwargs=eval_kwargs,
    n_fit=4,
    seed=123)
rskernel_stats = AgentStats(
    RSKernelUCBVIAgent,
    train_env,
    fit_budget=N_EPISODES,
    init_kwargs=params_kernel,
    eval_kwargs=eval_kwargs,
    n_fit=4,
    seed=123)
ppo_stats = AgentStats(
    PPOAgent,
    train_env,
    fit_budget=N_EPISODES,
    init_kwargs=params_ppo,
    eval_kwargs=eval_kwargs,
    n_fit=4,
    seed=123)


agent_stats_list = [rsucbvi_stats, rskernel_stats, ppo_stats]
for st in agent_stats_list:
    st.fit()

# learning curves
plot_writer_data(agent_stats_list,
                 tag='episode_rewards',
                 preprocess_func=np.cumsum,
                 title='cumulative rewards',
                 show=False)

# compare final policies
output = evaluate_agents(agent_stats_list)
print(output)

for st in agent_stats_list:
    st.clear_output_dir()
