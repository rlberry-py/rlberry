""" 
 ===================== 
 Demo: demo_ppo_bonus 
 =====================
"""
import numpy as np
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.agents.experimental.torch.ppo import PPOAgent
from rlberry.manager import AgentManager, plot_writer_data, evaluate_agents
from rlberry.exploration_tools.discrete_counter import DiscreteCounter

# --------------------------------
# Define train env
# --------------------------------
env = (get_benchmark_env, dict(level=4))


def uncertainty_estimator_fn(obs_space, act_space):
    counter = DiscreteCounter(obs_space, act_space, n_bins_obs=20)
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
    "gamma": GAMMA,
    "horizon": HORIZON,
    "batch_size": 16,
    "entr_coef": 8e-7,
    "k_epochs": 10,
    "eps_clip": 0.2,
    "learning_rate": 0.03,
}

params_ppo_bonus = {
    "gamma": GAMMA,
    "horizon": HORIZON,
    "batch_size": 16,
    "entr_coef": 8e-7,
    "k_epochs": 10,
    "eps_clip": 0.2,
    "learning_rate": 0.03,
    "use_bonus": True,
    "uncertainty_estimator_kwargs": {
        "uncertainty_estimator_fn": uncertainty_estimator_fn
    },
}

eval_kwargs = dict(eval_horizon=HORIZON, n_simulations=20)

# -----------------------------
# Run AgentManager
# -----------------------------
ppo_stats = AgentManager(
    PPOAgent,
    env,
    fit_budget=N_EPISODES,
    init_kwargs=params_ppo,
    eval_kwargs=eval_kwargs,
    n_fit=4,
    agent_name="PPO",
)
ppo_bonus_stats = AgentManager(
    PPOAgent,
    env,
    fit_budget=N_EPISODES,
    init_kwargs=params_ppo_bonus,
    eval_kwargs=eval_kwargs,
    n_fit=4,
    agent_name="PPO-Bonus",
)

agent_manager_list = [ppo_bonus_stats, ppo_stats]

for manager in agent_manager_list:
    manager.fit()

# learning curves
plot_writer_data(
    agent_manager_list,
    tag="episode_rewards",
    preprocess_func=np.cumsum,
    title="Cumulative Rewards",
    show=False,
)

# compare final policies
output = evaluate_agents(agent_manager_list)
print(output)
