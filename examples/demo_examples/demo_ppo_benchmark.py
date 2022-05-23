"""
 =====================
 Demo: demo_ppo_benchmark
 =====================
"""
import numpy as np
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.agents import MBQVIAgent
from rlberry.agents.experimental.torch.ppo import PPOAgent
from rlberry.wrappers import DiscretizeStateWrapper
from rlberry.manager import AgentManager, plot_writer_data, evaluate_agents

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
    "horizon": HORIZON,
}

params_ppo = {"gamma": GAMMA, "horizon": HORIZON, "learning_rate": 0.0003}

eval_kwargs = dict(eval_horizon=HORIZON, n_simulations=20)

# -----------------------------
# Run AgentManager
# -----------------------------
oracle_stats = AgentManager(
    MBQVIAgent,
    d_train_env,
    fit_budget=0,
    init_kwargs=params_oracle,
    eval_kwargs=eval_kwargs,
    n_fit=4,
    agent_name="Oracle",
)
ppo_stats = AgentManager(
    PPOAgent,
    train_env,
    fit_budget=N_EPISODES,
    init_kwargs=params_ppo,
    eval_kwargs=eval_kwargs,
    n_fit=4,
    agent_name="PPO",
)

agent_manager_list = [oracle_stats, ppo_stats]

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
