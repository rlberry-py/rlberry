import numpy as np
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents.torch.ppo import PPOAgent
from rlberry.stats import AgentStats, plot_writer_data, compare_policies
from torch.utils.tensorboard import SummaryWriter


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

params_ppo = {"n_episodes": N_EPISODES,
              "gamma": GAMMA,
              "horizon": HORIZON,
              "learning_rate": 0.0003}

# -----------------------------
# Run AgentStats
# -----------------------------
ppo_stats = AgentStats(PPOAgent, train_env, init_kwargs=params_ppo, n_fit=4)

ppo_stats.set_writer(0, SummaryWriter, writer_kwargs={'comment': 'worker_0'})
ppo_stats.set_writer(1, SummaryWriter, writer_kwargs={'comment': 'worker_1'})

agent_stats_list = [ppo_stats]

agent_stats_list[0].fit()
agent_stats_list[0].save()  # after fit, writers are set to None to avoid pickle problems.

# compare final policies
output = compare_policies(agent_stats_list, eval_env,
                          eval_horizon=HORIZON, n_sim=10)
print(output)
