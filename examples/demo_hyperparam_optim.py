import rlberry.seeding as seeding
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents import RSKernelUCBVIAgent, RSUCBVIAgent
from rlberry.agents.ppo import PPOAgent
from rlberry.wrappers import RescaleRewardWrapper
from rlberry.eval.agent_stats import AgentStats, plot_episode_rewards, compare_policies


# global seed
seeding.set_global_seed(1234)

# --------------------------------
# Define train and evaluation envs
# --------------------------------
train_env = PBall2D()
eval_env  = PBall2D()


# -----------------------------
# Parameters
# -----------------------------
N_EPISODES = 5
GAMMA = 0.99
HORIZON = 50
BONUS_SCALE_FACTOR = 0.1
MIN_DIST = 0.1
VERBOSE = 4


params_ppo = {"n_episodes" : N_EPISODES,
              "gamma" : GAMMA,
              "horizon" : HORIZON,
              "learning_rate": 0.0003,
              "verbose":5}

# -------------------------------
# Run AgentStats and save results
# --------------------------------
ppo_stats = AgentStats(PPOAgent, train_env, eval_horizon=HORIZON, init_kwargs=params_ppo, nfit=4)


# hyperparam optim
best_trial, data  = ppo_stats.hyperparam_optim(ntrials=100, max_time=10, nsim=5, nfit=2, njobs=2, 
                           sampler_method='random', pruner_method='halving')

# save 
ppo_stats.save('ppo_stats')


print("fitting agents after choosing hyperparams...")
ppo_stats.fit()  # fit the 4 agents 


