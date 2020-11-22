import rlberry.seeding as seeding
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents import RSKernelUCBVIAgent, RSUCBVIAgent
from rlberry.agents.ppo import PPOAgent
from rlberry.wrappers import RescaleRewardWrapper
from rlberry.stats import AgentStats, plot_episode_rewards, compare_policies


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
ppo_stats = AgentStats(PPOAgent, train_env, eval_horizon=HORIZON, init_kwargs=params_ppo, n_fit=4)


# hyperparam optim
best_trial, data  = ppo_stats.optimize_hyperparams(n_trials=10, timeout=None, n_sim=5, n_fit=2, n_jobs=2, 
                           sampler_method='optuna_default')

initial_n_trials = len(ppo_stats.study.trials)

# save 
ppo_stats.save('ppo_stats_backup')
del ppo_stats

# load 
ppo_stats = AgentStats.load('ppo_stats_backup')

# continue previous optimization, now with 5s of timeout
best_trial, data  = ppo_stats.optimize_hyperparams(n_trials=10, timeout=5, n_sim=5, n_fit=2, n_jobs=2, 
                           continue_previous=True)

print("number of initial trials = ", initial_n_trials)
print("number of trials after continuing= ", len(ppo_stats.study.trials))


print("----")
print("fitting agents after choosing hyperparams...")
ppo_stats.fit()  # fit the 4 agents 


