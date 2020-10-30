import rlberry.seeding as seeding
from rlberry.envs import PBall2D
from rlberry.agents import RSKernelUCBVIAgent, RSUCBVIAgent
from rlberry.agents.ppo import PPOAgent
from rlberry.wrappers import RescaleRewardWrapper 
from rlberry.eval.compare_policy import ComparePolicy


# global seed 
seeding.set_global_seed(123)

# ----------------------------
# Define list of agent classes
# ----------------------------
agents = [RSUCBVIAgent, RSKernelUCBVIAgent, PPOAgent]

# -----------------------------
# Parameters to fit each agent 
# -----------------------------
N_EPISODES         = 100
GAMMA              = 0.99
HORIZON            = 25
BONUS_SCALE_FACTOR = 0.1
MIN_DIST           = 0.1
VERBOSE            = 4

params_1 = {"n_episodes":N_EPISODES,
            "gamma":GAMMA,
            "horizon":HORIZON,
            "bonus_scale_factor":BONUS_SCALE_FACTOR,
            "min_dist":MIN_DIST,
            "verbose":VERBOSE}

params_2 = {"n_episodes":N_EPISODES,
            "gamma":GAMMA,
            "horizon":HORIZON,
            "bonus_scale_factor":BONUS_SCALE_FACTOR,
            "min_dist":MIN_DIST,
            "bandwidth":0.05,
            "beta": 1.0,
            "kernel_type": "gaussian",
            "verbose":VERBOSE}

params_3 = {"n_episodes":N_EPISODES,
            "gamma":GAMMA,
            "horizon":HORIZON,
            "lr": 0.0003}

agent_kwargs = [params_1, params_2, params_3]

# ----------------------------------
# Evaluation environment and horizon
# ----------------------------------
eval_env     = PBall2D()
eval_horizon = 50

# ---------------------
# Training environment 
# ---------------------
train_envs   = PBall2D()


# ---------------------
# Run evaluator and plot 
# ---------------------
evaluator = ComparePolicy(agents, eval_env, eval_horizon, train_envs, agent_kwargs, nsim=20, njobs=4, verbose=5)
evaluator.run()
print(evaluator.agents_rewards[-1])
print(evaluator.fitted_agents[-1].id)

evaluator.plot()


