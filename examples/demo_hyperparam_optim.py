from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents.torch.ppo import PPOAgent
from rlberry.manager import AgentManager

if __name__ == '__main__':
    # --------------------------------
    # Define train and evaluation envs
    # --------------------------------
    train_env = (PBall2D, None)

    # -----------------------------
    # Parameters
    # -----------------------------
    N_EPISODES = 100
    GAMMA = 0.99
    HORIZON = 50
    BONUS_SCALE_FACTOR = 0.1
    MIN_DIST = 0.1

    params_ppo = {"gamma": GAMMA,
                  "horizon": HORIZON,
                  "learning_rate": 0.0003}

    eval_kwargs = dict(eval_horizon=HORIZON, n_simulations=20)

    # -------------------------------
    # Run AgentManager and save results
    # --------------------------------
    ppo_stats = AgentManager(
        PPOAgent, train_env, fit_budget=N_EPISODES,
        init_kwargs=params_ppo,
        eval_kwargs=eval_kwargs,
        n_fit=4,
        output_dir='dev/')

    # hyperparam optim with multiple threads
    ppo_stats.optimize_hyperparams(
        n_trials=5, timeout=None,
        n_fit=2,
        sampler_method='optuna_default',
        optuna_parallelization='thread')

    initial_n_trials = len(ppo_stats.optuna_study.trials)

    # save
    ppo_stats_fname = ppo_stats.save()
    del ppo_stats

    # load
    ppo_stats = AgentManager.load(ppo_stats_fname)

    # continue previous optimization, now with 120s of timeout and multiprocessing
    ppo_stats.optimize_hyperparams(
        n_trials=512, timeout=120,
        n_fit=2,
        continue_previous=True,
        optuna_parallelization='process')

    print("number of initial trials = ", initial_n_trials)
    print("number of trials after continuing= ", len(ppo_stats.optuna_study.trials))

    print("----")
    print("fitting agents after choosing hyperparams...")
    ppo_stats.fit()  # fit the 4 agents
