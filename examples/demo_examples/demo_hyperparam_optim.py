"""
 =====================
 Demo: demo_hyperparam_optim
 =====================
"""
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents.torch import REINFORCEAgent
from rlberry.manager import AgentManager

if __name__ == "__main__":
    # --------------------------------
    # Define train and evaluation envs
    # --------------------------------
    train_env = (PBall2D, None)

    # -----------------------------
    # Parameters
    # -----------------------------
    N_EPISODES = 10
    GAMMA = 0.99
    HORIZON = 50
    BONUS_SCALE_FACTOR = 0.1
    MIN_DIST = 0.1

    params = {"gamma": GAMMA, "horizon": HORIZON, "learning_rate": 0.0003}

    eval_kwargs = dict(eval_horizon=HORIZON, n_simulations=20)

    # -------------------------------
    # Run AgentManager and save results
    # --------------------------------
    manager = AgentManager(
        REINFORCEAgent,
        train_env,
        fit_budget=N_EPISODES,
        init_kwargs=params,
        eval_kwargs=eval_kwargs,
        n_fit=4,
    )

    # hyperparam optim with multiple threads
    manager.optimize_hyperparams(
        n_trials=5,
        timeout=None,
        n_fit=2,
        sampler_method="optuna_default",
        optuna_parallelization="thread",
    )

    initial_n_trials = len(manager.optuna_study.trials)

    # save
    manager_fname = manager.save()
    del manager

    # load
    manager = AgentManager.load(manager_fname)

    # continue previous optimization, now with 120s of timeout and multiprocessing
    manager.optimize_hyperparams(
        n_trials=512,
        timeout=120,
        n_fit=8,
        continue_previous=True,
        optuna_parallelization="process",
        n_optuna_workers=4,
    )

    print("number of initial trials = ", initial_n_trials)
    print("number of trials after continuing= ", len(manager.optuna_study.trials))

    print("----")
    print("fitting agents after choosing hyperparams...")
    manager.fit()  # fit the 4 agents
