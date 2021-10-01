import numpy as np
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents import RSKernelUCBVIAgent, RSUCBVIAgent
from rlberry.agents.torch.a2c import A2CAgent
from rlberry.manager import AgentManager, plot_writer_data, evaluate_agents
from rlberry.seeding import set_external_seed


if __name__ == '__main__':
    set_external_seed(123)

    # --------------------------------
    # Define train and evaluation envs
    # --------------------------------
    train_env = (PBall2D, dict())
    eval_env = (PBall2D, dict())

    # -----------------------------
    # Parameters
    # -----------------------------
    N_EPISODES = 250
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

    params_a2c = {"gamma": GAMMA,
                  "horizon": HORIZON,
                  "learning_rate": 0.0003}

    eval_kwargs = dict(eval_horizon=HORIZON, n_simulations=20)

    # -----------------------------
    # Run AgentManager
    # -----------------------------
    rsucbvi_stats = AgentManager(
        RSUCBVIAgent,
        train_env,
        fit_budget=N_EPISODES,
        init_kwargs=params,
        eval_kwargs=eval_kwargs,
        n_fit=4,
        seed=123)
    rskernel_stats = AgentManager(
        RSKernelUCBVIAgent,
        train_env,
        fit_budget=N_EPISODES,
        init_kwargs=params_kernel,
        eval_kwargs=eval_kwargs,
        n_fit=4,
        seed=123)
    a2c_stats = AgentManager(
        A2CAgent,
        train_env,
        fit_budget=N_EPISODES,
        init_kwargs=params_a2c,
        eval_kwargs=eval_kwargs,
        n_fit=4,
        seed=123,
        parallelization='process')

    agent_manager_list = [rsucbvi_stats, rskernel_stats, a2c_stats]

    for st in agent_manager_list:
        st.fit()

    # learning curves
    plot_writer_data(agent_manager_list,
                     tag='episode_rewards',
                     preprocess_func=np.cumsum,
                     title='cumulative rewards',
                     show=False)

    # compare final policies
    output = evaluate_agents(agent_manager_list)
    print(output)

    for st in agent_manager_list:
        st.clear_output_dir()
