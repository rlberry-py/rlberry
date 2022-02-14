""" 
 ===================== 
 Demo: demo_agent_manager_set_writer 
 =====================
"""
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents.torch.ppo import PPOAgent
from rlberry.manager import AgentManager, evaluate_agents
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # --------------------------------
    # Define training env
    # --------------------------------
    train_env = (PBall2D, dict())

    # -----------------------------
    # Parameters
    # -----------------------------
    N_EPISODES = 100
    GAMMA = 0.99
    HORIZON = 50

    params_ppo = {"gamma": GAMMA, "horizon": HORIZON, "learning_rate": 0.0003}

    eval_kwargs = dict(eval_horizon=HORIZON, n_simulations=20)

    # -----------------------------
    # Run AgentManager
    # -----------------------------
    ppo_stats = AgentManager(
        PPOAgent,
        train_env,
        fit_budget=N_EPISODES,
        init_kwargs=params_ppo,
        eval_kwargs=eval_kwargs,
        n_fit=4,
    )

    ppo_stats.set_writer(0, SummaryWriter, writer_kwargs={"comment": "worker_0"})
    ppo_stats.set_writer(1, SummaryWriter, writer_kwargs={"comment": "worker_1"})

    agent_manager_list = [ppo_stats]

    agent_manager_list[0].fit()
    agent_manager_list[
        0
    ].save()  # after fit, writers are set to None to avoid pickle problems.

    # compare final policies
    output = evaluate_agents(agent_manager_list)
    print(output)
