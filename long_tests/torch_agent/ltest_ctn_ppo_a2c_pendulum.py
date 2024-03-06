from rlberry.envs import gym_make
from rlberry_research.agents.torch import A2CAgent, PPOAgent
from rlberry.manager import ExperimentManager, plot_writer_data, evaluate_agents
import matplotlib.pyplot as plt


def test_a2c_vs_ppo_pendul():
    """
    Long test to check that learning happens with ctn actions.
    We use default a2c hyps and sb3 zoo hyps for ppo.
    Results are saved in pdfs.
    """

    env_ctor = gym_make
    env_kwargs = dict(id="Pendulum-v1")

    a2c_xp_manager = ExperimentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        agent_name="A2CAgent",
        fit_budget=1e5,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=1,
        seed=42,
    )

    ppo_init_kwargs = dict(
        n_steps=1024,
        gamma=0.9,
        learning_rate=0.001,
        k_epochs=10,
    )
    ppo_xp_manager = ExperimentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        init_kwargs=ppo_init_kwargs,
        agent_name="PPOAgent",
        fit_budget=1e5,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=1,
        seed=42,
    )
    a2c_xp_manager.fit()
    ppo_xp_manager.fit()
    plot_writer_data(
        [a2c_xp_manager, ppo_xp_manager],
        tag="episode_rewards",
        # ylabel_="Cumulative Reward",
        title=" Rewards during training",
        show=False,
        savefig_fname="a2c_ppo_pendul_rewards.pdf",
    )
    plt.clf()
    evaluation = evaluate_agents(
        [a2c_xp_manager, ppo_xp_manager], n_simulations=100, show=False
    )
    ax = plt.boxplot(data=evaluation)
    ax.set_xlabel("agent")
    ax.set_ylabel("Cumulative Reward")
    plt.title("Evals")
    plt.gcf().savefig("a2c_ppo_pendul_eval.pdf")
    plt.clf()
