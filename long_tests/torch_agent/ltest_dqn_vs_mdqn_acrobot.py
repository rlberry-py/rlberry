from rlberry.envs import gym_make
from rlberry.agents.torch import DQNAgent
from rlberry.agents.torch import MunchausenDQNAgent as MDQNAgent
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
import matplotlib.pyplot as plt
import seaborn as sns


def test_dqn_vs_mdqn_acro():
    """
    Long test to verify dqn and mdqn perform similary on acrobot.
    Losses, Rewards during training, and Evaluations are saved as pdfs.

    Results of this test at the time of writing can be found in
    the following pull request: https://github.com/rlberry-py/rlberry/pull/266
    """
    env_ctor = gym_make
    env_kwargs = dict(id="Acrobot-v1")

    dqn_init_kwargs = dict(
        gamma=0.99,
        batch_size=32,
        chunk_size=8,
        lambda_=0.5,
        target_update_parameter=0.005,
        learning_rate=1e-3,
        epsilon_init=1.0,
        epsilon_final=0.1,
        epsilon_decay_interval=20_000,
        train_interval=10,
        gradient_steps=-1,
        max_replay_size=200_000,
        learning_starts=5_000,
    )

    mdqn_init_kwargs = dict(
        gamma=0.99,
        batch_size=32,
        chunk_size=8,
        lambda_=0.5,
        target_update_parameter=0.005,
        learning_rate=1e-3,
        epsilon_init=1.0,
        epsilon_final=0.1,
        epsilon_decay_interval=20_000,
        train_interval=10,
        gradient_steps=-1,
        max_replay_size=200_000,
        learning_starts=5_000,
    )

    dqnagent = AgentManager(
        DQNAgent,
        (env_ctor, env_kwargs),
        init_kwargs=dqn_init_kwargs,
        fit_budget=5e4,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=4,
        parallelization="process",
        mp_context="fork",
    )

    mdqnagent = AgentManager(
        MDQNAgent,
        (env_ctor, env_kwargs),
        init_kwargs=mdqn_init_kwargs,
        fit_budget=5e4,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=4,
        parallelization="process",
        mp_context="fork",
    )

    mdqnagent.fit()
    dqnagent.fit()
    plot_writer_data(
        [mdqnagent, dqnagent],
        tag="episode_rewards",
        # ylabel_="Cumulative Reward",
        title=" Rewards during training",
        show=False,
        savefig_fname="mdqn_acro_rewards.pdf",
    )
    plt.clf()
    plot_writer_data(
        [mdqnagent, dqnagent],
        tag="losses/q_loss",
        # ylabel_="Cumulative Reward",
        title="q_loss",
        show=False,
        savefig_fname="mdqn_acro_loss.pdf",
    )
    plt.clf()
    evaluation = evaluate_agents([mdqnagent, dqnagent], n_simulations=100, show=False)
    with sns.axes_style("whitegrid"):
        ax = sns.boxplot(data=evaluation)
        ax.set_xlabel("agent")
        ax.set_ylabel("Cumulative Reward")
    plt.title("Evals")
    plt.gcf().savefig("mdqn_acro_eval.pdf")
    plt.clf()
