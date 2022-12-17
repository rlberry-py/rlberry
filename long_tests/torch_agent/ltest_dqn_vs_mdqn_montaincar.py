from rlberry.envs import gym_make
from rlberry.agents.torch import DQNAgent
from rlberry.agents.torch import MunchausenDQNAgent as MDQNAgent
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


model_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": (256, 256),
    "reshape": False,
}
# hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
def test_dqn_vs_mdqn_montaincar():

    env_ctor = gym_make
    env_kwargs = dict(id="MountainCar-v0")

    dqnagent = AgentManager(
        DQNAgent,
        (env_ctor, env_kwargs),
        init_kwargs=dict(
            q_net_constructor="rlberry.agents.torch.utils.training.model_factory_from_env",
            q_net_kwargs=model_configs,
            batch_size=128,
            max_replay_size=10000,
            learning_rate=4e-3,
            learning_starts=1000,
            gamma=0.98,
            train_interval=16,
            gradient_steps=8,
            epsilon_init=0.2,
            epsilon_final=0.07,
            epsilon_decay_interval=600,
        ),
        fit_budget=1e5,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=4,
        parallelization="process",
        mp_context="fork",
        seed=42,
    )

    mdqnagent = AgentManager(
        MDQNAgent,
        (env_ctor, env_kwargs),
        init_kwargs=dict(
            q_net_constructor="rlberry.agents.torch.utils.training.model_factory_from_env",
            q_net_kwargs=model_configs,
            batch_size=128,
            max_replay_size=10000,
            learning_rate=4e-3,
            learning_starts=1000,
            gamma=0.98,
            train_interval=16,
            gradient_steps=8,
            epsilon_init=0.2,
            epsilon_final=0.07,
            epsilon_decay_interval=600,
        ),
        fit_budget=1e5,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=4,
        parallelization="process",
        mp_context="fork",
        seed=42,
    )

    dqnagent.fit()
    mdqnagent.fit()
    plot_writer_data(
        [dqnagent, mdqnagent],
        tag="episode_rewards",
        # ylabel_="Cumulative Reward",
        title=" Rewards during training",
        show=False,
        savefig_fname="rewards.pdf",
    )
    plt.clf()
    plot_writer_data(
        [dqnagent, mdqnagent],
        tag="losses/q_loss",
        # ylabel_="Cumulative Reward",
        title="q_loss",
        show=False,
        savefig_fname="losses.pdf",
    )
    plt.clf()
    evaluation = evaluate_agents([dqnagent, mdqnagent], n_simulations=100, show=False)
    with sns.axes_style("whitegrid"):
        ax = sns.boxplot(data=evaluation)
        ax.set_xlabel("agent")
        ax.set_ylabel("Cumulative Reward")
    plt.title("Evals")
    plt.gcf().savefig("eval.pdf")
    plt.clf()
