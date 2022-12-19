from rlberry.envs import gym_make
from rlberry.agents.torch import MunchausenDQNAgent as MDQNAgent
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
import matplotlib.pyplot as plt
import seaborn as sns


env_ctor = gym_make
env_kwargs = dict(id="Acrobot-v1")

mdqnagent = AgentManager(
    MDQNAgent,
    (env_ctor, env_kwargs),
    init_kwargs=dict(target_update_parameter=0.005),
    fit_budget=1e5,
    eval_kwargs=dict(eval_horizon=500),
    n_fit=1,
    parallelization="process",
    mp_context="fork",
    seed=42,
)


mdqnagent.fit()
plot_writer_data(
    [mdqnagent],
    tag="episode_rewards",
    # ylabel_="Cumulative Reward",
    title=" Rewards during training",
    show=False,
    savefig_fname="mdqn_acro_rewards.pdf",
)
plt.clf()
plot_writer_data(
    [mdqnagent],
    tag="losses/q_loss",
    # ylabel_="Cumulative Reward",
    title="q_loss",
    show=False,
    savefig_fname="mdqn_acro_loss.pdf",
)
plt.clf()
evaluation = evaluate_agents([mdqnagent], n_simulations=100, show=False)
with sns.axes_style("whitegrid"):
    ax = sns.boxplot(data=evaluation)
    ax.set_xlabel("agent")
    ax.set_ylabel("Cumulative Reward")
plt.title("Evals")
plt.gcf().savefig("mdqn_acro_eval.pdf")
plt.clf()
