import logging
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import rlberry.seeding as seeding

logger = logging.getLogger(__name__)


def plot_episode_rewards(agent_stats_list, cumulative=False,
                         fignum=None, show=True):
    plt.figure(fignum)
    for agent_stats in agent_stats_list:
        if 'episode_rewards' not in agent_stats.fit_info:
            logger.warning("episode_rewards not \
available for %s." % agent_stats.agent_name)
            continue
        else:
            # train agents if they are not already trained
            if agent_stats.fitted_agents is None:
                agent_stats.fit()
            # get reward statistics and plot them
            rewards = np.array(agent_stats.fit_statistics['episode_rewards'])
            if cumulative:
                rewards = np.cumsum(rewards, axis=1)
            mean_r = rewards.mean(axis=0)
            std_r = rewards.std(axis=0)
            episodes = np.arange(1, rewards.shape[1]+1)

            plt.plot(episodes, mean_r, label=agent_stats.agent_name)
            plt.fill_between(episodes, mean_r-std_r, mean_r+std_r, alpha=0.2)
            plt.legend()
            plt.xlabel("episodes")
            if not cumulative:
                plt.ylabel("reward in one episode")
            else:
                plt.ylabel("total reward")
            plt.grid(True, alpha=0.75)

    if show:
        plt.show()


def compare_policies(agent_stats_list, eval_env=None, eval_horizon=None,
                     stationary_policy=True, n_sim=10, fignum=None,
                     show=True, plot=True, **kwargs):
    """
    Compare the policies of each of the agents in agent_stats_list.
    Each element of the agent_stats_list contains a list of fitted agents.
    To evaluate the policy, we repeat n_sim times:
        * choose one of the fitted agents uniformly at random
        * run its policy in eval_env for eval_horizon time steps

    To do
    ------
    Paralellize evaluations of each agent.

    Parameters
    ----------
    agent_stats_list : list of AgentStats objects.
    eval_env : Model
        Environment where to evaluate the policies.
        If None, it is taken from AgentStats.
    eval_horizon : int
        Number of time steps for policy evaluation.
        If None, it is taken from AgentStats.
    stationary_policy : bool
        If False, the time step h (0<= h <= eval_horizon) is sent as input
        to agent.policy() for policy evaluation.
    n_sim : int
        Number of simulations to evaluate each policy.
    fignum: string or int
        Identifier of plot figure
    show: bool
        If true, calls plt.show()
    plot: bool
        If false, do not plot.
    """
    #
    # evaluation
    #
    use_eval_from_agent_stats = (eval_env is None)
    use_horizon_from_agent_stats = (eval_horizon is None)

    rng = seeding.get_rng()
    agents_rewards = []
    for agent_stats in agent_stats_list:
        # train agents if they are not already trained
        if agent_stats.fitted_agents is None:
            agent_stats.fit()

        # eval env and horizon
        if use_eval_from_agent_stats:
            eval_env = agent_stats.eval_env
            assert eval_env is not None, \
                "eval_env not in AgentStats %s" % agent_stats.agent_name
        if use_horizon_from_agent_stats:
            eval_horizon = agent_stats.eval_horizon
            assert eval_horizon is not None, \
                "eval_horizon not in AgentStats %s" % agent_stats.agent_name

        # evaluate agent
        episode_rewards = np.zeros(n_sim)
        for sim in range(n_sim):
            # choose one of the fitted agents randomly
            agent_idx = rng.integers(len(agent_stats.fitted_agents))
            agent = agent_stats.fitted_agents[agent_idx]
            # evaluate agent
            observation = eval_env.reset()
            for hh in range(eval_horizon):
                if stationary_policy:
                    action = agent.policy(observation,
                                          **agent_stats.policy_kwargs)
                else:
                    action = agent.policy(observation, hh,
                                          **agent_stats.policy_kwargs)
                observation, reward, done, _ = eval_env.step(action)
                episode_rewards[sim] += reward
                if done:
                    break
        # store rewards
        agents_rewards.append(episode_rewards)

    #
    # plot
    #

    # build unique agent IDs (in case there are two agents with the same ID)
    unique_ids = []
    id_count = {}
    for agent_stats in agent_stats_list:
        name = agent_stats.agent_name
        if name not in id_count:
            id_count[name] = 1
        else:
            id_count[name] += 1

        unique_ids.append(name + "*"*(id_count[name]-1))

    # convert output to DataFrame
    data = {}
    for agent_id, agent_rewards in zip(unique_ids, agents_rewards):
        data[agent_id] = agent_rewards
    output = pd.DataFrame(data)

    # plot
    if plot:
        plt.figure(fignum)

        with sns.axes_style("whitegrid"):
            ax = sns.boxplot(data=output, **kwargs)
            ax.set_xlabel("agent")
            ax.set_ylabel("rewards in one episode")
            plt.title("Environment = %s" % eval_env.unwrapped.name)
            if show:
                plt.show()

    return output

