import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import cycle


logger = logging.getLogger(__name__)


def mc_policy_evaluation(agent,
                         eval_env,
                         eval_horizon=10**5,
                         n_sim=10,
                         gamma=1.0,
                         policy_kwargs=None,
                         stationary_policy=True):
    """
    Monte-Carlo Policy evaluation [1]_ of an agent to estimate the value at the initial state.

    If a list of agents is provided as input, for each evaluation, one of the agents is sampled
    uniformly at random.

    Parameters
    ----------
    agent : Agent or list of agents.
        Trained agent(s).
    eval_env : Env
        Evaluation environment.
    eval_horizon : int, default: 10**5
        Horizon, maximum episode length.
    n_sim : int, default: 10
        Number of Monte Carlo simulations.
    gamma : double, default: 1.0
        Discount factor.
    policy_kwargs : dict or None
        Optional kwargs for agent.policy() method.
    stationary_policy : bool, default: True
        If False, the time step h (0<= h <= eval_horizon) is sent as argument
        to agent.policy() for policy evaluation.

    Return
    ------
    Numpy array of shape (n_sim, ) containing the sum of rewards in each simulation.

    References
    ----------
    .. [1] http://incompleteideas.net/book/first/ebook/node50.html
    """
    if not isinstance(agent, list):
        agents = [agent]
    else:
        agents = agent

    rng = agents[0].seeder.rng

    policy_kwargs = policy_kwargs or {}

    episode_rewards = np.zeros(n_sim)
    for sim in range(n_sim):
        idx = rng.integers(len(agents))

        observation = eval_env.reset()
        for hh in range(eval_horizon):
            if stationary_policy:
                action = agents[idx].policy(observation, **policy_kwargs)
            else:
                action = agents[idx].policy(observation, hh, **policy_kwargs)
            observation, reward, done, _ = eval_env.step(action)
            episode_rewards[sim] += reward * np.power(gamma, hh)
            if done:
                break

    return episode_rewards


def plot_episode_rewards(agent_stats,
                         cumulative=False,
                         fignum=None,
                         show=True,
                         max_value=None,
                         plot_regret=False,
                         grid=True):
    """
    Given a list of AgentStats, plot the rewards obtained in each episode.
    The dictionary returned by agents' .fit() method must contain a key 'episode_rewards'.

    Parameters
    ----------
    agent_stats : AgentStats, or list of AgentStats
    cumulative : bool, default: False
        If true, plot cumulative rewards.
    fignum: string or int
        Identifier of plot figure.
    show: bool
        If true, calls plt.show().
    max_value: double, default: None
        Maximum reward achievable in one episode.
    plot_regret: bool, default: False
        If true, plots the regret. Requires max_val to be given.
    grid: bool, default: True
        If False, disable grid in plot.
    """
    agent_stats_list = agent_stats
    if not isinstance(agent_stats_list, list):
        agent_stats_list = [agent_stats_list]

    if plot_regret and max_value is None:
        raise ValueError("max_value must be provided for regret plot")

    # line style
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)

    plt.figure(fignum)
    for stats in agent_stats_list:
        # train agents if they are not already trained
        if stats.fitted_agents is None:
            stats.fit()

        if 'episode_rewards' not in stats.writer_data:
            logger.warning("episode_rewards not available for %s." % stats.agent_name)
            continue

        # get reward statistics and plot them
        rewards = np.array(stats.writer_data['episode_rewards'])
        if cumulative and (not plot_regret):
            data = np.cumsum(rewards, axis=1)
            label = "total reward"
        elif plot_regret:
            data = np.cumsum(max_value-rewards, axis=1)
            label = "regret"
        else:
            data = rewards
            label = "reward in one episode"

        mean_r = data.mean(axis=0)
        std_r = data.std(axis=0)
        episodes = np.arange(1, data.shape[1]+1)

        plt.plot(episodes, mean_r, next(linecycler), label=stats.agent_name)
        plt.fill_between(episodes, mean_r-std_r, mean_r+std_r, alpha=0.4)
        plt.legend()
        plt.xlabel("episodes")
        plt.ylabel(label)
        if grid:
            plt.grid(True, alpha=0.75)

    if show:
        plt.show()


def compare_policies(agent_stats_list,
                     eval_env=None,
                     eval_horizon=None,
                     stationary_policy=True,
                     n_sim=10,
                     fignum=None,
                     show=True,
                     plot=True,
                     **kwargs):
    """
    Compare the policies of each of the agents in agent_stats_list.
    Each element of the agent_stats_list contains a list of fitted agents.
    To evaluate the policy, we repeat n_sim times:
        * choose one of the fitted agents uniformly at random
        * run its policy in eval_env for eval_horizon time steps


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
        If False, the time step h (0<= h <= eval_horizon) is sent as argument
        to agent.policy() for policy evaluation.
    n_sim : int
        Number of simulations to evaluate each policy.
    fignum: string or int
        Identifier of plot figure.
    show: bool
        If true, calls plt.show().
    plot: bool
        If false, do not plot.
    kwargs:
        Extra parameters for sns.boxplot
    """
    #
    # evaluation
    #
    use_eval_from_agent_stats = (eval_env is None)
    use_horizon_from_agent_stats = (eval_horizon is None)

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

        # get rng from agent_stats
        rng = agent_stats.seeder.rng

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
            plt.title("Environment = %s" %
                      getattr(eval_env.unwrapped, "name",
                              eval_env.unwrapped.__class__.__name__))
            if show:
                plt.show()

    return output


def plot_writer_data(agent_stats,
                     tag,
                     fignum=None,
                     show=True,
                     grid=True):
    """
    Given a list of AgentStats, plot data (corresponding to info) obtained in each episode.
    The dictionary returned by agents' .fit() method must contain a key equal to `info`.

    Parameters
    ----------
    agent_stats : AgentStats, or list of AgentStats
    tag : str
        Tag of data to plot.
    fignum: string or int
        Identifier of plot figure.
    show: bool
        If true, calls plt.show().
    grid: bool, default: True
        If False, disable grid in plot.
    """
    agent_stats_list = agent_stats
    if not isinstance(agent_stats_list, list):
        agent_stats_list = [agent_stats_list]

    # line style
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    plt.figure(fignum)
    for stats in agent_stats_list:
        # train agents if they are not already trained
        if stats.fitted_agents is None:
            stats.fit()

        if tag not in stats.writer_data:
            logger.warning("{} not available for {}.".format(tag, stats.agent_name))
            continue

        # get data and plot them
        data = np.array(stats.writer_data[tag])
        mean_data = data.mean(axis=0)
        std_data = data.std(axis=0)
        episodes = np.arange(1, data.shape[1]+1)

        plt.plot(episodes, mean_data, next(linecycler), label=stats.agent_name)
        plt.fill_between(episodes, mean_data-std_data, mean_data+std_data, alpha=0.4)
        plt.legend()
        plt.xlabel("episodes")
        plt.ylabel(tag)
        if grid:
            plt.grid(True, alpha=0.75)

    if show:
        plt.show()
