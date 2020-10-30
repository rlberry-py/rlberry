from copy import deepcopy
from joblib import Parallel, delayed
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AgentStats:
    """
    Class to train, evaluate and gather statistics about an agent.
    """

    def __init__(self, agent_class, train_env, eval_env=None, init_kwargs={}, fit_kwargs={}, nfit=4, njobs=4, verbose=5):
        """
        Parameters
        ----------
        agent_class 
            Class of the agent.
        train_env : Model
            Enviroment used to initialize/train the agent.
        eval_env : Model
            Environment used to evaluate the agent. If None, set to a reseeded deep copy of train_env.
        init_kwargs : dict 
            Arguments required by the agent's constructor.
        fit_kwargs : dict 
            Arguments required to train the agent.
        nfit : int
            Number of agent instances to fit.
        njobs : int 
            Number of jobs to train the agents in parallel using joblib.
        verbose : int 
            Verbosity level.
        """
        self.agent_name = agent_class.name
        self.fit_info = agent_class.fit_info

        self.agent_class = agent_class
        self.train_env = train_env
        if eval_env is None:
            self.eval_env = deepcopy(train_env)
            self.eval_env.reseed()
        else:
            self.eval_env = deepcopy(eval_env)
            self.eval_env.reseed()
        self.init_kwargs = deepcopy(init_kwargs)
        self.fit_kwargs = deepcopy(fit_kwargs)
        self.nfit = nfit
        self.njobs = njobs
        self.verbose = verbose

        # Create environment copies for training
        self.train_env_set = []
        for ii in range(nfit):
            _env = deepcopy(train_env)
            _env.reseed()
            self.train_env_set.append(_env)

        #
        self.fitted_agents = []
        self.fit_statistics = {}

    def fit(self):
        if self.verbose > 0:
            print("\n Training AgentStats for %s... \n" % self.agent_name)
        args = [(self.agent_class, train_env, self.init_kwargs, self.fit_kwargs)
                for train_env in self.train_env_set]

        workers_output = Parallel(n_jobs=self.njobs, verbose=self.verbose)(
            delayed(_fit_worker)(arg) for arg in args)

        self.fitted_agents, stats = (
            [i for i, j in workers_output],
            [j for i, j in workers_output])

        if self.verbose > 0:
            print("\n ... trained! \n")

        # gather all stats in a dictionary
        for entry in self.fit_info:
            self.fit_statistics[entry] = []
            for stat in stats:
                self.fit_statistics[entry].append(stat[entry])


def _fit_worker(args):
    agent_class, train_env, init_kwargs, fit_kwargs = args
    agent = agent_class(train_env, copy_env=False,
                        reseed_env=False, **init_kwargs)
    info = agent.fit(**fit_kwargs)
    return agent, info


def plot_episode_rewards(stats_list, cumulative=False, fignum=None, show=True):
    plt.figure(fignum)
    for agent_stats in stats_list:
        if not 'episode_rewards' in agent_stats.fit_info:
            logging.warning("episode_rewards not available for %s."%agent_stats.agent_name)
            continue 
        else:
            rewards = np.array(agent_stats.fit_statistics['episode_rewards'])
            if cumulative:
                rewards = np.cumsum(rewards, axis=1)
            mean_r  = rewards.mean(axis=0)
            std_r   = rewards.std(axis=0)
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