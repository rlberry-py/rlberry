import os
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
from joblib import Parallel, delayed
from copy import deepcopy
from rlberry.envs import OnlineModel


def _fit_worker(args):
    agent_class, train_env, init_kwargs, fit_kwargs = args
    agent = agent_class(train_env, copy_env=False, reseed_env=False, **init_kwargs)
    agent.fit(**fit_kwargs)
    return agent


def _monte_carlo_worker(args):
    """
    Note: eval_env is reseeded. If eval_env is not a native rlberry environment,
    reseeding is not guaranteed to work properly.
    """
    trained_agent, policy_kwargs, eval_env, eval_horizon, nsim = args
    eval_env.reseed()
    episode_rewards = np.zeros(nsim)
    for sim in range(nsim):
        observation = eval_env.reset()
        for hh in range(eval_horizon):
            action = trained_agent.policy(observation, **policy_kwargs)
            observation, reward, done, _ = eval_env.step(action)
            if done:
                break
            episode_rewards[sim] += reward
    return episode_rewards


class ComparePolicy:
    """
    Class to evaluate the policy learned by a set of agents.
    The evaluation environment must be an instance of OnlineModel, that is,
    simplement reset() and step() functions.

    Assumption:
        * When the agent is initialized with an environment instance env, this env is
        deep copied and reseeded. This is done by default in native rlberry agents.

    Notes:
        * eval_env is deep copied before calling _monte_carlo_worker, and reseeded before evaluation
    """

    def __init__(self,
                 agents,
                 eval_env,
                 eval_horizon,
                 train_envs=None,
                 agent_kwargs=None,
                 fit_kwargs=None,
                 policy_kwargs=None,
                 nsim=5,
                 fitted=False,
                 njobs=4,
                 verbose=5):
        """
        Parameters
        ----------
        agents : list
            List of the classes of the agents to be evaluated. If fitted = True, list of trained agent instances.
        eval_env : OnlineModel
            Evaluation environment.
        eval_horizon : int
            For how many steps to run the policy of each agent.
        train_envs : list or rlberry.envs.Model
            List of environment instances used to train each agent. Must be given if fitted is False.
        agent_kwargs : list
            List of dictionaries contaning the keyword parameters required to call __init__ on each agent class.
            Must be given if fitted is False.
        fit_kwargs :  list 
            List of dictionaries containing the parameters required to call fit() on each agent.
        policy_kwargs : list 
            List of dictionaries containing the parameters required to call policy() on each agent
        nsim : int
            Number of Monte Carlo simulations to evaluate a policy
        fitted : bool
            True if the agents are already fitted.
        njobs : int 
            Number of jobs to train the agents in parallel using joblib.
        verbose : int 
            Verbosity level.
        """
        self.n_agents = len(agents)
        self.agents = agents
        self.eval_env = eval_env   
        self.eval_horizon = eval_horizon
        self.agent_kwargs = agent_kwargs
        self.fit_kwargs = fit_kwargs
        self.policy_kwargs = policy_kwargs
        self.nsim = nsim
        self.fitted = fitted
        self.njobs = njobs
        self.verbose = verbose

        # If fitted, agents is a list of trained agent instances.
        self.fitted_agents = []
        if fitted:
            self.fitted_agents = agents

        # check and initialize kwargs
        assert isinstance(
            self.eval_env, OnlineModel), "ComparePolicy: Evaluation environment must implement OnlineModel"
        if not fitted:
            assert train_envs is not None,   "ComparePolicy: If agents are not fitted, train_envs must be given."
            assert agent_kwargs is not None, "ComparePolicy: If agents are not fitted, agent_kwargs must be given."
            assert len(agent_kwargs) == self.n_agents
        if fit_kwargs is not None:
            assert self.n_agents == len(self.fit_kwargs)
        else:
            self.fit_kwargs = [dict() for ii in range(self.n_agents)]
        if policy_kwargs is not None:
            assert self.n_agents == len(self.policy_kwargs)
        else:
            self.policy_kwargs = [dict() for ii in range(self.n_agents)]

        # check train_envs
        if not isinstance(train_envs, list):
            # single training environment is given, convert to a list
            # by deep copying and reseeding
            self.train_envs = []
            for ii in range(len(self.agents)):
                _env = deepcopy(train_envs)
                _env.reseed()
                self.train_envs.append(_env)  
        else:
            assert len(train_envs) == self.n_agents
            self.train_envs = train_envs

        # List where to store rewards from each algorithm
        self.agents_rewards = []

        # Output DataFrame
        self.output = None

    def run(self):
        """
        Run evaluation
        """
        # fit each one of the agents
        if not self.fitted:
            self._fit_agents()
        # run monte carlo simulations, store results in self.agents_rewards
        self._eval_agents()
        
        # build unique agent IDs (in case there are two agents with the same ID)
        unique_ids = []
        id_count = {}
        for agent in self.fitted_agents:
            if agent.name not in id_count:
                id_count[agent.name] = 1
            else:
                id_count[agent.name] += 1
            
            unique_ids.append(agent.name + "*"*(id_count[agent.name]-1))

        # build output
        data = {}
        for agent_id, agent_rewards in zip(unique_ids, self.agents_rewards):
            data[agent_id] = agent_rewards
        self.output = pd.DataFrame(data)
    
    def plot(self, show=True):
        if self.output is None:
            print("No output to be plotted.")
            return 
        
        with sns.axes_style("whitegrid"):
            ax = sns.boxplot(data=self.output)
            ax.set_xlabel("agent")
            ax.set_ylabel("rewards in one episode")
            plt.title("Environment = %s"%self.eval_env.name)
            if show:
                plt.show()


    def _fit_agents(self):
        if self.verbose > 0:
            print("\n Training agents... \n")
        args = [(agent_class, train_env, agent_init_kwargs, agent_fit_kwargs)
                for agent_class, train_env, agent_init_kwargs, agent_fit_kwargs in zip(self.agents, self.train_envs, self.agent_kwargs, self.fit_kwargs)]

        self.fitted_agents = Parallel(n_jobs=self.njobs, verbose=self.verbose)(
            delayed(_fit_worker)(arg) for arg in args)

        if self.verbose > 0:
            print("\n ... agents trained! \n")

    def _eval_agents(self):
        if self.verbose > 0:
            print("\n Evaluating agents... \n")

        args = [(trained_agent, policy_kwargs, deepcopy(self.eval_env), self.eval_horizon, self.nsim)
                for trained_agent, policy_kwargs in zip(self.fitted_agents, self.policy_kwargs)]

        self.agents_rewards = Parallel(n_jobs=self.njobs, verbose=self.verbose)(
            delayed(_monte_carlo_worker)(arg) for arg in args)

        if self.verbose > 0:
            print("\n ... agents evaluated! \n")
