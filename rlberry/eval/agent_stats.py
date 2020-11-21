from copy import deepcopy
from joblib import Parallel, delayed
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import signal
import rlberry.seeding as seeding

_OPTUNA_INSTALLED = True 
try:
    import optuna 
except:
    _OPTUNA_INSTALLED = False



#
# Main class 
#

class AgentStats:
    """
    Class to train, optimize hyperparameters, evaluate and gather statistics about an agent.
    """
    def __init__(self, 
                 agent_class, 
                 train_env, 
                 eval_env=None, 
                 eval_horizon=None,
                 init_kwargs={}, 
                 fit_kwargs={}, 
                 policy_kwargs={}, 
                 agent_name=None,
                 nfit=4, 
                 njobs=4, 
                 verbose=5):
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
        policy_kwargs : dict 
            Arguments required to call agent.policy().
        agent_name : str
            Name of the agent. If None, set to agent_class.name
        nfit : int
            Number of agent instances to fit.
        njobs : int 
            Number of jobs to train the agents in parallel using joblib.
        verbose : int 
            Verbosity level.
        """
        # agent_class should only be None when the constructor is called 
        # by the class method AgentStats.load(), since the agent class will be loaded.
        if agent_class is not None: 

            self.agent_name = agent_name
            if agent_name is None:
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

            self.eval_horizon = eval_horizon
            # init and fit kwargs are deep copied in fit()
            self.init_kwargs = init_kwargs
            self.fit_kwargs =  fit_kwargs
            self.policy_kwargs = deepcopy(policy_kwargs)
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
            self.fitted_agents = None
            self.fit_statistics = {}

            # 
            self.rng = seeding.get_rng()

    def fit(self):
        if self.verbose > 0:
            print("\n Training AgentStats for %s... \n" % self.agent_name)
        args = [(self.agent_class, train_env, deepcopy(self.init_kwargs), deepcopy(self.fit_kwargs))
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

    def save(self, filename, **kwargs):
        """
        Parameters
        ----------
        filename : string
            filename with .pickle extension 
        """
        if filename[-7:] != '.pickle':
            filename += '.pickle'
            
        with open(filename, 'wb') as ff:
            pickle.dump(self.__dict__, ff)
        
    
    @classmethod
    def load(cls, filename):
        if filename[-7:] != '.pickle':
            filename += '.pickle'

        obj = cls(None, None)
        with open(filename, 'rb') as ff:
            tmp_dict = pickle.load(ff)
        obj.__dict__.clear()
        obj.__dict__.update(tmp_dict)
        return obj


    def hyperparam_optim(self, ntrials=5, max_time=60, nsim=5, nfit=2, njobs=2, 
                         sampler_method='random', pruner_method='halving'):
        """ 
        Run hyperparameter optimization and updates init_kwargs with the best hyperparameters found.

        Note: max_time probably only works in Unix systems, set to None in other systems.

        Parameters
        ----------
        ntrials: int 
            Mumber of agent evaluations 
        max_time: int 
            Maximum time (in seconds) to run optimization. Set to None for unlimited time.
        nsim : int 
            Number of Monte Carlo simulations to evaluate a policy.
        nfit: int
            Number of agents to fit for each hyperparam evaluation.
        njobs: int
            Number of jobs to fit agents for each hyperparam evaluation, and also the number of jobs of optuna.
        sampler_method : str
            Optuna sampling method.
        pruner_method : str
            Optuna pruner method.
        """
        global _OPTUNA_INSTALLED
        if not _OPTUNA_INSTALLED:
            logging.warning("Optuna not installed, hyperparam_optim() cannot be run.")
            return 
        
        assert self.eval_horizon is not None, "To use hyperparam_optim(), eval_horizon must be given to AgentStats."

        # get sampler
        if sampler_method == 'random':
            sampler = optuna.samplers.RandomSampler(seed=self.rng.integers(2**16))
        else:
            raise NotImplementedError("Sampler method %s is not implemented."%sampler_method)
        
        # get pruner 
        if pruner_method == 'halving':
            pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        else:
            raise NotImplementedError("Pruner method %s is not implemented."%pruner_method)

        # optuna study
        study = optuna.create_study(sampler=sampler, pruner=pruner)

        def objective(trial):
            kwargs = deepcopy(self.init_kwargs)

            # will raise exception if sample_parameters() is not implemented by the agent class
            kwargs.update( self.agent_class.sample_parameters(trial) )
   
            #
            # fit and evaluate agents
            # 
            # Create AgentStats with hyperparams
            params_stats = AgentStats(self.agent_class,
                                        deepcopy(self.train_env),
                                        init_kwargs=kwargs,   # kwargs are being optimized
                                        fit_kwargs=deepcopy(self.fit_kwargs),
                                        policy_kwargs=deepcopy(self.policy_kwargs),
                                        agent_name='optim',
                                        nfit=nfit,
                                        njobs=njobs,
                                        verbose=0)
            
            # Fit and evaluate params_stats
            params_stats.fit()

            # Get rewards 
            params_eval_env = deepcopy(self.eval_env)
            params_eval_env.reseed()

            eval_result = compare_policies([params_stats], 
                                            eval_env=params_eval_env, 
                                            eval_horizon=self.eval_horizon, 
                                            stationary_policy=True,
                                            nsim=nsim, 
                                            plot=False)               

            rewards = eval_result['optim'].values.mean()

            return rewards


        # run with timeout
        if max_time is not None:
            signal.signal(signal.SIGALRM, _signal_timeout_handler)
            signal.alarm(max_time)

        try:
            study.optimize(objective, n_trials=ntrials, n_jobs=njobs)
        except:
            logging.warning("Evaluation timed out!")
        finally:
            # disable alarm
            if max_time is not None:   
                try:
                    signal.alarm(0)
                except:
                    pass

        # continue
        best_trial = study.best_trial

        if self.verbose > 0:
            print('Number of finished trials: ', len(study.trials))

            print('Best trial:')

            print('Value: ', best_trial.value)

            print('Params: ')
            for key, value in best_trial.params.items():
                print('    {}: {}'.format(key, value))
    
        # update using best parameters
        self.init_kwargs.update(best_trial.params)

        return best_trial, study.trials_dataframe()


#
# Aux functions 
#


def _fit_worker(args):
    agent_class, train_env, init_kwargs, fit_kwargs = args
    agent = agent_class(train_env, copy_env=False,
                        reseed_env=False, **init_kwargs)
    info = agent.fit(**fit_kwargs)
    return agent, info


def _signal_timeout_handler(signum, frame):
    raise Exception("timed out!")


#
# Analysis functions 
#


def plot_episode_rewards(agent_stats_list, cumulative=False, fignum=None, show=True):
    plt.figure(fignum)
    for agent_stats in agent_stats_list:
        if not 'episode_rewards' in agent_stats.fit_info:
            logging.warning("episode_rewards not available for %s."%agent_stats.agent_name)
            continue 
        else:
            # train agents if they are not already trained
            if agent_stats.fitted_agents is None:
                agent_stats.fit()
            # get reward statistics and plot them 
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


def compare_policies(agent_stats_list, eval_env=None, eval_horizon=None, stationary_policy=True, nsim=10, fignum=None, show=True, plot=True):
    """
    Compare the policies of each of the agents in agent_stats_list. 
    Each element of the agent_stats_list contains a list of fitted agents. 
    To evaluate the policy, we repeat nsim times:
        * choose one of the fitted agents uniformly at random
        * run its policy in eval_env for eval_horizon time steps
    
    To do
    ------
    Paralellize evaluations of each agent.

    Parameters
    ----------
    agent_stats_list : list of AgentStats objects.
    eval_env : Model
        Environment where to evaluate the policies. If None, it is taken from AgentStats.
    eval_horizon : int 
        Number of time steps for policy evaluation. If None, it is taken from AgentStats.
    stationary_policy : bool
        If False, the time step h (0<= h <= eval_horizon) is sent as input to agent.policy() for policy evaluation.
    nsim : int 
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
            assert eval_env is not None, "eval_env not given in AgentStats %s" % agent_stats.agent_name
        if use_horizon_from_agent_stats:
            eval_horizon = agent_stats.eval_horizon
            assert eval_horizon is not None, "eval_horizon not given in AgentStats %s" % agent_stats.agent_name

        # evaluate agent 
        episode_rewards = np.zeros(nsim)
        for sim in range(nsim):
            # choose one of the fitted agents randomly
            agent_idx = rng.integers(len(agent_stats.fitted_agents)) 
            agent = agent_stats.fitted_agents[agent_idx]
            # evaluate agent
            observation = eval_env.reset()
            for hh in range(eval_horizon):
                if stationary_policy:
                    action = agent.policy(observation, **agent_stats.policy_kwargs)
                else:
                    action = agent.policy(observation, hh, **agent_stats.policy_kwargs)
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
            ax = sns.boxplot(data=output)
            ax.set_xlabel("agent")
            ax.set_ylabel("rewards in one episode")
            plt.title("Environment = %s"%eval_env.unwrapped.name)
            if show:
                plt.show()
    
    return output