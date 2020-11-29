from copy import deepcopy
from datetime import datetime
from joblib import Parallel, delayed
import logging
import os
import pickle

import rlberry.seeding as seeding
from rlberry.agents import IncrementalAgent
from rlberry.stats.evaluation import compare_policies

_OPTUNA_INSTALLED = True
try:
    import optuna
except Exception:
    _OPTUNA_INSTALLED = False


#
# Main class
#

class AgentStats:
    """Class to train, optimize hyperparameters, evaluate and gather statistics about an agent."""

    def __init__(self,
                 agent_class,
                 train_env,
                 eval_env=None,
                 eval_horizon=None,
                 init_kwargs=None,
                 fit_kwargs=None,
                 policy_kwargs=None,
                 agent_name=None,
                 n_fit=4,
                 n_jobs=4,
                 output_dir='stats_data',
                 verbose=5):
        """
        Parameters
        ----------
        agent_class
            Class of the agent.
        train_env : Model
            Enviroment used to initialize/train the agent.
        eval_env : Model
            Environment used to evaluate the agent. If None, set to a
            reseeded deep copy of train_env.
        init_kwargs : dict
            Arguments required by the agent's constructor.
        fit_kwargs : dict
            Arguments required to train the agent.
        policy_kwargs : dict
            Arguments required to call agent.policy().
        agent_name : str
            Name of the agent. If None, set to agent_class.name
        n_fit : int
            Number of agent instances to fit.
        n_jobs : int
            Number of jobs to train the agents in parallel using joblib.
        output_dir : str
            Directory where to store data by default.
        verbose : int
            Verbosity level.
        """
        # agent_class should only be None when the constructor is called
        # by the class method AgentStats.load(), since the agent class
        # will be loaded.
        if agent_class is not None:

            self.agent_name = agent_name
            if agent_name is None:
                self.agent_name = agent_class.name

            # create oject identifier
            timestamp = datetime.timestamp(datetime.now())
            self.identifier = 'stats_{}_{}'.format(self.agent_name,
                                                   str(int(timestamp)))

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
            self.init_kwargs = deepcopy(init_kwargs)
            self.fit_kwargs = fit_kwargs
            self.policy_kwargs = deepcopy(policy_kwargs)
            self.n_fit = n_fit
            self.n_jobs = n_jobs
            self.output_dir = output_dir
            self.verbose = verbose

            if init_kwargs is None:
                self.init_kwargs = {}
            if fit_kwargs is None:
                self.fit_kwargs = {}
            if policy_kwargs is None:
                self.policy_kwargs = {}

            # Create environment copies for training
            self.train_env_set = []
            for _ in range(n_fit):
                _env = deepcopy(train_env)
                _env.reseed()
                self.train_env_set.append(_env)

            #
            self.fitted_agents = None
            self.fit_kwargs_list = None  # keep in memory for partial_fit()
            self.fit_statistics = {}

            #
            self.rng = seeding.get_rng()

            # optuna study
            self.study = None

            # default filename to save data
            self.default_filename = os.path.join(self.output_dir,
                                                 self.identifier)

    def fit(self):
        """
        Fit the agent instances in parallel.
        """
        if self.verbose > 0:
            print("\n Training AgentStats for %s... \n" % self.agent_name)
        args = [(self.agent_class, train_env,
                deepcopy(self.init_kwargs), deepcopy(self.fit_kwargs))
                for train_env in self.train_env_set]

        workers_output = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_worker)(arg) for arg in args)

        self.fitted_agents, stats = (
            [i for i, j in workers_output],
            [j for i, j in workers_output])

        if self.verbose > 0:
            print("\n ... trained! \n")

        # gather all stats in a dictionary
        self._process_fit_statistics(stats)

    def partial_fit(self, fraction):
        """
        Partially fit the agent instances (not parallel).
        """
        assert fraction > 0.0 and fraction <= 1.0
        assert issubclass(self.agent_class, IncrementalAgent)

        # Create instances if this is the first call
        if self.fitted_agents is None:
            self.fitted_agents = []
            self.fit_kwargs_list = []
            for train_env in self.train_env_set:
                init_kwargs = deepcopy(self.init_kwargs)
                agent = self.agent_class(train_env, copy_env=False,
                                         reseed_env=False, **init_kwargs)
                self.fitted_agents.append(agent)
                self.fit_kwargs_list.append(deepcopy(self.fit_kwargs))

        # Run partial fit
        stats = []
        for agent, fit_kwargs in zip(self.fitted_agents, self.fit_kwargs_list):
            info = agent.partial_fit(fraction, **fit_kwargs)
            stats.append(info)
        self._process_fit_statistics(stats)

    def _process_fit_statistics(self, stats):
        """Gather stats in a dictionary"""
        for entry in self.fit_info:
            self.fit_statistics[entry] = []
            for stat in stats:
                self.fit_statistics[entry].append(stat[entry])

    def save(self, filename=None, **kwargs):
        """
        Parameters
        ----------
        filename : string
            Filename with .pickle extension.
            If None, default_filename attribute is used.
        """
        if filename is None:
            filename = self.default_filename
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

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

    def optimize_hyperparams(self,
                             n_trials=5,
                             timeout=60,
                             n_sim=5,
                             n_fit=2,
                             n_jobs=2,
                             sampler_method='random',
                             pruner_method='halving',
                             continue_previous=False,
                             partial_fit_fraction=0.25,
                             sampler_kwargs={}):
        """
        Run hyperparameter optimization and updates init_kwargs with the
        best hyperparameters found.

        Currently supported sampler_method:
            'random' -> Random Search
            'optuna_default' -> TPE
            'grid' -> Grid Search
            'cmaes' -> CMA-ES

        Currently supported pruner_method:
            'none'
            'halving'

        Parameters
        ----------
        n_trials: int
            Number of agent evaluations
        timeout: int
            Stop study after the given number of second(s).
            Set to None for unlimited time.
        n_sim : int
            Number of Monte Carlo simulations to evaluate a policy.
        n_fit: int
            Number of agents to fit for each hyperparam evaluation.
        n_jobs: int
            Number of jobs to fit agents for each hyperparam evaluation,
            and also the number of jobs of Optuna.
        sampler_method : str
            Optuna sampling method.
        pruner_method : str
            Optuna pruner method.
        continue_previous : bool
            Set to true to continue previous Optuna study. If true,
            sampler_method and pruner_method will be
            the same as in the previous study.
        partial_fit_fraction : double, in ]0, 1]
            Fraction of the agent to fit for partial evaluation
            (allows pruning of trials).
            Only used for agents that implement partial_fit()
            (IncrementalAgent interface).
        sampler_kwargs : dict
            Allows users to use different Optuna samplers with
            personalized arguments.
        """
        global _OPTUNA_INSTALLED
        if not _OPTUNA_INSTALLED:
            logging.error("Optuna not installed.")
            return

        assert self.eval_horizon is not None, \
            "To use optimize_hyperparams(), eval_horizon must be given to AgentStats."

        assert partial_fit_fraction > 0.0 and partial_fit_fraction <= 1.0

        #
        # Create optuna study
        #
        if continue_previous:
            assert self.study is not None
            study = self.study

        else:
            # get sampler
            if sampler_method == 'random':
                optuna_seed = self.rng.integers(2**16)
                sampler = optuna.samplers.RandomSampler(seed=optuna_seed)
            elif sampler_method == 'grid':
                assert sampler_kwargs is not None, \
                    "To use GridSampler, a search_space dictionary must be provided."
                sampler = optuna.samplers.GridSampler(**sampler_kwargs)
            elif sampler_method == 'cmaes':
                optuna_seed = self.rng.integers(2**16)
                sampler_kwargs['seed'] = optuna_seed
                sampler = optuna.samplers.CmaEsSampler(**sampler_kwargs)
            elif sampler_method == 'optuna_default':
                sampler = optuna.samplers.TPESampler(**sampler_kwargs)
            else:
                raise NotImplementedError("Sampler method %s is \
                                          not implemented." % sampler_method)

            # get pruner
            if pruner_method == 'halving':
                pruner = optuna.pruners.SuccessiveHalvingPruner(
                            min_resource=1,
                            reduction_factor=4,
                            min_early_stopping_rate=0)
            elif pruner_method == 'none':
                pruner = None
            else:
                raise NotImplementedError("Pruner method %s is not implemented." \
                                          % pruner_method)

            # optuna study
            study = optuna.create_study(sampler=sampler,
                                        pruner=pruner,
                                        direction='maximize')
            self.study = study

        def objective(trial):
            kwargs = deepcopy(self.init_kwargs)

            # will raise exception if sample_parameters() is not
            # implemented by the agent class
            kwargs.update(self.agent_class.sample_parameters(trial))

            #
            # fit and evaluate agents
            #
            # Create AgentStats with hyperparams
            params_stats = AgentStats(
                self.agent_class,
                deepcopy(self.train_env),
                init_kwargs=kwargs,   # kwargs are being optimized
                fit_kwargs=deepcopy(self.fit_kwargs),
                policy_kwargs=deepcopy(self.policy_kwargs),
                agent_name='optim',
                n_fit=n_fit,
                n_jobs=n_jobs,
                verbose=0)

            # Evaluation environment copy
            params_eval_env = deepcopy(self.eval_env)
            params_eval_env.reseed()

            #
            # Case 1: partial fit, that allows pruning
            #
            if partial_fit_fraction < 1.0 \
                    and issubclass(params_stats.agent_class, IncrementalAgent):
                fraction_complete = 0.0
                step = 0
                while fraction_complete < 1.0:
                    #
                    params_stats.partial_fit(partial_fit_fraction)
                    # Get rewards
                    eval_result = compare_policies(
                                [params_stats],
                                eval_env=params_eval_env,
                                eval_horizon=self.eval_horizon,
                                stationary_policy=True,
                                n_sim=n_sim,
                                plot=False)

                    rewards = eval_result['optim'].values.mean()
                    # Report intermediate objective value
                    trial.report(rewards, step)

                    #
                    fraction_complete += partial_fit_fraction
                    step += 1
                    #

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            #
            # Case 2: full fit
            #
            else:
                # Fit and evaluate params_stats
                params_stats.fit()

                # Get rewards
                eval_result = compare_policies(
                            [params_stats],
                            eval_env=params_eval_env,
                            eval_horizon=self.eval_horizon,
                            stationary_policy=True,
                            n_sim=n_sim,
                            plot=False)

                rewards = eval_result['optim'].values.mean()

            return rewards

        try:
            study.optimize(objective,
                           n_trials=n_trials,
                           n_jobs=n_jobs,
                           timeout=timeout)
        except KeyboardInterrupt:
            logging.warning("Evaluation stopped.")

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

