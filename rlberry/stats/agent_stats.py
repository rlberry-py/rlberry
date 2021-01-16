"""
Notes
-----

dill[1] is required to extend pickle (see https://stackoverflow.com/a/25353243)

If possible, pickle is prefered (since it is faster).

[1] https://github.com/uqfoundation/dill
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed
import json
import logging
import dill
import pickle
import pandas as pd
import rlberry.seeding as seeding
from rlberry.agents import IncrementalAgent
from rlberry.utils.logging import configure_logging
from rlberry.stats.evaluation import mc_policy_evaluation


_OPTUNA_INSTALLED = True
try:
    import optuna
except Exception:
    _OPTUNA_INSTALLED = False


logger = logging.getLogger(__name__)


#
# Main class
#

class AgentStats:
    """
    Class to train, optimize hyperparameters, evaluate and gather
    statistics about an agent.

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
        Arguments required to call agent.fit().
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
    joblib_backend: str, {'threading', 'loky' or 'multiprocessing'}, default: 'threading'
        Backend for joblib Parallel.
    thread_logging_level : str, default: 'INFO'
        Logging level in each of the threads used to fit agents.
    """

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
                 output_dir=None,
                 joblib_backend='threading',
                 thread_logging_level='INFO'):
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
            self.joblib_backend = joblib_backend
            self.thread_logging_level = thread_logging_level

            # output dir
            output_dir = output_dir or self.identifier
            self.output_dir = Path(output_dir)

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

            # Create list of writers for each agent that will be trained
            self.writers = [('default', None) for _ in range(n_fit)]

            #
            self.fitted_agents = None
            self.fit_kwargs_list = None  # keep in memory for partial_fit()
            self.fit_statistics = None
            self.best_hyperparams = None

            #
            self.rng = seeding.get_rng()

            # optuna study
            self.study = None

    def set_output_dir(self, output_dir):
        """
        Change output directory.

        Parameters
        -----------
        output_dir : str
        """
        self.output_dir = Path(output_dir)

    def set_writer(self, idx, writer_fn, writer_kwargs=None):
        """
        Note
        -----
        Must be called right after creating an instance of AgentStats.

        Parameters
        ----------
        writer_fn : callable, None or 'default'
            Returns a writer for an agent, e.g. tensorboard SummaryWriter,
            rlberry PeriodicWriter.
            If 'default', use the default writer in the Agent class.
            If None, disable any writer
        writer_kwargs : dict or None
            kwargs for writer_fn
        idx : int
            Index of the agent to set the writer (0 <= idx < `n_fit`).
            AgentStats fits `n_fit` agents, the writer of each one of them
            needs to be set separetely.
        """
        assert idx >= 0 and idx < self.n_fit, \
            "Invalid index sent to AgentStats.set_writer()"
        writer_kwargs = writer_kwargs or {}
        self.writers[idx] = (writer_fn, writer_kwargs)

    def disable_writers(self):
        """
        Set all writers to None.
        """
        self.writers = [('default', None) for _ in range(self.n_fit)]
        if self.fitted_agents is not None:
            for agent in self.fitted_agents:
                agent.set_writer(None)

    def fit(self):
        """
        Fit the agent instances in parallel.
        """
        logger.info(f"Training AgentStats for {self.agent_name}... ")
        args = [(self.agent_class,
                train_env,
                deepcopy(self.init_kwargs),
                deepcopy(self.fit_kwargs),
                writer,
                self.thread_logging_level,
                thread_idx)
                for thread_idx, (train_env, writer)
                in enumerate(zip(self.train_env_set, self.writers))]

        workers_output = Parallel(n_jobs=self.n_jobs,
                                  verbose=5,
                                  backend=self.joblib_backend)(
            delayed(_fit_worker)(arg) for arg in args)

        self.fitted_agents, stats = (
            [i for i, j in workers_output],
            [j for i, j in workers_output])

        logger.info("... trained!")

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
            for idx, train_env in enumerate(self.train_env_set):
                init_kwargs = deepcopy(self.init_kwargs)
                # create agent instance
                agent = self.agent_class(train_env, copy_env=False,
                                         reseed_env=False, **init_kwargs)
                # set agent writer
                if self.writers[idx][0] is None:
                    agent.set_writer(None)
                elif self.writers[idx][0] != 'default':
                    writer_fn = self.writers[idx][0]
                    writer_kwargs = self.writers[idx][1]
                    agent.set_writer(writer_fn(**writer_kwargs))
                #
                self.fitted_agents.append(agent)
                #
                self.fit_kwargs_list.append(deepcopy(self.fit_kwargs))

        # Run partial fit
        stats = []
        for agent, fit_kwargs in zip(self.fitted_agents, self.fit_kwargs_list):
            info = agent.partial_fit(fraction, **fit_kwargs)
            stats.append(info)
        self._process_fit_statistics(stats)

    def _process_fit_statistics(self, stats):
        """Gather stats in a dictionary"""
        self.fit_statistics = {}
        for entry in self.fit_info:
            self.fit_statistics[entry] = []
            for stat in stats:
                self.fit_statistics[entry].append(stat[entry])

    def save_results(self, output_dir=None, **kwargs):
        """
        Save the results obtained by optimize_hyperparameters(),
        fit() and partial_fit() to a directory.

        Parameters
        ----------
        output_dir : str or None
            Output directory. If None, use self.output_dir.
        """
        # use default self.output_dir if another one is not provided.
        output_dir = output_dir or self.output_dir
        output_dir = Path(output_dir)

        # create dir if it does not exist
        output_dir.mkdir(parents=True, exist_ok=True)
        # save optimized hyperparameters
        if self.best_hyperparams is not None:
            fname = Path(output_dir) / 'best_hyperparams.json'
            _safe_serialize_json(self.best_hyperparams, fname)
        # save fit_statistics that can be aggregated in a pandas DataFrame
        if self.fit_statistics is not None:
            for entry in self.fit_statistics:
                # gather data for entry
                all_data = {}
                for run, data in enumerate(self.fit_statistics[entry]):
                    all_data[f'run_{run}'] = data
                try:
                    output = pd.DataFrame(all_data)
                    # save
                    fname = Path(output_dir) / f'stats_{entry}.csv'
                    output.to_csv(fname, index=None)
                except Exception:
                    logger.warning(f"Could not save entry [{entry}]"
                                   + " of fit_statistics.")

    def save(self, filename='stats', **kwargs):
        """
        Pickle the AgentStats object completely, so that
        it can be loaded and continued later.

        Removes writers, since they usually cannot be pickled.

        This is useful, for instance:
        * If we want to run hyperparameter optimization for
        a few minutes/hours, save the results, then continue
        the optimization later.
        * If we ran some experiments and we want to reload
        the trained agents to visualize their policies
        policy in a rendered environment and create videos.

        Parameters
        ----------
        filename : string
            Filename with .pickle extension.
            Saves to output_dir / filename
        """
        # remove writers
        self.disable_writers()

        # save
        filename = Path(filename).with_suffix('.pickle')
        filename = self.output_dir / filename
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            with filename.open("wb") as ff:
                pickle.dump(self.__dict__, ff)
            logger.info("Saved AgentStats({}) using pickle.".format(self.agent_name))
        except Exception:
            with filename.open("wb") as ff:
                dill.dump(self.__dict__, ff)
            logger.info("Saved AgentStats({}) using dill.".format(self.agent_name))

    @classmethod
    def load(cls, filename):
        filename = Path(filename).with_suffix('.pickle')

        obj = cls(None, None)
        try:
            with filename.open('rb') as ff:
                tmp_dict = pickle.load(ff)
            logger.info("Loaded AgentStats using pickle.")
        except Exception:
            with filename.open('rb') as ff:
                tmp_dict = dill.load(ff)
            logger.info("Loaded AgentStats using dill.")

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
                             sampler_kwargs=None,
                             evaluation_function=None,
                             evaluation_function_kwargs=None):
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
        sampler_kwargs : dict or None
            Allows users to use different Optuna samplers with
            personalized arguments.
        evaluation_function : callable(agent_list, eval_env, **kwargs)->double, default: None
            Function to maximize, that takes a list of agents and an environment as input, and returns a double.
            If None, search for hyperparameters that maximize the mean reward.
        evaluation_function_kwargs : dict or None
            kwargsfor evaluation_function
        """
        #
        # setup
        #
        global _OPTUNA_INSTALLED
        if not _OPTUNA_INSTALLED:
            logging.error("Optuna not installed.")
            return

        assert self.eval_horizon is not None, \
            "To use optimize_hyperparams(), " + \
            "eval_horizon must be given to AgentStats."

        assert partial_fit_fraction > 0.0 and partial_fit_fraction <= 1.0

        evaluation_function_kwargs = evaluation_function_kwargs or {}
        if evaluation_function is None:
            evaluation_function = mc_policy_evaluation
            evaluation_function_kwargs = {
                'eval_horizon': self.eval_horizon,
                'n_sim': n_sim,
                'gamma': 1.0,
                'policy_kwargs': self.policy_kwargs,
                'stationary_policy': True,
            }

        #
        # Create optuna study
        #
        if continue_previous:
            assert self.study is not None
            study = self.study

        else:
            if sampler_kwargs is None:
                sampler_kwargs = {}
            # get sampler
            if sampler_method == 'random':
                optuna_seed = self.rng.integers(2**16)
                sampler = optuna.samplers.RandomSampler(seed=optuna_seed)
            elif sampler_method == 'grid':
                assert sampler_kwargs is not None, \
                    "To use GridSampler, " + \
                    "a search_space dictionary must be provided."
                sampler = optuna.samplers.GridSampler(**sampler_kwargs)
            elif sampler_method == 'cmaes':
                optuna_seed = self.rng.integers(2**16)
                sampler_kwargs['seed'] = optuna_seed
                sampler = optuna.samplers.CmaEsSampler(**sampler_kwargs)
            elif sampler_method == 'optuna_default':
                sampler = optuna.samplers.TPESampler(**sampler_kwargs)
            else:
                raise NotImplementedError(
                      "Sampler method %s is not implemented." % sampler_method)

            # get pruner
            if pruner_method == 'halving':
                pruner = optuna.pruners.SuccessiveHalvingPruner(
                            min_resource=1,
                            reduction_factor=4,
                            min_early_stopping_rate=0)
            elif pruner_method == 'none':
                pruner = None
            else:
                raise NotImplementedError(
                      "Pruner method %s is not implemented." % pruner_method)

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
                thread_logging_level='WARNING')

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
                    # Evaluate params
                    eval_result = evaluation_function(params_stats.fitted_agents,
                                                      params_eval_env,
                                                      **evaluation_function_kwargs)

                    eval_value = eval_result.mean()

                    # Report intermediate objective value
                    trial.report(eval_value, step)

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

                # Evaluate params
                eval_result = evaluation_function(params_stats.fitted_agents,
                                                  params_eval_env,
                                                  **evaluation_function_kwargs)

                eval_value = eval_result.mean()

            return eval_value

        try:
            study.optimize(objective,
                           n_trials=n_trials,
                           n_jobs=n_jobs,
                           timeout=timeout)
        except KeyboardInterrupt:
            logger.warning("Evaluation stopped.")

        # continue
        best_trial = study.best_trial

        logger.info(f'Number of finished trials: {len(study.trials)}')
        logger.info('Best trial:')
        logger.info(f'Value: {best_trial.value}')
        logger.info('Params:')
        for key, value in best_trial.params.items():
            logger.info(f'    {key}: {value}')

        # store best parameters
        self.best_hyperparams = best_trial.params

        # update using best parameters
        self.init_kwargs.update(best_trial.params)

        return best_trial, study.trials_dataframe()


#
# Aux functions
#

def _fit_worker(args):
    agent_class, train_env, init_kwargs, \
        fit_kwargs, writer, thread_logging_level, thread_id = args
    # logging level in thread
    configure_logging(thread_logging_level)
    # create agent
    agent = agent_class(train_env, copy_env=False,
                        reseed_env=False, **init_kwargs)
    agent.name += f"(worker {thread_id})"

    # set writer
    if writer[0] is None:
        agent.set_writer(None)
    elif writer[0] != 'default':
        writer_fn = writer[0]
        writer_kwargs = writer[1]
        agent.set_writer(writer_fn(**writer_kwargs))
    # fit agent
    info = agent.fit(**fit_kwargs)

    # Remove writer after fit (prevent pickle problems)
    agent.set_writer(None)

    return agent, info


def _safe_serialize_json(obj, filename):
    """
    Source: https://stackoverflow.com/a/56138540/5691288
    """
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    with open(filename, 'w') as fp:
        json.dump(obj, fp, sort_keys=True, indent=4, default=default)
