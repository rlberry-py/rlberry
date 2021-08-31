import concurrent.futures
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from rlberry.seeding import safe_reseed, set_external_seed
from rlberry.seeding import Seeder

import functools
import json
import logging
import dill
import numpy as np
import pickle
import pandas as pd
import shutil
import threading
import multiprocessing
from rlberry.utils.logging import configure_logging
from rlberry.utils.writers import DefaultWriter
from rlberry.stats.utils import create_database
from typing import Tuple


# Using a lock when creating envs and agents, to avoid problems
# as here: https://github.com/openai/gym/issues/281
_LOCK_THREAD = threading.Lock()
_LOCK_PROCESS = multiprocessing.Manager().Lock()

_OPTUNA_INSTALLED = True
try:
    import optuna
except Exception:
    _OPTUNA_INSTALLED = False


logger = logging.getLogger(__name__)


#
# Aux
#

class AgentHandler:
    """
    Wraps an Agent so that it can be either loaded in memory
    or represented by a file storing the Agent data.
    It is necessary because not all agents can be pickled.

    Parameters
    ----------
    id: int
        Integer identifying the handler.
    filename: str or Path
        File where to save/load the agent instance
    seeder: Seeder
        Required for reseeding.
    agent_class:
        Class of the agent to be wrapped
    agent_instance:
        An instance of agent_class, or None (if not loaded).
    **agent_kwargs:
        Arguments required by __init__ method of agent_class.
    """
    def __init__(self,
                 id,
                 filename,
                 seeder,
                 agent_class,
                 agent_instance=None,
                 **agent_kwargs) -> None:
        self._id = id
        self._fname = Path(filename)
        self._seeder = seeder
        self._agent_class = agent_class
        self._agent_instance = agent_instance
        self._agent_kwargs = agent_kwargs

    @property
    def id(self):
        return self._id

    def set_instance(self, agent_instance):
        self._agent_instance = agent_instance

    def is_empty(self):
        return self._agent_instance is None and (not self._fname.exists())

    def is_loaded(self):
        return self._agent_instance is not None

    def load(self) -> bool:
        with _LOCK_PROCESS:
            with _LOCK_THREAD:
                try:
                    self._agent_instance = self._agent_class.load(self._fname, **self._agent_kwargs)
                    safe_reseed(self._agent_instance.env, self._seeder)
                    logger.info(f'Sucessful call to AgentHandler.load() for {self._agent_class}')
                    return True
                except Exception as ex:
                    self._agent_instance = None
                    logger.info(f'Failed call to AgentHandler.load() for {self._agent_class}: {ex}')
                    return False

    def dump(self):
        """Saves agent to file and remove it from memory."""
        with _LOCK_PROCESS:
            with _LOCK_THREAD:
                if self._agent_instance is not None:
                    saved_filename = self._agent_instance.save(self._fname)
                    # saved_filename might have appended the correct extension, for instance,
                    # so self._fname must be updated.
                    if not saved_filename:
                        logger.warning(f'Instance of {self._agent_class} cannot be saved and will be kept in memory.')
                        return
                    self._fname = Path(saved_filename)
                    del self._agent_instance
                    self._agent_instance = None
                    logger.info(f'Sucessful call to AgentHandler.dump() for {self._agent_class}')

    def __getattr__(self, attr):
        """
        Allows AgentHandler to behave like the handled Agent.
        """
        if attr[:2] == '__':
            raise AttributeError(attr)
        if attr in self.__dict__:
            return getattr(self, attr)

        assert not self.is_empty(), 'Calling AgentHandler with no agent instance stored.'
        if not self.is_loaded():
            loaded = self.load()
            if not loaded:
                raise RuntimeError(f'Could not load Agent from {self._fname}.')
        return getattr(self._agent_instance, attr)

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
    train_env : Tuple (constructor, kwargs)
        Enviroment used to initialize/train the agent.
    fit_budget : int
        Argument required to call agent.fit(). If None, must be given in fit_kwargs['fit_budget'].
    eval_env : Tuple (constructor, kwargs)
        Environment used to evaluate the agent. If None, set to a
        reseeded deep copy of train_env.
    init_kwargs : dict
        Arguments required by the agent's constructor.
    fit_kwargs : dict
        Extra required to call agent.fit(bugdet, **fit_kwargs).
    eval_kwargs : dict
        Arguments required to call agent.eval().
    agent_name : str
        Name of the agent. If None, set to agent_class.name
    n_fit : int
        Number of agent instances to fit.
    output_dir : str
        Directory where to store data by default.
    parallelization: {'thread', 'process'}, default: 'process'
        Whether to parallelize  agent training using threads or processes.
    thread_logging_level : str, default: 'INFO'
        Logging level in each of the threads used to fit agents.
    seed : np.random.SeedSequence, rlberry.seeding.Seeder or int, default : None
        Seed sequence from which to spawn the random number generator.
        If None, generate random seed.
        If int, use as entropy for SeedSequence.
        If seeder, use seeder.seed_seq
    """

    def __init__(self,
                 agent_class,
                 train_env,
                 fit_budget=None,
                 eval_env=None,
                 init_kwargs=None,
                 fit_kwargs=None,
                 eval_kwargs=None,
                 agent_name=None,
                 n_fit=4,
                 output_dir=None,
                 parallelization='process',
                 thread_logging_level='INFO',
                 seed=None):
        # agent_class should only be None when the constructor is called
        # by the class method AgentStats.load(), since the agent class
        # will be loaded.

        if agent_class is None:
            return None  # Must only happen when load() method is called.

        self.seeder = Seeder(seed)

        self.agent_name = agent_name
        if agent_name is None:
            self.agent_name = agent_class.name

        # Check train_env and eval_env
        assert isinstance(
            train_env, Tuple), "[AgentStats]train_env must be Tuple (constructor, kwargs)"
        if eval_env is not None:
            assert isinstance(
                eval_env, Tuple), "[AgentStats]train_env must be Tuple (constructor, kwargs)"

        # create oject identifier
        timestamp = datetime.timestamp(datetime.now())
        timestamp = str(int(timestamp))
        self.timestamp = timestamp
        self.identifier = f'stats_{self.agent_name}_{timestamp}'
        self.unique_id = str(id(self)) + str(timestamp)

        # Agent class
        self.agent_class = agent_class

        # Train env
        self.train_env = train_env

        # Check eval_env
        if eval_env is None:
            eval_env = deepcopy(train_env)

        self._eval_env = eval_env

        # check kwargs
        init_kwargs = init_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        eval_kwargs = eval_kwargs or {}

        # params
        self.init_kwargs = deepcopy(init_kwargs)
        self.fit_kwargs = deepcopy(fit_kwargs)
        self.eval_kwargs = deepcopy(eval_kwargs)
        self.n_fit = n_fit
        self.parallelization = parallelization
        self.thread_logging_level = thread_logging_level
        if fit_budget is not None:
            self.fit_budget = fit_budget
        else:
            try:
                self.fit_budget = fit_kwargs.pop('fit_budget')
            except KeyError:
                raise ValueError('[AgentStats] fit_budget missing in __init__().')

        # output dir
        output_dir = output_dir or ('temp/' + self.identifier)
        self.output_dir = Path(output_dir)

        # Create list of writers for each agent that will be trained
        self.writers = [('default', None) for _ in range(n_fit)]

        #
        self.agent_handlers = None
        self._reset_agent_handlers()
        self.default_writer_data = None
        self.best_hyperparams = None

        # optuna study and database
        self.optuna_study = None
        self.db_filename = None
        self.optuna_storage_url = None

    def _init_optuna_storage_url(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_filename = self.output_dir / f'data_{self.unique_id}.db'
        if create_database(self.db_filename):
            self.optuna_storage_url = f"sqlite:///{self.db_filename}"
        else:
            self.db_filename = None
            self.optuna_storage_url = "sqlite:///:memory:"
            logger.warning(f'Unable to create databate {self.db_filename}. Using sqlite:///:memory:')

    def _reset_agent_handlers(self):
        handlers_seeders = self.seeder.spawn(self.n_fit, squeeze=False)
        self.agent_handlers = [
            AgentHandler(
                id=ii,
                filename=self.output_dir / Path(f'agent_handlers/{self.unique_id}_{ii}'),
                seeder=handlers_seeders[ii],
                agent_class=self.agent_class,
                agent_instance=None,
                # kwargs
                env=self.train_env,
                **self.init_kwargs,
            )
            for ii in range(self.n_fit)
        ]
        self.clear_handlers()

    def build_eval_env(self):
        """
        Return an instantiated and reseeded evaluation environment.
        """
        return _preprocess_env(self._eval_env, self.seeder)

    @property
    def writer_data(self):
        return self.default_writer_data

    def eval(self, eval_env=None):
        """
        Call .eval() method in all fitted agents and return average result.
        """
        if eval_env is None:
            eval_env = self.build_eval_env()
        values = [agent.eval(eval_env,
                             **self.eval_kwargs) for agent in self.agent_handlers if not agent.is_empty()]
        if len(values) == 0:
            return np.nan
        return np.mean(values)

    def set_output_dir(self, output_dir):
        """
        Change output directory.

        Parameters
        -----------
        output_dir : str
        """
        self.output_dir = Path(output_dir)

    def clear_output_dir(self):
        """Delete output_dir and all its data."""
        try:
            shutil.rmtree(self.output_dir)
        except FileNotFoundError:
            logger.warning(f'No directory {self.output_dir} found to be deleted.')

    def clear_handlers(self):
        """Delete files from output_dir/agent_handlers that are managed by this class."""
        for handler in self.agent_handlers:
            if handler._fname.exists():
                handler._fname.unlink()

    def set_writer(self, idx, writer_fn, writer_kwargs=None):
        """
        Note
        -----
        Must be called right after creating an instance of AgentStats.

        Parameters
        ----------
        writer_fn : callable, None or 'default'
            Returns a writer for an agent, e.g. tensorboard SummaryWriter,
            rlberry DefaultWriter.
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
        for agent in self.agent_handlers:
            if not agent.is_empty():
                agent.set_writer(None)

    def fit(self, budget=None, **kwargs):
        """
        Fit the agent instances in parallel.
        """
        del kwargs
        budget = budget or self.fit_budget

        logger.info(f"Training AgentStats for {self.agent_name}... ")
        seeders = self.seeder.spawn(self.n_fit)
        if not isinstance(seeders, list):
            seeders = [seeders]

        # remove agent instances from memory to that the agent handlers can be sent to different workers
        for handler in self.agent_handlers:
            handler.dump()

        args = [(
                handler,
                self.agent_class,
                self.train_env,
                budget,
                deepcopy(self.init_kwargs),
                deepcopy(self.fit_kwargs),
                writer,
                self.thread_logging_level,
                seeder)
                for (handler, seeder, writer)
                in zip(self.agent_handlers, seeders, self.writers)]

        if self.parallelization == 'thread':
            executor_class = concurrent.futures.ThreadPoolExecutor
        elif self.parallelization == 'process':
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            raise ValueError(f'Invalid backend for parallelization: {self.parallelization}')

        if len(args) == 1:
            workers_output = [_fit_worker(args[0])]

        else:
            with executor_class() as executor:
                futures = []
                for arg in args:
                    futures.append(executor.submit(_fit_worker, arg))

                workers_output = []
                for future in concurrent.futures.as_completed(futures):
                    workers_output.append(
                        future.result()
                    )
                executor.shutdown()

        workers_output.sort(key=lambda x: x.id)
        self.agent_handlers = workers_output

        logger.info("... trained!")

        # gather all stats in a dictionary
        self._gather_default_writer_data()

    def _gather_default_writer_data(self):
        """Gather DefaultWriter data in a dictionary"""
        self.default_writer_data = {}
        for ii, agent in enumerate(self.agent_handlers):
            if not agent.is_empty() and isinstance(agent.writer, DefaultWriter):
                self.default_writer_data[ii] = agent.writer.data

    def save(self, output_dir=None):
        """
        Save AgentStats data to a folder. The data can be
        later loaded to recreate an AgentStats instance.

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
        # save default_writer_data that can be aggregated in a pandas DataFrame
        if self.default_writer_data is not None:
            data_list = []
            for idx in self.default_writer_data:
                df = self.default_writer_data[idx]
                data_list.append(df)
            if len(data_list) > 0:
                all_writer_data = pd.concat(data_list, ignore_index=True)
                try:
                    output = pd.DataFrame(all_writer_data)
                    # save
                    fname = Path(output_dir) / 'data.csv'
                    output.to_csv(fname, index=None)
                except Exception:
                    logger.warning("Could not save default_writer_data.")

        #
        # Pickle AgentStats instance
        #

        # remove writers
        self.disable_writers()

        # clear agent handlers
        for handler in self.agent_handlers:
            handler.dump()

        # save
        filename = Path('stats').with_suffix('.pickle')
        filename = output_dir / filename
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            with filename.open("wb") as ff:
                pickle.dump(self.__dict__, ff)
            logger.info("Saved AgentStats({}) using pickle.".format(self.agent_name))
        except Exception:
            try:
                with filename.open("wb") as ff:
                    dill.dump(self.__dict__, ff)
                logger.info("Saved AgentStats({}) using dill.".format(self.agent_name))
            except Exception as ex:
                logger.warning("[AgentStats] Instance cannot be pickled: " + str(ex))

    @classmethod
    def load(cls, filename):
        filename = Path(filename).with_suffix('.pickle')

        obj = cls(None, None, None)
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
                             n_trials=256,
                             timeout=60,
                             n_fit=2,
                             n_optuna_workers=2,
                             optuna_parallelization='thread',
                             sampler_method='optuna_default',
                             pruner_method='halving',
                             continue_previous=False,
                             fit_fraction=1.0,
                             sampler_kwargs=None,
                             disable_evaluation_writers=True):
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

        Note
        ----
        After calling this method, agent handlers from previous calls to fit() will be erased.
        It is suggested to call fit() *after* a call to optimize_hyperparams().

        Parameters
        ----------
        n_trials: int
            Number of agent evaluations
        timeout: int
            Stop study after the given number of second(s).
            Set to None for unlimited time.
        n_fit: int
            Number of agents to fit for each hyperparam evaluation.
        n_optuna_workers: int
            Number of workers used by optuna for optimization.
        optuna_parallelization : 'thread' or 'process'
            Whether to use threads or processes for optuna parallelization.
        sampler_method : str
            Optuna sampling method.
        pruner_method : str
            Optuna pruner method.
        continue_previous : bool
            Set to true to continue previous Optuna study. If true,
            sampler_method and pruner_method will be
            the same as in the previous study.
        fit_fraction : double, in ]0, 1]
            Fraction of the agent to fit for partial evaluation
            (allows pruning of trials).
        sampler_kwargs : dict or None
            Allows users to use different Optuna samplers with
            personalized arguments.
        evaluation_function : callable(agent_list, eval_env, **kwargs)->double, default: None
            Function to maximize, that takes a list of agents and an environment as input, and returns a double.
            If None, search for hyperparameters that maximize the mean reward.
        evaluation_function_kwargs : dict or None
            kwargs for evaluation_function
        disable_evaluation_writers : bool, default: True
            If true, disable writers of agents used in the hyperparameter evaluation.
        """
        #
        # setup
        #
        TEMP_DIR = 'temp/optim'
        global _OPTUNA_INSTALLED
        if not _OPTUNA_INSTALLED:
            logging.error("Optuna not installed.")
            return

        assert fit_fraction > 0.0 and fit_fraction <= 1.0

        #
        # Create optuna study
        #
        if continue_previous:
            assert self.optuna_study is not None
            study = self.optuna_study

        else:
            if sampler_kwargs is None:
                sampler_kwargs = {}
            # get sampler
            if sampler_method == 'random':
                optuna_seed = self.seeder.rng.integers(2**16)
                sampler = optuna.samplers.RandomSampler(seed=optuna_seed)
            elif sampler_method == 'grid':
                assert sampler_kwargs is not None, \
                    "To use GridSampler, " + \
                    "a search_space dictionary must be provided."
                sampler = optuna.samplers.GridSampler(**sampler_kwargs)
            elif sampler_method == 'cmaes':
                optuna_seed = self.seeder.rng.integers(2**16)
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

            # storage
            self._init_optuna_storage_url()
            storage = optuna.storages.RDBStorage(self.optuna_storage_url)

            # optuna study
            study = optuna.create_study(sampler=sampler,
                                        pruner=pruner,
                                        storage=storage,
                                        direction='maximize')
            self.optuna_study = study

        #
        # Objective function
        #
        objective = functools.partial(
            _optuna_objective,
            init_kwargs=self.init_kwargs,  # self.init_kwargs
            agent_class=self.agent_class,  # self.agent_class
            train_env=self.train_env,    # self.train_env
            eval_env=self._eval_env,
            fit_budget=self.fit_budget,   # self.fit_budget
            eval_kwargs=self.eval_kwargs,  # self.eval_kwargs
            n_fit=n_fit,
            seeder=self.seeder,       # self.seeder
            temp_dir=TEMP_DIR,     # TEMP_DIR
            disable_evaluation_writers=disable_evaluation_writers,
            fit_fraction=fit_fraction
        )

        try:
            if optuna_parallelization == 'thread':
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for _ in range(n_optuna_workers):
                        executor.submit(
                            study.optimize,
                            objective,
                            n_trials=n_trials,
                            timeout=timeout)
            elif optuna_parallelization == 'process':
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for _ in range(n_optuna_workers):
                        executor.submit(
                            study.optimize,
                            objective,
                            n_trials=n_trials // n_optuna_workers,
                            timeout=timeout)
            else:
                raise ValueError(f'Invalid value for optuna_parallelization: {optuna_parallelization}.')

        except KeyboardInterrupt:
            logger.warning("Evaluation stopped.")

        # clear temp folder
        shutil.rmtree(TEMP_DIR)

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

        # reset agent handlers, so that they take the new parameters
        self._reset_agent_handlers()

        return best_trial, study.trials_dataframe()


#
# Aux functions
#

def _preprocess_env(env, seeder):
    """
    If env is a tuple (constructor, kwargs), creates an instance.

    Reseeds the env before returning.
    """
    if isinstance(env, tuple):
        constructor, kwargs = env
        kwargs = kwargs or {}
        env = constructor(**kwargs)

    reseeded = safe_reseed(env, seeder)
    assert reseeded

    return env


def _fit_worker(args):
    """
    Create and fit an agent instance
    """
    agent_handler, agent_class, train_env, fit_budget, init_kwargs, \
        fit_kwargs, writer, thread_logging_level, seeder = args

    # reseed external libraries
    set_external_seed(seeder)

    # logging level in thread
    configure_logging(thread_logging_level)

    with _LOCK_PROCESS:
        with _LOCK_THREAD:
            if agent_handler.is_empty():
                # preprocess and train_env
                train_env_instance = _preprocess_env(train_env, seeder)
                # create agent
                agent = agent_class(train_env_instance, copy_env=False, seeder=seeder, **init_kwargs)
                agent.name += f"(spawn_key{seeder.seed_seq.spawn_key})"
                # seed agent
                agent.reseed(seeder)
                agent_handler.set_instance(agent)
            agent = agent_handler

    # set writer
    if writer[0] is None:
        agent_handler.set_writer(None)
    elif writer[0] != 'default':
        writer_fn = writer[0]
        writer_kwargs = writer[1]
        agent_handler.set_writer(writer_fn(**writer_kwargs))
    # fit agent
    agent_handler.fit(fit_budget, **fit_kwargs)

    # Remove writer after fit (prevent pickle problems),
    # unless the agent uses DefaultWriter
    if not isinstance(agent_handler.writer, DefaultWriter):
        agent_handler.set_writer(None)

    # remove from memory to avoid pickle issues
    agent_handler.dump()

    return agent_handler


def _safe_serialize_json(obj, filename):
    """
    Source: https://stackoverflow.com/a/56138540/5691288
    """
    def default(obj):
        return f"<<non-serializable: {type(obj).__qualname__}>>"
    with open(filename, 'w') as fp:
        json.dump(obj, fp, sort_keys=True, indent=4, default=default)


def _optuna_objective(
    trial,
    init_kwargs,  # self.init_kwargs
    agent_class,  # self.agent_class
    train_env,    # self.train_env
    eval_env,
    fit_budget,   # self.fit_budget
    eval_kwargs,  # self.eval_kwargs
    n_fit,
    seeder,       # self.seeder
    temp_dir,     # TEMP_DIR
    disable_evaluation_writers,
    fit_fraction
):
    kwargs = deepcopy(init_kwargs)

    # will raise exception if sample_parameters() is not
    # implemented by the agent class
    kwargs.update(agent_class.sample_parameters(trial))

    #
    # fit and evaluate agents
    #
    # Create AgentStats with hyperparams
    params_stats = AgentStats(
        agent_class,
        train_env,
        fit_budget,
        eval_env=eval_env,
        init_kwargs=kwargs,   # kwargs are being optimized
        eval_kwargs=deepcopy(eval_kwargs),
        agent_name='optim',
        n_fit=n_fit,
        thread_logging_level='WARNING',
        parallelization='thread',
        seed=seeder,
        output_dir=temp_dir)

    if disable_evaluation_writers:
        for ii in range(params_stats.n_fit):
            params_stats.set_writer(ii, None, None)

    #
    # Case 1: partial fit, that allows pruning
    #
    if fit_fraction < 1.0:
        fraction_complete = 0.0
        step = 0
        while fraction_complete < 1.0:
            #
            params_stats.fit(int(fit_budget * fit_fraction))
            # Evaluate params
            eval_value = params_stats.eval()

            # Report intermediate objective value
            trial.report(eval_value, step)

            #
            fraction_complete += fit_fraction
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
        eval_value = params_stats.eval()

    # clear aux data
    params_stats.clear_handlers()

    return eval_value
