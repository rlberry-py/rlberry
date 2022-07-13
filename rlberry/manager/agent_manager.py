import concurrent.futures
from copy import deepcopy
from pathlib import Path
import cProfile, pstats
from pstats import SortKey
import functools
import json
import logging
import dill
import gc
import pickle
import shutil
import threading
import multiprocessing
from multiprocessing.spawn import _check_not_importing_main
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import rlberry
from rlberry.seeding import safe_reseed, set_external_seed
from rlberry.seeding import Seeder
from rlberry import metadata_utils
from rlberry.envs.utils import process_env
from rlberry.utils.logging import configure_logging
from rlberry.utils.writers import DefaultWriter
from rlberry.manager.utils import create_database
from rlberry import types


_OPTUNA_INSTALLED = True
try:
    import optuna
except Exception:
    _OPTUNA_INSTALLED = False

logger = rlberry.logger

# Aux
#


class AgentHandler:
    """
    Wraps an Agent so that it can be either loaded in memory
    or represented by a file storing the Agent data.
    It is used by `class`:~rlberry.manager.AgentManager` to handle the fact that
    not all agents can be pickled, when returning from the processes that
    train the agents.

    Parameters
    ----------
    id: int
        Integer identifying the handler.
    filename: str or Path
        File where to save/load the agent instance
    seeder: :class:`~rlberry.seeding.seeder.Seeder`
        Required for reseeding.
    agent_class:
        Class of the agent to be wrapped
    agent_instance:
        An instance of agent_class, or None (if not loaded).
    agent_kwargs: dict
        Arguments required by __init__ method of agent_class.
    """

    def __init__(
        self,
        id: int,
        filename: Union[str, Path],
        seeder: Seeder,
        agent_class,
        agent_instance=None,
        agent_kwargs=None,
    ) -> None:
        self._id = id
        self._fname = Path(filename)
        self._seeder = seeder
        self._agent_class = agent_class
        self._agent_instance = agent_instance
        self._agent_kwargs = agent_kwargs or {}

    @property
    def id(self):
        """AgentHandler identifier (int)."""
        return self._id

    def set_instance(self, agent_instance):
        self._agent_instance = agent_instance

    def get_instance(self):
        if not self.is_loaded():
            self.load()
        return self._agent_instance

    def is_empty(self):
        return self._agent_instance is None and (not self._fname.exists())

    def is_loaded(self):
        return self._agent_instance is not None

    def load(self) -> bool:
        """Load agent from file."""
        try:
            self._agent_instance = self._agent_class.load(
                self._fname, **self._agent_kwargs
            )
            safe_reseed(self._agent_instance.env, self._seeder)
            return True
        except Exception as ex:
            self._agent_instance = None
            logger.error(
                f"Failed call to AgentHandler.load() for {self._agent_class}: {ex}"
            )
            return False

    def dump(self):
        """Saves agent to file and remove it from memory."""
        if self._agent_instance is not None:
            saved_filename = self._agent_instance.save(self._fname)

            # saved_filename might have appended the correct extension, for instance,
            # so self._fname must be updated.
            if not saved_filename:
                logger.warning(
                    f"Instance of {self._agent_class} cannot be saved and will be kept in memory."
                )
                return

            self._fname = Path(saved_filename)

            del self._agent_instance
            self._agent_instance = None

    def __getattr__(self, attr):
        """Allows AgentHandler to behave like the handled Agent."""
        if attr[:2] == "__":
            raise AttributeError(attr)
        if attr in self.__dict__:
            return getattr(self, attr)

        assert (
            not self.is_empty()
        ), "Calling AgentHandler with no agent instance stored."
        if not self.is_loaded():
            loaded = self.load()
            if not loaded:
                raise RuntimeError(f"Could not load Agent from {self._fname}.")
        return getattr(self._agent_instance, attr)


#
# Main class
#


class AgentManager:
    """
    Class to train, optimize hyperparameters, evaluate and gather
    statistics about an agent.

    Notes
    -----
    If parallelization="process" and mp_context="spawn" or mp_context="forkserver", make sure your main code
    has a guard `if __name__ == '__main__'`. See https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming.

    Parameters
    ----------
    agent_class
        Class of the agent.
    train_env : tuple (constructor, kwargs)
        Enviroment used to initialize/train the agent.
    fit_budget : int
        Budget used to call :meth:`rlberry.agents.agent.Agent.fit`.
        If None, must be given in ``fit_kwargs['fit_budget']``.
    eval_env : Tuple (constructor, kwargs)
        Environment used to evaluate the agent. If None, set to ``train_env``.
    init_kwargs : dict
        Arguments required by the agent's constructor. Shared across all n_fit instances.
    fit_kwargs : dict
        Extra arguments to call :meth:`rlberry.agents.agent.Agent.fit`.
    eval_kwargs : dict
        Arguments required to call :meth:`rlberry.agents.agent.Agent.eval`.
    agent_name : str
        Name of the agent. If None, set to agent_class.name
    n_fit : int
        Number of agent instances to fit.
    output_dir : str or :class:`pathlib.Path`
        Directory where to store data.
    parallelization: {'thread', 'process'}, default: 'thread'
        Whether to parallelize  agent training using threads or processes.
    max_workers: None or int, default: None
        Number of processes/threads used in a call to fit().
        If None and parallelization='process', it will default to the
        number of processors on the machine.
        If None and parallelization='thread', it will default to the
        number of processors on the machine, multiplied by 5.
    mp_context: {'spawn', 'fork', 'forkserver}, default: 'spawn'.
        Context for python multiprocessing module.
        Warning: If you're using JAX or PyTorch, it only works with 'spawn'.
                 If running code on a notebook or interpreter, use 'fork'.
                 forkserver and fork are available on Unix OS only.
        worker_logging_level : str, default: None
            Logging level in each of the threads/processes used to fit agents.
            If None, use default logger level.
    seed : :class:`numpy.random.SeedSequence`, :class:`~rlberry.seeding.seeder.Seeder` or int, default : None
        Seed sequence from which to spawn the random number generator.
        If None, generate random seed.
        If int, use as entropy for SeedSequence.
        If seeder, use seeder.seed_seq
    enable_tensorboard : bool, default : False
        If True, enable tensorboard logging in Agent's :class:`~rlberry.utils.writers.DefaultWriter`.
    outdir_id_style: {None, 'unique', 'timestamp'}, default = 'timestamp'
        If None, data is saved to output_dir/manager_data
        If 'unique', data is saved to ``output_dir/manager_data/<AGENT_NAME_UNIQUE_ID>``
        If 'timestamp', data is saved to ``output_dir/manager_data/<AGENT_NAME_TIMESTAMP_SHORT_ID>``
    default_writer_kwargs : dict
        Optional arguments for :class:`~rlberry.utils.writers.DefaultWriter`.
    init_kwargs_per_instance : List[dict] (optional)
        List of length ``n_fit`` containing the params to initialize each of
        the ``n_fit`` agent instances. It can be useful if different instances
        require different parameters. If the same parameter is defined by
        ``init_kwargs`` and ``init_kwargs_per_instance``, the value given by
        ``init_kwargs_per_instance`` will be used.
        Attention: parameters that are passed individually to each agent instance
        cannot be optimized in the method optimize_hyperparams().
    thread_shared_data : dict, optional
        Data to be shared among agent instances in different threads.
        If parallelization='process', data will be copied instead of shared.

    Attributes
    ----------
    output_dir : :class:`pathlib.Path`
        Directory where the manager saves data.
    """

    def __init__(
        self,
        agent_class,
        train_env=(None, None),
        fit_budget=None,
        eval_env=None,
        init_kwargs=None,
        fit_kwargs=None,
        eval_kwargs=None,
        agent_name=None,
        n_fit=4,
        output_dir=None,
        parallelization="thread",
        max_workers=None,
        mp_context="spawn",
        worker_logging_level=None,
        seed=None,
        enable_tensorboard=False,
        outdir_id_style="timestamp",
        default_writer_kwargs=None,
        init_kwargs_per_instance=None,
        thread_shared_data=None,
    ):
        # agent_class should only be None when the constructor is called
        # by the class method AgentManager.load(), since the agent class
        # will be loaded.

        if agent_class is None:
            return None  # Must only happen when load() method is called.

        self.seeder = Seeder(seed)
        self.eval_seeder = self.seeder.spawn(1)

        self.agent_name = agent_name
        if agent_name is None:
            self.agent_name = agent_class.name

        # Check train_env and eval_env
        assert isinstance(
            train_env, Tuple
        ), "[AgentManager]train_env must be Tuple (constructor, kwargs)"
        if eval_env is not None:
            assert isinstance(
                eval_env, Tuple
            ), "[AgentManager]train_env must be Tuple (constructor, kwargs)"

        # check options
        assert outdir_id_style in [None, "unique", "timestamp"]

        # create oject identifier
        self.unique_id = metadata_utils.get_unique_id(self)
        self.timestamp_id = metadata_utils.get_readable_id(self)

        # Agent class
        self.agent_class = agent_class

        # Train env
        self.train_env = train_env

        # Check eval_env
        if eval_env is None:
            eval_env = deepcopy(train_env)

        self._eval_env = eval_env

        # shared data
        self.thread_shared_data = thread_shared_data  # do not deepcopy for sharing!
        if parallelization != "thread" and thread_shared_data is not None:
            logger.warning(
                f"Using thread_shared_data and parallelization = {parallelization}"
                " in AgentManager does *not* share data among Agent instances!"
                " Each process will have its copy of thread_shared_data."
            )

        # check kwargs
        fit_kwargs = fit_kwargs or {}
        eval_kwargs = eval_kwargs or {}

        # params
        base_init_kwargs = init_kwargs or {}

        self._base_init_kwargs = deepcopy(base_init_kwargs)
        self.fit_kwargs = deepcopy(fit_kwargs)
        self.eval_kwargs = deepcopy(eval_kwargs)
        self.n_fit = n_fit
        self.parallelization = parallelization
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.worker_logging_level = worker_logging_level or logging.getLevelName(
            logger.getEffectiveLevel()
        )
        self.output_dir = output_dir
        if fit_budget is not None:
            self.fit_budget = fit_budget
        else:
            try:
                self.fit_budget = self.fit_kwargs.pop("fit_budget")
            except KeyError:
                raise ValueError("[AgentManager] fit_budget missing in __init__().")
        # extra params per instance
        if init_kwargs_per_instance is not None:
            assert len(init_kwargs_per_instance) == n_fit
            init_kwargs_per_instance = deepcopy(init_kwargs_per_instance)
        self.init_kwargs_per_instance = init_kwargs_per_instance or [
            dict() for _ in range(n_fit)
        ]

        # output dir
        if output_dir is None:
            output_dir_ = metadata_utils.RLBERRY_TEMP_DATA_DIR
        else:
            output_dir_ = output_dir
        self.output_dir_ = Path(output_dir_) / "manager_data"
        if outdir_id_style == "unique":
            self.output_dir_ = self.output_dir_ / (
                self.agent_name + "_" + self.unique_id
            )
        elif outdir_id_style == "timestamp":
            self.output_dir_ = self.output_dir_ / (
                self.agent_name + "_" + self.timestamp_id
            )

        # Create list of writers for each agent that will be trained
        # 'default' will keep Agent's use of DefaultWriter.
        self.writers = [("default", None) for _ in range(n_fit)]

        # Parameters to setup Agent's DefaultWriter
        self.agent_default_writer_kwargs = [
            dict(
                name=self.agent_name,
                log_interval=3,
                tensorboard_kwargs=None,
                execution_metadata=metadata_utils.ExecutionMetadata(obj_worker_id=idx),
            )
            for idx in range(n_fit)
        ]
        self.tensorboard_dir = None
        if enable_tensorboard:
            self.tensorboard_dir = self.output_dir_ / "tensorboard"
            for idx, params in enumerate(self.agent_default_writer_kwargs):
                params["tensorboard_kwargs"] = dict(
                    log_dir=self.tensorboard_dir / str(idx)
                )
        # Update DefaultWriter according to user's settings.
        default_writer_kwargs = default_writer_kwargs or {}
        if default_writer_kwargs:
            logger.warning(
                "(Re)defining the following DefaultWriter"
                f" parameters in AgentManager: {list(default_writer_kwargs.keys())}"
            )
        for ii in range(n_fit):
            self.agent_default_writer_kwargs[ii].update(default_writer_kwargs)

        # agent handlers and init kwargs
        self._set_init_kwargs()  # init_kwargs for each agent
        self.agent_handlers = None
        self._reset_agent_handlers()
        self.default_writer_data = None
        self.best_hyperparams = None

        # optuna study and database
        self.optuna_study = None
        self.db_filename = None
        self.optuna_storage_url = None

        # rlberry version for reproducibility purpose
        self.rlberry_version = rlberry.__version__

    def _init_optuna_storage_url(self):
        self.output_dir_.mkdir(parents=True, exist_ok=True)
        self.db_filename = self.output_dir_ / "optuna_data.db"

        if create_database(self.db_filename):
            self.optuna_storage_url = f"sqlite:///{self.db_filename}"
        else:
            self.db_filename = None
            self.optuna_storage_url = "sqlite:///:memory:"
            logger.warning(
                f"Unable to create databate {self.db_filename}. Using sqlite:///:memory:"
            )

    def _set_init_kwargs(self):
        init_seeders = self.seeder.spawn(self.n_fit, squeeze=False)
        self.init_kwargs = []
        for ii in range(self.n_fit):
            # deepcopy base_init_kwargs
            kwargs_ii = deepcopy(self._base_init_kwargs)
            # include shared data, without deep copy!
            kwargs_ii["_thread_shared_data"] = self.thread_shared_data
            kwargs_ii.update(
                dict(
                    env=self.train_env,
                    eval_env=self._eval_env,
                    copy_env=False,
                    seeder=init_seeders[ii],
                    output_dir=Path(self.output_dir_) / f"output_{ii}",
                    _execution_metadata=self.agent_default_writer_kwargs[ii][
                        "execution_metadata"
                    ],
                    _default_writer_kwargs=self.agent_default_writer_kwargs[ii],
                )
            )
            per_instance_kwargs = self.init_kwargs_per_instance[ii]
            kwargs_ii.update(per_instance_kwargs)
            self.init_kwargs.append(kwargs_ii)

    def _reset_agent_handlers(self):
        handlers_seeders = self.seeder.spawn(self.n_fit, squeeze=False)
        self.agent_handlers = [
            AgentHandler(
                id=ii,
                filename=self.output_dir_ / Path(f"agent_handlers/idx_{ii}"),
                seeder=handlers_seeders[ii],
                agent_class=self.agent_class,
                agent_instance=None,
                # kwargs
                agent_kwargs=self.init_kwargs[ii],
            )
            for ii in range(self.n_fit)
        ]
        self.clear_handlers()

    def build_eval_env(self) -> types.Env:
        """Return an instantiated and reseeded evaluation environment.

        Returns
        -------
        :class:`types.Env`
            Instance of evaluation environment.
        """
        return process_env(self._eval_env, self.seeder)

    def get_writer_data(self):
        """Return a dataframe containing data from the writer of the agents.

        Returns
        -------
        :class:`pandas.DataFrame`
            Data from the agents' writers.
        """
        return self.default_writer_data

    def get_agent_instances(self):
        """Returns a list containing ``n_fit`` agent instances.

        Returns
        -------
        list of :class:`~rlberry.agents.agent.Agent`
            ``n_fit`` instances of the managed agents.
        """
        if self.agent_handlers:
            return [
                agent_handler.get_instance() for agent_handler in self.agent_handlers
            ]
        return []

    def eval_agents(
        self,
        n_simulations: Optional[int] = None,
        eval_kwargs: Optional[dict] = None,
        agent_id: Optional[int] = None,
        verbose: Optional[bool] = True,
    ) -> List[float]:
        """
        Call :meth:`eval` method in the managed agents and returns a list with the results.

        Parameters
        ----------
        n_simulations : int, optional
            Total number of agent evaluations. If None, set to 2*(number of agents)
        eval_kwargs : dict, optional
            Arguments to be sent to the .eval() method of each trained instance.
            If None, set to self.eval_kwargs.
        agent_id: int, optional
            Index of the agent to be evaluated. If None, choose randomly.
        verbose: bool, optional
            Whether to print a progress report.

        Returns
        -------
        list
            list of length ``n_simulations`` containing the outputs
            of :meth:`~rlberry.agents.agent.Agent.eval`.
        """
        eval_kwargs = eval_kwargs or self.eval_kwargs
        if not n_simulations:
            n_simulations = 2 * self.n_fit
        values = []

        if logger.getEffectiveLevel() > 10:
            previous_handlers = logger.handlers
            ch = logging.StreamHandler()
            ch.terminator = ""
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            logger.handlers = [ch]
            logger.info("[INFO] Evaluation:")
        for ii in range(n_simulations):
            if agent_id is None:
                # randomly choose one of the fitted agents
                agent_idx = self.eval_seeder.rng.choice(len(self.agent_handlers))
            else:
                agent_idx = agent_id
            agent = self.agent_handlers[agent_idx]
            if agent.is_empty():
                logger.error(
                    "Calling eval() in an AgentManager instance contaning an empty AgentHandler."
                    " Returning []."
                )
                return []
            values.append(agent.eval(**eval_kwargs))
            if logger.getEffectiveLevel() <= 10:  # If debug
                logger.debug(f"[eval]... simulation {ii + 1}/{n_simulations}")
            else:
                logger.info(".")
        if logger.getEffectiveLevel() > 10:
            logger.info("  Evaluation finished \n")
            logger.handlers = previous_handlers

        return values

    def clear_output_dir(self):
        """Delete output_dir and all its data."""
        try:
            shutil.rmtree(self.output_dir_)
        except FileNotFoundError:
            logger.warning(f"No directory {self.output_dir_} found to be deleted.")

    def clear_handlers(self):
        """Delete files from output_dir/agent_handlers that are managed by this class."""
        for handler in self.agent_handlers:
            if handler._fname.exists():
                handler._fname.unlink()

    def set_writer(self, idx, writer_fn, writer_kwargs=None):
        """Defines the writer for one of the managed agents.

        Note
        -----
        Must be called right after creating an instance of AgentManager.

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
            AgentManager fits `n_fit` agents, the writer of each one of them
            needs to be set separetely.
        """
        assert (
            idx >= 0 and idx < self.n_fit
        ), "Invalid index sent to AgentManager.set_writer()"
        writer_kwargs = writer_kwargs or {}
        self.writers[idx] = (writer_fn, writer_kwargs)

    def generate_profile(self, budget=None, fname=None):
        """
        Do a fit to produce a profile (i.e. the cumulative time spent on each operation done during a fit).
        The 20 first lines are printed out and the whole profile is saved in a file.
        See `https://docs.python.org/3/library/profile.html`_ for more information on python profiler.

        Parameters
        ----------
        budget: int or None, default=None
            budget of the fit done to generate the profile

        fname: string or None, default=None
            name of the file where we save the profile. If None, the file is saved in self.output_dir/self.agent_name_profile.prof.
        """
        budget = budget or self.fit_budget

        if self.output_dir is None:
            output_dir_ = metadata_utils.RLBERRY_TEMP_DATA_DIR
        else:
            output_dir_ = self.output_dir

        filename = fname or (str(output_dir_) + self.agent_name + "_profile.prof")
        logger.info("Doing a profile run.")
        with cProfile.Profile() as pr:
            agent = self.agent_class(**(self.init_kwargs[0]))
            agent.fit(budget, **deepcopy(self.fit_kwargs))
        pr.dump_stats(filename)
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr).sort_stats(sortby)
        logger.info("Printing the 20 first lines of the profile")
        ps.print_stats(20)

    def fit(self, budget=None, **kwargs):
        """Fit the agent instances in parallel.

        Parameters
        ----------
        budget: int or None
            Computational or sample complexity budget.
        """
        del kwargs
        budget = budget or self.fit_budget

        # If spawn, test that protected by if __name__ == "__main__"
        if self.mp_context == "spawn":
            try:
                _check_not_importing_main()
            except RuntimeError as exc:
                raise RuntimeError(
                    """Warning: in AgentManager, if mp_context='spawn' and
                        parallelization="process" then the script must be run
                        outside a notebook and protected by a  if __name__ == '__main__':
                        For example:
                            if __name__ == '__main__':
                                agent = AgentManager(UCBVIAgent,(Chain, {}),
                                                mp_context="spawn",
                                                parallelization="process")

                                agent.fit(10)
                                   """
                ) from exc

        logger.info(
            f"Running AgentManager fit() for {self.agent_name}"
            f" with n_fit = {self.n_fit} and max_workers = {self.max_workers}."
        )
        seeders = self.seeder.spawn(self.n_fit)
        if not isinstance(seeders, list):
            seeders = [seeders]

        # remove agent instances from memory so that the agent handlers can be
        # sent to different workers
        for handler in self.agent_handlers:
            handler.dump()

        if self.parallelization == "thread":
            executor_class = concurrent.futures.ThreadPoolExecutor
            lock = threading.Lock()
        elif self.parallelization == "process":
            executor_class = functools.partial(
                concurrent.futures.ProcessPoolExecutor,
                mp_context=multiprocessing.get_context(self.mp_context),
            )
            lock = multiprocessing.Manager().Lock()
        else:
            raise ValueError(
                f"Invalid backend for parallelization: {self.parallelization}"
            )

        args = [
            (
                lock,
                handler,
                self.agent_class,
                budget,
                init_kwargs,
                deepcopy(self.fit_kwargs),
                writer,
                self.worker_logging_level,
                seeder,
            )
            for init_kwargs, handler, seeder, writer in zip(
                self.init_kwargs, self.agent_handlers, seeders, self.writers
            )
        ]

        if len(args) == 1:
            workers_output = [_fit_worker(args[0])]

        else:
            with executor_class(max_workers=self.max_workers) as executor:
                futures = []
                for arg in args:
                    futures.append(executor.submit(_fit_worker, arg))

                workers_output = []
                for future in concurrent.futures.as_completed(futures):
                    workers_output.append(future.result())
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

    def save(self):
        """Save AgentManager data to :attr:`~rlberry.manager.agent_manager.AgentManager.output_dir`.

        Saves object so that the data can be later loaded to recreate an AgentManager instance.

        Returns
        -------
        :class:`pathlib.Path`
            Filename where the AgentManager object was saved.
        """
        # use self.output_dir
        output_dir = self.output_dir_
        output_dir = Path(output_dir)

        # create dir if it does not exist
        output_dir.mkdir(parents=True, exist_ok=True)
        # save optimized hyperparameters
        if self.best_hyperparams is not None:
            fname = Path(output_dir) / "best_hyperparams.json"
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
                    fname = Path(output_dir) / "data.csv"
                    output.to_csv(fname, index=None)
                except Exception:
                    logger.warning("Could not save default_writer_data.")

        #
        # Pickle AgentManager instance
        #

        # clear agent handlers
        for handler in self.agent_handlers:
            handler.dump()

        # save
        filename = Path("manager_obj").with_suffix(".pickle")
        filename = output_dir / filename
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            with filename.open("wb") as ff:
                pickle.dump(self.__dict__, ff)
            logger.info("Saved AgentManager({}) using pickle.".format(self.agent_name))
        except Exception:
            try:
                with filename.open("wb") as ff:
                    dill.dump(self.__dict__, ff)
                logger.info(
                    "Saved AgentManager({}) using dill.".format(self.agent_name)
                )
            except Exception as ex:
                logger.warning("[AgentManager] Instance cannot be pickled: " + str(ex))

        return filename

    @classmethod
    def load(cls, filename):
        """Loads an AgentManager instance from a file.

        Parameters
        ----------
        filename: str or :class:`pathlib.Path`

        Returns
        -------
        :class:`rlberry.manager.AgentManager`
            Loaded instance of AgentManager.
        """
        filename = Path(filename).with_suffix(".pickle")

        obj = cls(None, None, None)
        try:
            with filename.open("rb") as ff:
                tmp_dict = pickle.load(ff)
            logger.info("Loaded AgentManager using pickle.")
        except Exception:
            with filename.open("rb") as ff:
                tmp_dict = dill.load(ff)
            logger.info("Loaded AgentManager using dill.")

        obj.__dict__.clear()
        obj.__dict__.update(tmp_dict)

        return obj

    def __eq__(self, other):

        result = True
        self_init_kwargs = [_strip_seed_dir(kw) for kw in self.init_kwargs]
        other_init_kwargs = [_strip_seed_dir(kw) for kw in other.init_kwargs]
        result = result and all(
            [
                self_init_kwargs[f] == other_init_kwargs[f]
                for f in range(len(self_init_kwargs))
            ]
        )

        self_eval_kwargs = self.eval_kwargs or {}
        other_eval_kwargs = other.eval_kwargs or {}
        result = result and (self_eval_kwargs == other_eval_kwargs)

        result = result and (other.agent_class == self.agent_class)

        result = result and (self.fit_kwargs == other.fit_kwargs)

        result = result and (self.fit_budget == other.fit_budget)

        return result

    def optimize_hyperparams(
        self,
        n_trials=256,
        timeout=60,
        n_fit=2,
        n_optuna_workers=2,
        optuna_parallelization="thread",
        sampler_method="optuna_default",
        pruner_method="halving",
        continue_previous=False,
        fit_fraction=1.0,
        sampler_kwargs=None,
        disable_evaluation_writers=True,
        custom_eval_function=None,
    ):
        """Run hyperparameter optimization and updates init_kwargs with the best hyperparameters found.

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
        * After calling this method, agent handlers from previous calls to fit() will be erased.
        It is suggested to call fit() *after* a call to optimize_hyperparams().
        * This method calls self.save() before the optuna optimization starts, to ensure
        that we can continue the optimization later even if the program is stopped before the
        optimization is finished.

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
        custom_eval_function : Callable
            Takes as input a list of trained agents and output a scalar.
            If given, the value of custom_eval_funct(trained_agents) is
            optimized instead of mean([agent.eval() for agent in trained_agents]).

        Returns
        -------
        dict
            Optimized hyperparameters.
        """
        #
        # setup
        #
        TEMP_DIR = self.output_dir_ / "optim"

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
            if sampler_method == "random":
                sampler = optuna.samplers.RandomSampler()
            elif sampler_method == "grid":
                assert (
                    sampler_kwargs is not None
                ), "To use GridSampler, a search_space dictionary must be provided."
                sampler = optuna.samplers.GridSampler(**sampler_kwargs)
            elif sampler_method == "cmaes":
                sampler = optuna.samplers.CmaEsSampler(**sampler_kwargs)
            elif sampler_method == "optuna_default":
                sampler = optuna.samplers.TPESampler(**sampler_kwargs)
            else:
                raise NotImplementedError(
                    "Sampler method %s is not implemented." % sampler_method
                )

            # get pruner
            if pruner_method == "halving":
                pruner = optuna.pruners.SuccessiveHalvingPruner(
                    min_resource=1, reduction_factor=4, min_early_stopping_rate=0
                )
            elif pruner_method == "none":
                pruner = None
            else:
                raise NotImplementedError(
                    "Pruner method %s is not implemented." % pruner_method
                )

            # storage
            self._init_optuna_storage_url()
            storage = optuna.storages.RDBStorage(self.optuna_storage_url)

            # optuna study
            study = optuna.create_study(
                sampler=sampler, pruner=pruner, storage=storage, direction="maximize"
            )
            self.optuna_study = study

        # save, to that optimization can be resumed later
        self.save()

        #
        # Objective function
        #
        objective = functools.partial(
            _optuna_objective,
            base_init_kwargs=self._base_init_kwargs,  # self._base_init_kwargs
            agent_class=self.agent_class,  # self.agent_class
            train_env=self.train_env,  # self.train_env
            eval_env=self._eval_env,
            fit_budget=self.fit_budget,  # self.fit_budget
            eval_kwargs=self.eval_kwargs,  # self.eval_kwargs
            n_fit=n_fit,
            temp_dir=TEMP_DIR,  # TEMP_DIR
            disable_evaluation_writers=disable_evaluation_writers,
            fit_fraction=fit_fraction,
            init_kwargs_per_instance=self.init_kwargs_per_instance[
                :n_fit
            ],  # init_kwargs_per_instance only for the first n_fit instances
            custom_eval_function=custom_eval_function,
            thread_shared_data=self.thread_shared_data,
        )

        try:
            if optuna_parallelization == "thread":
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for _ in range(n_optuna_workers):
                        executor.submit(
                            study.optimize,
                            objective,
                            n_trials=n_trials,
                            timeout=timeout,
                            gc_after_trial=True,
                        )
                    executor.shutdown()
            elif optuna_parallelization == "process":
                with concurrent.futures.ProcessPoolExecutor(
                    mp_context=multiprocessing.get_context(self.mp_context)
                ) as executor:
                    for _ in range(n_optuna_workers):
                        executor.submit(
                            study.optimize,
                            objective,
                            n_trials=n_trials // n_optuna_workers,
                            timeout=timeout,
                            gc_after_trial=True,
                        )
                    executor.shutdown()
            else:
                raise ValueError(
                    f"Invalid value for optuna_parallelization: {optuna_parallelization}."
                )

        except KeyboardInterrupt:
            logger.warning("Evaluation stopped.")

        # clear temp folder
        try:
            shutil.rmtree(TEMP_DIR)
        except FileNotFoundError as ex:
            logger.warning(f"Could not delete {TEMP_DIR}: {ex}")

        # continue
        try:
            best_trial = study.best_trial
        except ValueError as ex:
            logger.error(f"Hyperparam optimization failed due to the error: {ex}")
            return dict()

        logger.info(f"Number of finished trials: {len(study.trials)}")
        logger.info("Best trial:")
        logger.info(f"Value: {best_trial.value}")
        logger.info("Params:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

        # store best parameters
        self.best_hyperparams = best_trial.params

        # update using best parameters
        self._base_init_kwargs.update(best_trial.params)

        # reset init_kwargs and agent handlers, so that they take the new
        # parameters
        self._set_init_kwargs()
        self._reset_agent_handlers()

        return deepcopy(best_trial.params)


#
# Aux functions
#


def _fit_worker(args):
    """Create and fit an agent instance"""
    (
        lock,
        agent_handler,
        agent_class,
        fit_budget,
        init_kwargs,
        fit_kwargs,
        writer,
        worker_logging_level,
        seeder,
    ) = args

    # reseed external libraries
    set_external_seed(seeder)

    # logging level in thread
    configure_logging(worker_logging_level)

    # Using a lock when creating envs and agents, to avoid problems
    # as here: https://github.com/openai/gym/issues/281
    with lock:
        if agent_handler.is_empty():
            # create agent
            agent = agent_class(**init_kwargs)
            # seed agent
            # TODO: check if extra reseeding here is necessary
            agent.reseed(seeder)
            agent_handler.set_instance(agent)

    # set writer
    if writer[0] is None:
        agent_handler.set_writer(None)
    elif (
        writer[0] != "default"
    ):  # 'default' corresponds to DefaultWriter created by Agent.__init__()
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

    # garbage collector
    gc.collect()

    return agent_handler


def _safe_serialize_json(obj, filename):
    """
    Source: https://stackoverflow.com/a/56138540/5691288
    """

    def default(obj):
        return f"<<non-serializable: {type(obj).__qualname__}>>"

    with open(filename, "w") as fp:
        json.dump(obj, fp, sort_keys=True, indent=4, default=default)


def _optuna_objective(
    trial,
    base_init_kwargs,  # self._base_init_kwargs
    agent_class,  # self.agent_class
    train_env,  # self.train_env
    eval_env,
    fit_budget,  # self.fit_budget
    eval_kwargs,  # self.eval_kwargs
    n_fit,
    temp_dir,  # TEMP_DIR
    disable_evaluation_writers,
    fit_fraction,
    init_kwargs_per_instance,
    custom_eval_function,
    thread_shared_data,
):
    kwargs = deepcopy(base_init_kwargs)

    # will raise exception if sample_parameters() is not
    # implemented by the agent class
    kwargs.update(agent_class.sample_parameters(trial))

    #
    # fit and evaluate agents
    #
    # Create AgentManager with hyperparams
    params_stats = AgentManager(
        agent_class,
        train_env,
        fit_budget,
        eval_env=eval_env,
        init_kwargs=kwargs,  # kwargs are being optimized
        eval_kwargs=deepcopy(eval_kwargs),
        agent_name="optim",
        n_fit=n_fit,
        worker_logging_level="INFO",
        parallelization="thread",
        output_dir=temp_dir,
        enable_tensorboard=False,
        outdir_id_style="unique",
        init_kwargs_per_instance=init_kwargs_per_instance,
        thread_shared_data=thread_shared_data,
    )

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
            if not custom_eval_function:
                eval_value = np.mean(params_stats.eval_agents())
            else:
                eval_value = custom_eval_function(params_stats.get_agent_instances())

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
        if not custom_eval_function:
            eval_value = np.mean(params_stats.eval_agents())
        else:
            eval_value = custom_eval_function(params_stats.get_agent_instances())

    # clear aux data
    params_stats.clear_output_dir()
    del params_stats

    return eval_value


def _strip_seed_dir(dico):
    """Remove keys that should not be compared in __eq__"""
    res = deepcopy(dico)
    del res["seeder"]
    del res["output_dir"]
    return res
