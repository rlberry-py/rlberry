from abc import ABC, abstractmethod
import dill
import pickle
import bz2
import _pickle as cPickle
import numpy as np
from inspect import signature
from pathlib import Path
from rlberry import metadata_utils
from rlberry import types
from rlberry.seeding.seeder import Seeder
from rlberry.seeding import safe_reseed
from rlberry.envs.utils import process_env
from rlberry.utils.writers import DefaultWriter
from typing import Optional
import inspect

import rlberry

logger = rlberry.logger


class Agent(ABC):
    """Basic interface for agents.

    If the 'inherited class' from Agent use the torch lib, that is higly recommanded to inherit :class:`~rlberry.agents.AgentTorch` instead.

    .. note::
        | 1 - Abstract Class : can't be cannot be instantiated. The abstract methods have to be overwriten by the 'inherited class' agent.
        | 2 - Classes that implements this interface can send `**kwargs` to initiate :code:`Agent.__init__()`, but the keys must match the parameters.

    Parameters
    ----------
    env : :class:`gymnasium.Env` or tuple (constructor, kwargs)
        Environment on which to train the agent.
    eval_env : :class:`gymnasium.Env` or tuple (constructor, kwargs)
        Environment on which to evaluate the agent. If None, copied from env.
    copy_env : bool
        If true, makes a deep copy of the environment.
    compress_pickle : bool
        If true, compress the save files using bz2.
    seeder : :class:`~rlberry.seeding.seeder.Seeder`, int, or None
        Seeder/seed for random number generation.
    output_dir : str or Path
        Directory that the agent can use to store data.
    _execution_metadata : ExecutionMetadata, optional
        Extra information about agent execution (e.g. about which is the process id where the agent is running).
        Used by :class:`~rlberry.manager.ExperimentManager`.
    _default_writer_kwargs : dict, optional
        Parameters to initialize :class:`~rlberry.utils.writers.DefaultWriter` (attribute self.writer).
        Used by :class:`~rlberry.manager.ExperimentManager`.
    _thread_shared_data : dict, optional
        Used by :class:`~rlberry.manager.ExperimentManager` to share data across Agent
        instances created in different threads.

    Attributes
    ----------
    name : string
        Agent identifier (not necessarily unique).
    env : :class:`gymnasium.Env` or tuple (constructor, kwargs)
        Environment on which to train the agent.
    eval_env : :class:`gymnasium.Env` or tuple (constructor, kwargs)
        Environment on which to evaluate the agent. If None, copied from env.
    writer : object, default: None
        Writer object to log the output (e.g. tensorboard SummaryWriter).
    seeder : :class:`~rlberry.seeding.seeder.Seeder`, int, or None
        Seeder/seed for random number generation.
    rng : :class:`numpy.random._generator.Generator`
        Random number generator. If you use random numbers in your agent, this
        attribute must be used in order to ensure reproducibility. See `numpy's
        documentation <https://numpy.org/doc/stable/reference/random/generator.html>`_.
    output_dir : str or Path
        Directory that the agent can use to store data.
    unique_id : str
        Unique identifier for the agent instance. Can be used, for example,
        to create files/directories for the agent to log data safely.
    thread_shared_data : dict
        Data shared by agent instances among different threads.
    """

    name = ""

    def __init__(
        self,
        env: types.Env = None,
        eval_env: Optional[types.Env] = None,
        copy_env: bool = True,
        compress_pickle: bool = True,
        seeder: Optional[types.Seed] = None,
        output_dir: Optional[str] = None,
        _execution_metadata: Optional[metadata_utils.ExecutionMetadata] = None,
        _default_writer_kwargs: Optional[dict] = None,
        _thread_shared_data: Optional[dict] = None,
    ):
        self.seeder = Seeder(seeder)
        self.env = process_env(env, self.seeder, copy_env=copy_env)

        self.compress_pickle = compress_pickle
        # evaluation environment
        eval_env = eval_env or env
        self.eval_env = process_env(eval_env, self.seeder, copy_env=copy_env)

        # metadata
        self._execution_metadata = (
            _execution_metadata or metadata_utils.ExecutionMetadata()
        )
        self._unique_id = metadata_utils.get_unique_id(self)
        if self.name:
            self._unique_id = self.name + "_" + self._unique_id

        # create writer
        _default_writer_kwargs = _default_writer_kwargs or dict(
            name=self.name, execution_metadata=self._execution_metadata
        )
        self._writer = DefaultWriter(**_default_writer_kwargs)

        # output directory for the agent instance
        self._output_dir = output_dir or f"output_{self._unique_id}"
        self._output_dir = Path(self._output_dir)

        # shared data among threads
        self._thread_shared_data = _thread_shared_data

    @property
    def writer(self):
        """
        Writer object to log the output (e.g. tensorboard SummaryWriter)..
        """
        return self._writer

    @property
    def unique_id(self):
        """
        Unique identifier for the agent instance. Can be used, for example, to create files/directories for the agent to log data safely.
        """
        return self._unique_id

    @property
    def output_dir(self):
        """
        Directory that the agent can use to store data.
        """
        return self._output_dir

    @property
    def rng(self):
        """
        Random number generator.
        """
        return self.seeder.rng

    @property
    def thread_shared_data(self):
        """
        Data shared by agent instances among different threads.
        """
        if self._thread_shared_data is None:
            return dict()
        return self._thread_shared_data

    @abstractmethod
    def fit(self, budget: int, **kwargs):
        """
        Abstract method to be overwriten by the 'inherited agent' developer.

        Train the agent with a fixed budget, using the provided environment.

        Parameters
        ----------
        budget: int
            Computational (or sample complexity) budget.
            It can be, for instance:

            * The number of timesteps taken by the environment (env.step) or the number of episodes;

            * The number of iterations for algorithms such as value/policy iteration;

            * The number of searches in MCTS (Monte-Carlo Tree Search) algorithms;

            among others.

            Ideally, calling

            .. code-block:: python

                fit(budget1)
                fit(budget2)

            should be equivalent to one call

            .. code-block:: python

                fit(budget1 + budget2)

            This property is required to reduce the time required for hyperparameter
            optimization (by allowing early stopping), but it is not strictly required
            elsewhere in the library.

            If the agent does not require a budget, set it to -1.
        **kwargs: Keyword Arguments
            Extra parameters specific to the implemented fit.
        """
        pass

    @abstractmethod
    def eval(self, **kwargs):
        """
        Abstract method.

        Returns a float measuring the quality of the agent (e.g. MC policy evaluation).

        Parameters
        ----------
        eval_env: object
            Environment for evaluation.
        **kwargs: Keyword Arguments
            Extra parameters specific to the implemented evaluation.
        """
        pass

    def set_writer(self, writer):
        """set self._writer. If is not None, add parameters values to writer."""
        self._writer = writer

        if self._writer:
            init_args = signature(self.__init__).parameters
            kwargs = [f"| {key} | {getattr(self, key, None)} |" for key in init_args]
            writer.add_text(
                "Hyperparameters",
                "| Parameter | Value |\n|-------|-------|\n" + "\n".join(kwargs),
            )

    @classmethod
    def sample_parameters(cls, trial):
        """
        Sample hyperparameters for hyperparam optimization using
        Optuna (https://optuna.org/)

        Note: only the kwargs sent to __init__ are optimized. Make sure to
        include in the Agent constructor all "optimizable" parameters.

        Parameters
        ----------
        trial: optuna.trial
        """
        raise NotImplementedError("agent.sample_parameters() not implemented.")

    def reseed(self, seed_seq=None):
        """
        Get new random number generator for the agent.

        Parameters
        ----------
        seed_seq : :class:`numpy.random.SeedSequence`, :class:`rlberry.seeding.seeder.Seeder` or int, default : None
            Seed sequence from which to spawn the random number generator.
            If None, generate random seed.
            If int, use as entropy for SeedSequence.
            If seeder, use seeder.seed_seq
        """
        # self.seeder
        if seed_seq is None:
            self.seeder = self.seeder.spawn()
        else:
            self.seeder = Seeder(seed_seq)
        safe_reseed(self.env, self.seeder)
        safe_reseed(self.eval_env, self.seeder)

    def save(self, filename):
        """
        Save agent object. By default, the agent is pickled.

        If overridden, the load() method must also be overriden.

        Before saving, consider setting writer to None if it can't be pickled (tensorboard writers
        keep references to files and cannot be pickled).

        Note: dill[1]_ is used when pickle fails
        (see https://stackoverflow.com/a/25353243, for instance).
        Pickle is tried first, since it is faster.

        Parameters
        ----------
        filename: Path or str
            File in which to save the Agent.

        Returns
        -------
        pathlib.Path
            If save() is successful, a Path object corresponding to the filename is returned.
            Otherwise, None is returned.

        .. warning:: The returned filename might differ from the input filename: For instance,
        the method can append the correct suffix to the name before saving.

        References
        ----------
        .. [1] https://github.com/uqfoundation/dill
        """
        # remove writer if not pickleable
        if not dill.pickles(self.writer):
            self.set_writer(None)
        # save
        filename = Path(filename).with_suffix(".pickle")
        filename.parent.mkdir(parents=True, exist_ok=True)

        dict_to_save = dict(self.__dict__)

        try:
            if not self.compress_pickle:
                with filename.open("wb") as ff:
                    pickle.dump(dict_to_save, ff)
            else:
                with bz2.BZ2File(filename, "wb") as ff:
                    cPickle.dump(dict_to_save, ff)
        except Exception as ex:
            try:
                if not self.compress_pickle:
                    with filename.open("wb") as ff:
                        dill.dump(dict_to_save, ff)
                else:
                    with bz2.BZ2File(filename, "wb") as ff:
                        dill.dump(dict_to_save, ff)
            except Exception as ex:
                logger.warning("Agent instance cannot be pickled: " + str(ex))
                return None

        return filename

    @classmethod
    def load(cls, filename, **kwargs):
        # If overridden, save() method must also be overriden.
        """Load agent object from filepath.

        If overridden, save() method must also be overriden.

        Parameters
        ----------
        filename: str
            Path to the object (pickle) to load.
        **kwargs: Keyword Arguments
            Arguments required by the __init__ method of the Agent subclass to load.
        """

        filename = Path(filename).with_suffix(".pickle")
        obj = cls(**kwargs)

        try:
            if not obj.compress_pickle:
                with filename.open("rb") as ff:
                    tmp_dict = pickle.load(ff)
            else:
                with bz2.BZ2File(filename, "rb") as ff:
                    tmp_dict = cPickle.load(ff)
        except Exception as ex:
            if not obj.compress_pickle:
                with filename.open("rb") as ff:
                    tmp_dict = dill.load(ff)
            else:
                with bz2.BZ2File(filename, "rb") as ff:
                    tmp_dict = dill.load(ff)

        obj.__dict__.clear()
        obj.__dict__.update(tmp_dict)

        return obj

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the Agent"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the Agent parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this agent.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this agent and
            contained subobjects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out


class AgentWithSimplePolicy(Agent):
    """Interface for agents whose policy is a function of observations only.

    Requires a :meth:`policy` method, and a simple evaluation method (Monte-Carlo policy evaluation).

    The :meth:`policy` method takes an observation as input and returns an action.

    .. note::
        | 1 - Abstract Class : can't be cannot be instantiated. The abstract methods have to be overwriten by the 'inherited class' agent.
        | 2 - Classes that implements this interface can send `**kwargs` to initiate :code:`Agent.__init__()` (:class:`~rlberry.agents.Agent`), but the keys must match the parameters.

    Parameters
    ----------
    env : gymnasium.Env or tuple (constructor, kwargs)
        Environment used to fit the agent.
    eval_env : gymnasium.Env or tuple (constructor, kwargs)
        Environment on which to evaluate the agent. If None, copied from env.
    copy_env : bool
        If true, makes a deep copy of the environment.
    compress_pickle : bool
        If true, compress the save files using bz2.
    seeder : :class:`~rlberry.seeding.seeder.Seeder`, int, or None
        Seeder/seed for random number generation.
    output_dir : str or Path
        Directory that the agent can use to store data.
    _execution_metadata : ExecutionMetadata, optional
        Extra information about agent execution (e.g. about which is the process id where the agent is running).
        Used by :class:`~rlberry.manager.ExperimentManager`.
    _default_writer_kwargs : dict, optional
        Parameters to initialize :class:`~rlberry.utils.writers.DefaultWriter` (attribute self.writer).
        Used by :class:`~rlberry.manager.ExperimentManager`.
    _thread_shared_data : dict, optional
        Used by :class:`~rlberry.manager.ExperimentManager` to share data across Agent instances created in different threads.
    **kwargs : dict
        Classes that implement this interface must send ``**kwargs``
        to :code:`AgentWithSimplePolicy.__init__()`.

    Attributes
    ----------
    name : string
        Agent identifier (not necessarily unique).
    env : :class:`gymnasium.Env` or tuple (constructor, kwargs)
        Environment on which to train the agent.
    eval_env : :class:`gymnasium.Env` or tuple (constructor, kwargs)
        Environment on which to evaluate the agent. If None, copied from env.
    writer : object, default: None
        Writer object to log the output(e.g. tensorboard SummaryWriter).
    seeder : :class:`~rlberry.seeding.seeder.Seeder`, int, or None
        Seeder/seed for random number generation.
    rng : :class:`numpy.random._generator.Generator`
        Random number generator. If you use random numbers in your agent, this
        attribute must be used in order to ensure reproducibility. See `numpy's
        documentation <https://numpy.org/doc/stable/reference/random/generator.html>`_.
    output_dir : str or Path
        Directory that the agent can use to store data.
    unique_id : str
        Unique identifier for the agent instance. Can be used, for example,
        to create files/directories for the agent to log data safely..
    thread_shared_data : dict
        Data shared by agent instances among different threads.

    Examples
    --------
    >>> class RandomAgent(AgentWithSimplePolicy):
    >>>     name = "RandomAgent"
    >>>
    >>>     def __init__(self, env, **kwargs):
    >>>         AgentWithSimplePolicy.__init__(self, env, **kwargs)
    >>>
    >>>         def fit(self, budget=100, **kwargs):
    >>>             observation,info = self.env.reset()
    >>>             for ep in range(budget):
    >>>                 action = self.policy(observation)
    >>>                 observation, reward, terminated, truncated, info = self.env.step(action)
    >>>
    >>>         def policy(self, observation):
    >>>             return self.env.action_space.sample()  # choose an action at random
    """

    @abstractmethod
    def policy(self, observation):
        """
        Abstract method.
        The policy function takes an observation from the environment and returns an action.
        The specific implementation of the policy function depends on the agent's learning algorithm
        or strategy, which can be deterministic or stochastic.
        Parameters
        ----------
        observation (any): An observation from the environment.
        Returns
        -------
        action (any): The action to be taken based on the provided observation.
        Notes
        -----
        The data type of 'observation' and 'action' can vary depending on the specific agent
        and the environment it interacts with.
        """
        pass

    def eval(self, eval_horizon=10**5, n_simulations=10, gamma=1.0):
        """
        Monte-Carlo policy evaluation [1]_ method to estimate the mean discounted reward
        using the current policy on the evaluation environment.

        Parameters
        ----------
        eval_horizon : int, optional, default: 10**5
            Maximum episode length, representing the horizon for each simulation.
        n_simulations : int, optional, default: 10
            Number of Monte Carlo simulations to perform for the evaluation.
        gamma : float, optional, default: 1.0
            Discount factor for future rewards.

        Returns
        -------
        float
            The mean value over 'n_simulations' of the sum of rewards obtained in each simulation.

        References
        ----------
        .. [1] Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
            MIT Press.
        """

        episode_rewards = np.zeros(n_simulations)
        for sim in range(n_simulations):
            observation, info = self.eval_env.reset()
            tt = 0
            while tt < eval_horizon:
                action = self.policy(observation)
                observation, reward, terminated, truncated, info = self.eval_env.step(
                    action
                )
                done = terminated or truncated
                episode_rewards[sim] += reward * np.power(gamma, tt)
                tt += 1
                if done:
                    break
        return episode_rewards.mean()


class AgentTorch(Agent):
    # Need a specific save and load to manage torch.
    """
    Abstract Class to inherit for torch agents.

    This class use the 'torch' functions to save/load agents.

    .. note::

        | 1 - Abstract Class : can't be cannot be instantiated. The abstract methods (from Agent) have to be overwriten by the 'inherited class' agent.
        | 2 - Classes that implements this interface can send `**kwargs` to initiate :code:`Agent.__init__()`(:class:`~rlberry.agents.Agent`), but the keys must match the parameters.

    Parameters
    ----------
    env : gymnasium.Env or tuple (constructor, kwargs)
        Environment used to fit the agent.
    eval_env : gymnasium.Env or tuple (constructor, kwargs)
        Environment on which to evaluate the agent. If None, copied from env.
    copy_env : bool
        If true, makes a deep copy of the environment.
    compress_pickle : bool
        If true, compress the save files using bz2.
    seeder : :class:`~rlberry.seeding.seeder.Seeder`, int, or None
        Seeder/seed for random number generation.
    output_dir : str or Path
        Directory that the agent can use to store data.
    _execution_metadata : ExecutionMetadata, optional
        Extra information about agent execution (e.g. about which is the process id where the agent is running).
        Used by :class:`~rlberry.manager.ExperimentManager`.
    _default_writer_kwargs : dict, optional
        Parameters to initialize :class:`~rlberry.utils.writers.DefaultWriter` (attribute self.writer).
        Used by :class:`~rlberry.manager.ExperimentManager`.
    _thread_shared_data : dict, optional
        Used by :class:`~rlberry.manager.ExperimentManager` to share data across Agent
        instances created in different threads.

    Attributes
    ----------
    name : string
        Agent identifier (not necessarily unique).
    env : :class:`gymnasium.Env` or tuple (constructor, kwargs)
        Environment on which to train the agent.
    eval_env : :class:`gymnasium.Env` or tuple (constructor, kwargs)
        Environment on which to evaluate the agent. If None, copied from env.
    writer : object, default: None
        Writer object (e.g. tensorboard SummaryWriter).
    seeder : :class:`~rlberry.seeding.seeder.Seeder`, int, or None
        Seeder/seed for random number generation.
    rng : :class:`numpy.random._generator.Generator`
        Random number generator. If you use random numbers in your agent, this
        attribute must be used in order to ensure reproducibility. See `numpy's
        documentation <https://numpy.org/doc/stable/reference/random/generator.html>`_.
    output_dir : str or Path
        Directory that the agent can use to store data.
    unique_id : str
        Unique identifier for the agent instance. Can be used, for example,
        to create files/directories for the agent to log data safely.
    thread_shared_data : dict
        Data shared by agent instances among different threads.
    """

    def save(self, filename):
        # Overwrite the 'save' method to manage CPU and GPU with torch agent
        # If overridden, the load() method must also be overriden.
        """
        ----- documentation from original save -----

        Save agent object. By default, the agent is pickled.

        Before saving, consider setting writer to None if it can't be pickled (tensorboard writers
        keep references to files and cannot be pickled).

        Note: dill[1]_ is used when pickle fails
        (see https://stackoverflow.com/a/25353243, for instance).
        Pickle is tried first, since it is faster.

        Parameters
        ----------
        filename: Path or str
            File in which to save the Agent.

        Returns
        -------
        pathlib.Path
            If save() is successful, a Path object corresponding to the filename is returned.
            Otherwise, None is returned.

        .. warning:: The returned filename might differ from the input filename: For instance,
        the method can append the correct suffix to the name before saving.

        References
        ----------
        .. [1] https://github.com/uqfoundation/dill
        """

        import torch

        # remove writer if not pickleable
        if not dill.pickles(self.writer):
            self.set_writer(None)
        # save
        filename = Path(filename).with_suffix(".pickle")
        filename.parent.mkdir(parents=True, exist_ok=True)

        dict_to_save = dict(self.__dict__)

        try:
            if not self.compress_pickle:
                with filename.open("wb") as ff:
                    torch.save(dict_to_save, ff, pickle)
            else:
                with bz2.BZ2File(filename, "wb") as ff:
                    torch.save(dict_to_save, ff, cPickle)
        except Exception as ex:
            try:
                if not self.compress_pickle:
                    with filename.open("wb") as ff:
                        torch.save(dict_to_save, ff, dill)
                else:
                    with bz2.BZ2File(filename, "wb") as ff:
                        torch.save(dict_to_save, ff, dill)
            except Exception as ex:
                logger.warning("Agent instance cannot be pickled: " + str(ex))
                return None

        return filename

    @classmethod
    def load(cls, filename, **kwargs):
        # Overwrite 'load' method to manage CPU vs GPU with torch agent.
        # If overridden, save() method must also be overriden.
        """
        ----- documentation from original load -----
        Load agent object.

        Parameters
        ----------
        filename: str
            Path to the object (pickle) to load.
        **kwargs: Keyword Arguments
            Arguments required by the __init__ method of the Agent subclass to load.
        """

        from rlberry.utils.torch import choose_device
        import torch

        device_str = "cuda:best"
        if "device" in kwargs.keys():
            device_str = kwargs.pop("device", None)
        device = choose_device(device_str)

        filename = Path(filename).with_suffix(".pickle")
        obj = cls(**kwargs)

        try:
            if not obj.compress_pickle:
                with filename.open("rb") as ff:
                    tmp_dict = torch.load(ff, map_location=device, pickle_module=pickle)
            else:
                with bz2.BZ2File(filename, "rb") as ff:
                    tmp_dict = torch.load(
                        ff, map_location=device, pickle_module=cPickle
                    )
        except Exception as ex:
            if not obj.compress_pickle:
                with filename.open("rb") as ff:
                    tmp_dict = torch.load(ff, map_location=device, pickle_module=dill)
            else:
                with bz2.BZ2File(filename, "rb") as ff:
                    tmp_dict = torch.load(ff, map_location=device, pickle_module=dill)

        obj.__dict__.clear()
        obj.__dict__.update(tmp_dict)

        obj.device = device
        return obj
