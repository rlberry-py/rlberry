from abc import ABC, abstractmethod
import dill
import pickle
import logging
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


logger = logging.getLogger(__name__)


class Agent(ABC):
    """ Basic interface for agents.

    Parameters
    ----------
    env : gym.Env or tuple (constructor, kwargs)
        Environment used to fit the agent.
    eval_env : gym.Env or tuple (constructor, kwargs)
        Environment on which to evaluate the agent. If None, copied from env.
    copy_env : bool
        If true, makes a deep copy of the environment.
    seeder : rlberry.seeding.Seeder, int, or None
        Object for random number generation.
    _execution_metadata : ExecutionMetadata (optional)
        Extra information about agent execution (e.g. about which is the process id where the agent is running).
    _default_writer_kwargs : dict (optional)
        Parameters to initialize DefaultWriter (attribute self.writer).
    .. note::
        Classes that implement this interface should send ``**kwargs`` to :code:`Agent.__init__()`


    Attributes
    ----------
    name : string
        Agent identifier
    env : Model or tuple (constructor, kwargs)
        Environment on which to train the agent.
    eval_env : Model or tuple (constructor, kwargs)
        Environment on which to evaluate the agent. If None, copied from env.
    writer : object, default: None
        Writer object (e.g. tensorboard SummaryWriter).
    seeder : rlberry.seeding.Seeder, int, or None
        Object for random number generation.
    output_dir : str or Path
        Directory that the agent can use to store data.
    unique_id : str
        Unique identifier for the agent instance. Can be used, for example,
        to create files/directories for the agent to log data safely.
    """

    name = ""

    def __init__(self,
                 env: types.Env,
                 eval_env: Optional[types.Env] = None,
                 copy_env: bool = True,
                 seeder: Optional[types.Seed] = None,
                 output_dir: Optional[str] = None,
                 _execution_metadata: Optional[metadata_utils.ExecutionMetadata] = None,
                 _default_writer_kwargs: Optional[dict] = None,
                 **kwargs):
        # Check if wrong parameters have been sent to an agent.
        assert kwargs == {}, \
            'Unknown parameters sent to agent:' + str(kwargs.keys())

        self.seeder = Seeder(seeder)
        self.env = process_env(env, self.seeder, copy_env=copy_env)

        # evaluation environment
        eval_env = eval_env or env
        self.eval_env = process_env(eval_env, self.seeder, copy_env=True)

        # metadata
        self._execution_metadata = _execution_metadata or metadata_utils.ExecutionMetadata()
        self._unique_id = metadata_utils.get_unique_id(self)
        if self.name:
            self._unique_id = self.name + '_' + self._unique_id

        # create writer
        _default_writer_kwargs = _default_writer_kwargs or dict(
            name=self.name, execution_metadata=self._execution_metadata)
        self._writer = DefaultWriter(**_default_writer_kwargs)

        # output directory for the agent instance
        self._output_dir = output_dir or f"output_{self._unique_id}"
        self._output_dir = Path(self._output_dir)

    @property
    def writer(self):
        return self._writer

    @property
    def unique_id(self):
        return self._unique_id

    @property
    def output_dir(self):
        return self._output_dir

    @abstractmethod
    def fit(self, budget: int, **kwargs):
        """Train the agent using the provided environment.

        Ideally, calling

        .. code-block:: python

            fit(budget1)
            fit(budget2)

        should be equivalent to one call fit(budget1+budget2).
        This property is required to reduce the time required for hyperparam
        optimization (by allowing early stopping), but it is not strictly required
        elsewhere in the library.

        If the agent does not require a budget, set it to -1.

        Parameters
        ----------
        budget: int
            Computational (or sample complexity) budget.
        **kwargs
            Extra arguments.
        """
        pass

    @abstractmethod
    def eval(self, **kwargs):
        """Returns a float measuring the quality of the agent (e.g. MC policy evaluation).

        Parameters
        ----------
        eval_env: object
            Environment for evaluation.
        **kwargs: dict
            Extra parameters.
        """
        pass

    def set_writer(self, writer):
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

    @property
    def rng(self):
        """ Random number generator. """
        return self.seeder.rng

    def reseed(self, seed_seq=None):
        """
        Get new random number generator for the agent.

        Parameters
        ----------
        seed_seq : np.random.SeedSequence, rlberry.seeding.Seeder or int, default : None
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

        If overridden, load() method must also be overriden.

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
        If save() is successful, a Path object corresponding to the filename is returned.
        Otherwise, None is returned.
        Important: the returned filename might differ from the input filename: For instance,
        the method can append the correct suffix to the name before saving.

        References
        ----------
        .. [1] https://github.com/uqfoundation/dill
        """
        # remove writer if not pickleable
        if not dill.pickles(self.writer):
            self.set_writer(None)
        # save
        filename = Path(filename).with_suffix('.pickle')
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            with filename.open("wb") as ff:
                pickle.dump(self.__dict__, ff)
        except Exception:
            try:
                with filename.open("wb") as ff:
                    dill.dump(self.__dict__, ff)
            except Exception as ex:
                logger.warning("Agent instance cannot be pickled: " + str(ex))
                return None

        return filename

    @classmethod
    def load(cls, filename, **kwargs):
        """Load agent object.

        If overridden, save() method must also be overriden.

        Parameters
        ----------
        **kwargs: dict
            Arguments to required by the __init__ method of the Agent subclass.
        """
        filename = Path(filename).with_suffix('.pickle')

        obj = cls(**kwargs)
        try:
            with filename.open('rb') as ff:
                tmp_dict = pickle.load(ff)
        except Exception:
            with filename.open('rb') as ff:
                tmp_dict = dill.load(ff)

        obj.__dict__.clear()
        obj.__dict__.update(tmp_dict)
        return obj


class AgentWithSimplePolicy(Agent):
    """A subclass of Agent implementing a policy() method, and a simple evaluation
    method (Monte-Carlo policy evaluation).

    The policy() method takes an observation as input and returns an action.
    """

    @abstractmethod
    def policy(self, observation):
        """Returns an action, given an observation."""
        pass

    def eval(self,
             eval_horizon=10 ** 5,
             n_simimulations=10,
             gamma=1.0,
             **kwargs):
        """
        Monte-Carlo policy evaluation [1]_ of an agent to estimate the value at the initial state.

        Parameters
        ----------
        eval_horizon : int, default: 10**5
            Horizon, maximum episode length.
        n_simimulations : int, default: 10
            Number of Monte Carlo simulations.
        gamma : double, default: 1.0
            Discount factor.

        Return
        ------
        Mean over the n simulations of the sum of rewards in each simulation.

        References
        ----------
        .. [1] http://incompleteideas.net/book/first/ebook/node50.html
            """
        del kwargs  # unused
        episode_rewards = np.zeros(n_simimulations)
        for sim in range(n_simimulations):
            observation = self.eval_env.reset()
            tt = 0
            while tt < eval_horizon:
                action = self.policy(observation)
                observation, reward, done, _ = self.eval_env.step(action)
                episode_rewards[sim] += reward * np.power(gamma, tt)
                tt += 1
                if done:
                    break
        return episode_rewards.mean()
