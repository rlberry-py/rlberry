from abc import ABC, abstractmethod
from copy import deepcopy
import dill
import pickle
import logging
import numpy as np
from inspect import signature
from pathlib import Path
from rlberry.seeding.seeder import Seeder


logger = logging.getLogger(__name__)


class Agent(ABC):
    """ Basic interface for agents.

    Parameters
    ----------
    env : Model
        Environment used to fit the agent.
    copy_env : bool
        If true, makes a deep copy of the environment.

    .. note::
        Classes that implement this interface should send ``**kwargs`` to :code:`Agent.__init__()`


    Attributes
    ----------
    name : string
        Agent identifier
    env : Model
        Environment on which to train the agent.
    writer : object, default: None
        Writer object (e.g. tensorboard SummaryWriter).
    seeder : rlberry.seeding.Seeder, int, or None
        Object for random number generation.
    """

    name = ""

    def __init__(self,
                 env,
                 copy_env=True,
                 seeder=None,
                 **kwargs):
        # Check if wrong parameters have been sent to an agent.
        assert kwargs == {}, \
            'Unknown parameters sent to agent:' + str(kwargs.keys())

        self.env = env

        if copy_env:
            try:
                self.env = deepcopy(env)
            except Exception as ex:
                logger.warning("[Agent] Not possible to deepcopy env: " + str(ex))

        self.writer = None

        self.seeder = Seeder(seeder)

    @abstractmethod
    def fit(self, budget: int):
        """Train the agent using the provided environment.

        Ideally, calling

        .. code-block:: python

            fit(budget1)
            fit(budget2)

        should be equivalent to one call fit(budget1+budget2).
        This property is required to reduce the time required for hyperparam
        optimization (by allowing early stopping), but it is not strictly required
        elsewhere in the library.

        Parameters
        ----------
        budget: int
            Computational (or sample complexity) budget.
        """
        pass

    @abstractmethod
    def eval(self, eval_env, **kwargs):
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
        self.writer = writer

        if self.writer:
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

    def save(self, filename):
        """
        Save agent object. By default, the agent is pickled.

        If overridden, load() method must also be overriden.

        Before saving, writer is set to None (tensorboard writers
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

        References
        ----------
        .. [1] https://github.com/uqfoundation/dill
        """
        # disable writer
        self.set_writer(None)

        # save
        filename = Path(filename).with_suffix('.pickle')
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            with filename.open("wb") as ff:
                pickle.dump(self.__dict__, ff)
            logger.info("Saved Agent instance using pickle. ({})".format(filename))
        except Exception:
            try:
                with filename.open("wb") as ff:
                    dill.dump(self.__dict__, ff)
                logger.info("Saved Agent instance using dill. ({})".format(filename))
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
            logger.info("Loaded AgentStats using pickle.")
        except Exception:
            with filename.open('rb') as ff:
                tmp_dict = dill.load(ff)
            logger.info("Loaded AgentStats using dill.")

        obj.__dict__.clear()
        obj.__dict__.update(tmp_dict)
        return obj


class AgentWithSimplePolicy(Agent):
    """A subclass of Agent implementing a policy() method, and a simple evaluation
    method (Monte-Carlo policy evaluation).

    The policy() method takes an observation as input and returns an action.
    """
    @abstractmethod
    def policy(self, observation, **kwargs):
        """Returns an action, given an observation."""
        pass

    def eval(self,
             eval_env,
             eval_horizon=10**5,
             n_simimulations=10,
             gamma=1.0,
             **kwargs):
        """
        Monte-Carlo policy evaluation [1]_ of an agent to estimate the value at the initial state.

        Parameters
        ----------
        eval_env : Env
            Evaluation environment.
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
            observation = eval_env.reset()
            for hh in range(eval_horizon):
                action = self.policy(observation)
                observation, reward, done, _ = eval_env.step(action)
                episode_rewards[sim] += reward * np.power(gamma, hh)
                if done:
                    break
        return episode_rewards.mean()
