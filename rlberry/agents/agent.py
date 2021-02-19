from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from inspect import signature
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
    seeder : rlberry.seeding.Seeder
        Object for random number generation.
    """

    name = ""

    def __init__(self,
                 env,
                 copy_env=True,
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

        self.seeder = Seeder()

    @abstractmethod
    def fit(self, **kwargs):
        """Train the agent using the provided environment.

        Returns
        -------
        info: dict
            Dictionary with useful info.
        """
        pass

    @abstractmethod
    def policy(self, observation, **kwargs):
        """Returns an action, given an observation."""
        pass

    def reset(self, **kwargs):
        """Put the agent in default setup."""
        pass

    def save(self, filename, **kwargs):
        """Save agent object."""
        raise NotImplementedError("agent.save() not implemented.")

    def load(self, filename, **kwargs):
        """Load agent object."""
        raise NotImplementedError("agent.load() not implemented.")

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
        Get new random number generator for the model.

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
