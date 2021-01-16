from abc import ABC, abstractmethod
from copy import deepcopy


class Agent(ABC):
    """ Basic interface for agents.

    Parameters
    ----------
    env : Model
        Environment used to fit the agent.
    copy_env : bool
        If true, makes a deep copy of the environment.
    reseed_env : bool
        If true, reseeds the environment.

    .. note:: Classes that implement this interface should send **kwargs to :code:`Agent.__init__()`

    Attributes
    ----------
    name : string
        Agent identifier
    fit_info : tuple
        Tuple of strings containing the keys in the dictionary
        returned by :code:`fit()`
    env : Model
        Environment on which to train the agent.
    writer : object, default: None
        Writer object (e.g. tensorboard SummaryWriter).
    """

    name = ""
    fit_info = ()

    def __init__(self,
                 env,
                 copy_env=True,
                 reseed_env=True,
                 **kwargs):
        # Check if wrong parameters have been sent to an agent.
        assert kwargs == {}, \
            'Unknown parameters sent to agent:' + str(kwargs.keys())

        if copy_env:
            self.env = deepcopy(env)
        else:
            self.env = env
        if reseed_env:
            self.env.reseed()

        #
        self.writer = None

    @abstractmethod
    def fit(self, **kwargs):
        """Train the agent using the provided environment.

        Returns
        -------
        info: dict
            Dictionary whose keys are in the tuple :attr:`Agent.fit_info`
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
