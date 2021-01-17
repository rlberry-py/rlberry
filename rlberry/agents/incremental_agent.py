from abc import abstractmethod
from rlberry.agents import Agent


class IncrementalAgent(Agent):
    """Basic interface for agents that can be trained incrementally."""

    name = ""

    def __init__(self, env, **kwargs):
        """

        Parameters
        ----------
        env : Model
            Environment used to fit the agent.
        """
        Agent.__init__(self, env, **kwargs)

    def fit(self, **kwargs):
        return self.partial_fit(1.0, **kwargs)

    @abstractmethod
    def partial_fit(self, fraction, **kwargs):
        """Partially fits the agent, according to the fraction parameter.

        For instance, if the agent requires N episodes for a "full" fit,
        calling partial_fit(0.5) will fit the agent for 0.5*N episodes.

        Also, calling partial_fit(0.5) twice must be equivalent to
        a single call to fit().

        Parameters
        ---------
        fraction: double, in [0,1]
            Fraction of the agent to fit.

        Returns
        -------
        info : dict
        """
        raise NotImplementedError("agent.partial_fit() not implemented.")
