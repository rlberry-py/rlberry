from rlberry.agents.agent import AgentWithSimplePolicy
from rlberry.agents.dynprog.utils import backward_induction, value_iteration
from rlberry.envs.finite.finite_mdp import FiniteMDP


class ValueIterationAgent(AgentWithSimplePolicy):
    """
    Value iteration for enviroments of type FiniteMDP
    (rlberry.envs.finite.finite_mdp.FiniteMDP)

    Important: the discount gamma is also used if the problem is
    finite horizon, but, in this case, gamma can be set to 1.0.

    Parameters
    -----------
    env : rlberry.envs.finite.finite_mdp.FiniteMDP
    gamma : double
        discount factor in [0, 1]
    horizon : int
        horizon, if the problem is finite-horizon. if None, the discounted
        problem is solved
        default = None
    epsilon : double
        precision of value iteration, only used in discounted problems
        (when horizon is None).
    """

    name = "ValueIteration"

    def __init__(self, env, gamma=0.95, horizon=None, epsilon=1e-6, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        # initialize base class
        assert isinstance(self.env, FiniteMDP), \
            "Value iteration requires a FiniteMDP model."
        #

        self.gamma = gamma
        self.horizon = horizon
        self.epsilon = epsilon

        # value functions
        self.Q = None
        self.V = None

    def fit(self, budget=None, **kwargs):
        """
        Run value iteration.
        """
        del kwargs
        info = {}
        if self.horizon is None:
            assert self.gamma < 1.0, \
                "The discounted setting requires gamma < 1.0"
            self.Q, self.V, n_it = value_iteration(self.env.R, self.env.P,
                                                   self.gamma, self.epsilon)
            info["n_iterations"] = n_it
            info["precision"] = self.epsilon
        else:
            self.Q, self.V = backward_induction(self.env.R, self.env.P,
                                                self.horizon, self.gamma)
            info["n_iterations"] = self.horizon
            info["precision"] = 0.0
        return info

    def policy(self, observation):
        state = observation
        if self.horizon is None:
            return self.Q[state, :].argmax()
        else:
            return self.Q[0, state, :].argmax()
