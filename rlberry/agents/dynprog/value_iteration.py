from rlberry.agents.dynprog.utils import backward_induction, value_iteration
from rlberry.agents.agent import Agent
from rlberry.envs.finite.finite_mdp import FiniteMDP

class ValueIterationAgent(Agent):
    """
    Value iteration for enviroments of type FiniteMDP (rlberry.envs.finite.finite_mdp.FiniteMDP)

    Important: the discount gamma is also used if the problem is finite horizon, but, in this case,
    gamma can be set to 1.0.
    """
    def __init__(self, env, gamma=0.95, horizon=None):
        """
        Parameters
        -----------
        env : rlberry.envs.finite.finite_mdp.FiniteMDP
        gamma : double 
            discount factor in [0, 1]
        horizon : int
            horizon, if the problem is finite-horizon. if None, the discounted problem is solved
            default = None
        """
        # initialize base class
        assert isinstance(env, FiniteMDP), "Value iteration requires a FiniteMDP model."
        Agent.__init__(self, env)
        self.id = "ValueIterationAgent"

        #
        self.gamma   = gamma 
        self.horizon = horizon

        # value functions
        self.Q = None 
        self.V = None 


    def fit(self, epsilon=1e-6, **kwargs):
        """
        Parameters
        -----------
        epsilon : double
            precision of value iteration, only used in discounted problems (when horizon is None).
        """
        if self.horizon is None:
            assert self.gamma < 1.0, "The discounted setting requires gamma < 1.0"
            self.Q, self.V = value_iteration(self.env.R, self.env.P, self.gamma, epsilon)
        else:
            self.Q, self.V = backward_induction(self.env.R, self.env.P, self.horizon, self.gamma)


    def policy(self, state, hh=0, **kwargs):
        """
        Parameters
        -----------
        state : int 
        hh : int
            stage when action is taken (for finite horizon problems, the optimal policy depends on hh)
            not used if horizon is None.
        """
        assert self.env.observation_space.contains(state)
        if self.horizon is None:
            return self.Q[state, :].argmax()
        else:
            assert hh >= 0 and hh < self.horizon
            return self.Q[hh, state, :].argmax()
