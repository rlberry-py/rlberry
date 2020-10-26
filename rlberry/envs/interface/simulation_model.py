from rlberry.envs.interface.generative_model import GenerativeModel
from rlberry.envs.interface.online_model import OnlineModel


class SimulationModel(OnlineModel, GenerativeModel):
    """
    Base class for simulation models.

    A simulation model is, at the same time, an online model and a generative
    model:
    it allows us to sample trajectories *and* transitions from any state-action
    pair.

    Note: a call to sample() show not change the internal state of the model!

    Attributes
    ----------
    id : string
        environment identifier
    observation_space : rlberry.spaces.Space
        observation space
    action_space : rlberry.spaces.Space
        action space
    reward_range : tuple
        tuple (r_min, r_max) containing the minimum and the maximum reward
    rng : numpy.random._generator.Generator
        random number generator provided by rlberry.seeding

    Methods
    --------
    reset()
        puts the environment in a default state and returns this state
    step(action)
        returns the outcome of an action
    sample(state, action)
        returns a transition sampled from taking an action in a given state
    """

    def __init__(self):
        OnlineModel.__init__(self)
        GenerativeModel.__init__(self)
