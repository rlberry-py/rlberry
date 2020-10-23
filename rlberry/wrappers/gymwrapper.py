_GYM_INSTALLED = True
try:
    import gym 
except:
    _GYM_INSTALLED = False

import rlberry
from rlberry.envs.interface import OnlineModel


def convert_space_from_gym(gym_space):
    if isinstance(gym_space, gym.spaces.Discrete):
        n = gym_space.n
        rlberry_space = rlberry.spaces.Discrete(n)
    elif isinstance(gym_space, gym.spaces.Box):
        assert gym_space.high.ndim == 1, "Conversion from gym.spaces.Box requires high and low to be 1d."
        high = gym_space.high
        low  = gym_space.low
        rlberry_space = rlberry.spaces.Box(low, high)
    
    return rlberry_space


class GymWrapper(OnlineModel):
    """
    Wraps an OpenAI gym environment in an rlberry OnlineModel.
    """
    def __init__(self, gym_env):
        """
        Parameters
        ----------
        gym_env:  gym.Env
            environment to be wrapped
        """
        OnlineModel.__init__(self)
        assert _GYM_INSTALLED, "Error: trying to init GymWrapper without gym installed."
        assert isinstance(gym_env, gym.Env)

        # Warnings
        print("Warning (GymWrapper): seeding of gym.Env does not follow the same protocol as rlberry. Make sure to properly seed each instance before using the wrapped environment.")
        print("Warning (GymWrapper): rendering gym.Env does not follow the same protocol as rlberry.")

        # Convert spaces
        self.observation_space = convert_space_from_gym(gym_env.observation_space)
        self.action_space      = convert_space_from_gym(gym_env.action_space)

        # Reward range
        self.reward_range      = gym_env.reward_range

        # wrapped env
        self.gym_env = gym_env

    def reset(self):
        return self.gym_env.reset()

    def step(self, action):
        return self.gym_env.step(action)

    def render(self, **kwargs):
        return self.gym_env.render(**kwargs)
    
    def close(self):
        return self.gym_env.close()
    
    def seed(self, seed=None):
        return self.gym_env.seed(seed)

    @property
    def unwrapped(self):
        return self.gym_env