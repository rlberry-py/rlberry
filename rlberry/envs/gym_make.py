import gymnasium as gym

from rlberry.envs.basewrapper import Wrapper
import numpy as np
from typing import List


def gym_make(id, wrap_spaces=False, **kwargs):
    """
    Same as gym.make, but wraps the environment
    to ensure unified seeding with rlberry.

    Parameters
    ----------
    id : str
        Environment id.
    wrap_spaces : bool, default = False
        If true, also wraps observation_space and action_space using classes in rlberry.spaces,
        that define a reseed() method.
    **kwargs
        Optional arguments to configure the environment.

    Examples
    --------
    >>> from rlberry.envs import gym_make
    >>> env_ctor = gym_make
    >>> env_kwargs = {"id": "CartPole-v0"}
    >>> env = env_ctor(**env_kwargs)
    """
    if "module_import" in kwargs:
        __import__(kwargs.pop("module_import"))

    env = gym.make(id, **kwargs)

    return Wrapper(env, wrap_spaces=wrap_spaces)


def atari_make(id, **kwargs):
    """
    Adaptator to work with 'make_atari_env' in stableBaselines.
    Use 'ScalarizeEnvWrapper' to ignore the vectorizedEnv from StableBaseline


    Parameters
    ----------
    id : str
        Environment id.
    **kwargs
        Optional arguments to configure the environment.
        (render_mode, n_frame_stack, and other arguments for StableBaselines's make_atari_env : https://stable-baselines3.readthedocs.io/en/master/common/env_util.html#stable_baselines3.common.env_util.make_atari_env )
    Returns
    -------
    Atari env with wrapper to be used as Gymnasium env.
    Examples
    --------
    >>> from rlberry.envs.gym_make import atari_make
    >>> env_ctor = atari_make
    >>> env_kwargs = {"id": "ALE/Freeway-v5", "atari_wrappers_dict":dict(terminal_on_life_loss=False),"n_frame_stack":5}}
    >>> env = env_ctor(**env_kwargs)
    """

    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack
    from rlberry.wrappers.scalarize import ScalarizeEnvWrapper

    if "atari_wrappers_dict" in kwargs.keys():
        atari_wrappers_dict = kwargs.pop("atari_wrappers_dict")
    else:
        atari_wrappers_dict = dict(
            terminal_on_life_loss=False
        )  # hack, some errors with the "terminal_on_life_loss" wrapper : The 'false reset' can lead to make a step on a 'done' environment, then a crash.

    render_mode = None
    if "render_mode" in kwargs.keys():
        render_mode = kwargs["render_mode"]
        kwargs.pop("render_mode", None)

    if "n_frame_stack" in kwargs.keys():
        n_frame_stack = kwargs.pop("n_frame_stack")
    else :
        n_frame_stack = 4


    env = make_atari_env(env_id=id, wrapper_kwargs=atari_wrappers_dict, **kwargs)

    env = VecFrameStack(
        env, n_stack=n_frame_stack
    )  # Stack previous images to have an "idea of the motion"
    env = SB3_Atari_Wrapper(
        env
    )  # Convert from SB3 API to gymnasium API, and to PyTorch format.
    env = ScalarizeEnvWrapper(env)  # wrap the vectorized env into a single env.

    env.render_mode = render_mode

    return env


class SB3_Atari_Wrapper(Wrapper):
    """
    Convert from SB3 API to gymnasium API, and to PyTorch format.

    _observation :
    transform the observations shape.
    from: n_env, height, width, chan
    to: n_env, chan, width, height

    _convert_info_list_to_dict :
    transform the info format from "list of dict" to "dict of list"

    WARNING : Check the Reset and Step format :
    https://github.com/DLR-RM/stable-baselines3/pull/1327/files#diff-a0b0c17357564df74e097f3094a5478e9b28b2af9dfdab2a91e60b6dbe174092
    https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api

    """

    def __init__(self, env):
        super(SB3_Atari_Wrapper, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def _observation(self, observation):
        return np.transpose(observation, (0, 3, 2, 1))  # transform

    def reset(self, seed=None, options=None):
        if seed:
            self.env.seed(seed=seed)
        obs = self.env.reset()
        infos = self.env.venv.reset_infos
        infos = self._convert_info_list_to_dict(infos)

        return self._observation(obs), infos

    def step(self, actions):
        next_observations, rewards, done, infos = self.env.step(actions)
        infos = self._convert_info_list_to_dict(infos)
        return (
            self._observation(next_observations),
            rewards,
            done,
            infos["TimeLimit.truncated"],
            infos,
        )

    def _convert_info_list_to_dict(self, infos: List[dict]) -> dict:
        """
        Convert the list info of the vectorized environment into a dict of list where each key has a list of the specific info for each env

        Args:
            infos (list): info list coming from the envs.
        Returns:
            dict_info (dict): converted info.

        ----------------------------
        This is the opposit of the VectorListInfo wrapper:
        https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.VectorListInfo

        because StableBaselines and Gymnasium don't use the same 'info' API:
        https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
        """
        all_keys = set().union(
            *[dictIni.keys() for dictIni in infos]
        )  # Get all unique keys for all the dict in the list

        dict_info = {}
        for key in all_keys:
            values = [
                dictIni.get(key) for dictIni in infos
            ]  # Get the values of the key for each dictionary
            dict_info[key] = values

        return dict_info
