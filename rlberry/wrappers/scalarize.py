from rlberry.envs import Wrapper
from numpy import ndarray


class ScalarizeEnvWrapper(Wrapper):
    """
    Wrapper for stable_baselines VecEnv, so that they accept non-vectorized actions,
    and return non-vectorized states.
    """

    def __init__(self, env):
        Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        scalarized_info = self._scalarize_info(infos)
        return obs[0], scalarized_info

    def step(self, action):
        if type(action) is ndarray:
            observation, reward, done, truncated, infos = self.env.step(action)
        else:
            observation, reward, done, truncated, infos = self.env.step(
                [action] * self.env.env.num_envs
            )

        scalarized_info = self._scalarize_info(infos)
        return observation[0], reward[0], done[0], truncated[0], scalarized_info

    def _scalarize_info(self, infos):
        """
        Check the format of info (VecEnv or Gymnasium), then return the info for the first env (scalarize it).

        Args:
            infos (list[Dict] or dict{list}): infos coming from the envs.
        Returns:
            dict_info (dict): scalarized info.
        -----------------------------------------------------

        What are the format with VecEnv or Gymnasium ? :

        VecEnv : [info_dict_env1{keyA,keyB},info_dict_env2{keyA,keyC},info_dict_env3{keyB,keyC},...]
        Gymnasium : {
                        KeyA:[value_env1,value_env2,None,...]
                        KeyB:[value_env1,None,value_env3,...]
                        KeyC:[None,value_env2,value_env3,...]
                    }

        other informations about it, here:
        https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
        https://gymnasium.farama.org/gym_release_notes/#release-0-24-0 (VectorListInfo)
        """
        if type(infos) in [list, ndarray]:  # StableBaseline/VecEnv/old gym Format
            scalarized_info = infos[0]
        elif type(infos) == dict:  # Gymnastium Format
            scalarized_info_dict = {}
            for key, values in infos.items():
                scalarized_info_dict[key] = values[0]
            scalarized_info = scalarized_info_dict
        else:
            raise ValueError(
                "In ScalariseEnvWrapper, the 'info' element should be an array or a dict"
            )
        return scalarized_info
