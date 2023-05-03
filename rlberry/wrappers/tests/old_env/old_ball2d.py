"""
This files provides a set of 2D environments with increasing difficulty
of exploration.

The difficulty is ranked by the level.

Important:
    * To create instances, use the function get_benchmark_env(level).
    * The horizon H is also set as an attribute of the environment.
"""

import numpy as np

from rlberry.wrappers.autoreset import AutoResetWrapper
from rlberry.wrappers.tests.old_env.old_pball import Old_PBall2D


def get_benchmark_env(level=1):
    if level == 0:
        env = _get_autoreset_env(Old_BallLevel0())
        return env
    elif level == 1:
        env = _get_autoreset_env(Old_BallLevel1())
        return env
    elif level == 2:
        env = _get_autoreset_env(Old_BallLevel2())
        return env
    elif level == 3:
        env = _get_autoreset_env(Old_BallLevel3())
        return env
    elif level == 4:
        env = _get_autoreset_env(Old_BallLevel4())
        return env
    elif level == 5:
        env = _get_autoreset_env(Old_BallLevel5())
        return env
    else:
        raise NotImplementedError("Invalid benchmark level.")


def _get_autoreset_env(env):
    horizon = env.horizon
    return AutoResetWrapper(env, horizon)


#
# Level 0 (reward free!)
#
class Old_BallLevel0(Old_PBall2D):
    """
    'Old' because it's a pre-gymnasium version (like gym=0.21)
        - step with 'done' instead of 'terminated' and 'truncated'
        - reset with 'observation' only, instead of 'observation' and 'info'

    Reward-free (0 reward)
    """

    def __init__(self):
        self.horizon = 30
        #
        self.p = 2
        self.action_list = [
            np.array([0.0, 0.0]),
            0.05 * np.array([1.0, 0.0]),
            -0.05 * np.array([1.0, 0.0]),
            0.05 * np.array([0.0, 1.0]),
            -0.05 * np.array([0.0, 1.0]),
        ]

        self.reward_amplitudes = []
        self.reward_smoothness = []
        self.reward_centers = []
        self.A = np.eye(2)
        self.B = np.eye(2)
        self.sigma = 0.01
        self.sigma_init = 0.001
        self.mu_init = np.array([0.0, 0.0])

        Old_PBall2D.__init__(
            self,
            self.p,
            self.action_list,
            self.reward_amplitudes,
            self.reward_smoothness,
            self.reward_centers,
            self.A,
            self.B,
            self.sigma,
            self.sigma_init,
            self.mu_init,
        )
        self.name = "Ball Exploration Benchmark - Level 0 (Reward-Free)"


#
# Level 1
#


class Old_BallLevel1(Old_PBall2D):
    """
    'Old' because it's a pre-gymnasium version (like gym=0.21)
        - step with 'done' instead of 'terminated' and 'truncated'
        - reset with 'observation' only, instead of 'observation' and 'info'

    Dense rewards
    """

    def __init__(self):
        self.horizon = 30
        #
        self.p = 2
        self.action_list = [
            np.array([0.0, 0.0]),
            0.05 * np.array([1.0, 0.0]),
            -0.05 * np.array([1.0, 0.0]),
            0.05 * np.array([0.0, 1.0]),
            -0.05 * np.array([0.0, 1.0]),
        ]

        self.reward_amplitudes = np.array([1.0])
        self.reward_smoothness = np.array([0.5 * np.sqrt(2)])
        self.reward_centers = [np.array([0.5, 0.5])]
        self.A = np.eye(2)
        self.B = np.eye(2)
        self.sigma = 0.01
        self.sigma_init = 0.001
        self.mu_init = np.array([0.0, 0.0])

        Old_PBall2D.__init__(
            self,
            self.p,
            self.action_list,
            self.reward_amplitudes,
            self.reward_smoothness,
            self.reward_centers,
            self.A,
            self.B,
            self.sigma,
            self.sigma_init,
            self.mu_init,
        )
        self.name = "Ball Exploration Benchmark - Level 1"


#
# Level 2
#


class Old_BallLevel2(Old_BallLevel1):
    """
    'Old' because it's a pre-gymnasium version (like gym=0.21)
        - step with 'done' instead of 'terminated' and 'truncated'
        - reset with 'observation' only, instead of 'observation' and 'info'

    Sparse rewards
    """

    def __init__(self):
        Old_BallLevel1.__init__(self)
        self.reward_amplitudes = np.array([1.0])
        self.reward_smoothness = np.array([0.2])
        self.reward_centers = [np.array([0.5, 0.5])]
        self.name = "Ball Exploration Benchmark - Level 2"


#
# Level 3
#


class Old_BallLevel3(Old_BallLevel2):
    """
    'Old' because it's a pre-gymnasium version (like gym=0.21)
        - step with 'done' instead of 'terminated' and 'truncated'
        - reset with 'observation' only, instead of 'observation' and 'info'

    Sparse rewards, noisier
    """

    def __init__(self):
        Old_BallLevel2.__init__(self)
        self.sigma = 0.025
        self.name = "Ball Exploration Benchmark - Level 3"


#
# Level 4
#


class Old_BallLevel4(Old_BallLevel1):
    """
    'Old' because it's a pre-gymnasium version (like gym=0.21)
        - step with 'done' instead of 'terminated' and 'truncated'
        - reset with 'observation' only, instead of 'observation' and 'info'

    Far sparse reward (as lvl 2) + dense suboptimal rewards
    """

    def __init__(self):
        Old_BallLevel1.__init__(self)

        self.reward_amplitudes = np.array([1.0, 0.1])
        self.reward_smoothness = np.array([0.2, 0.5 * np.sqrt(2)])
        self.reward_centers = [
            np.array([-0.5, -0.5]),  # far sparse
            np.array([0.5, 0.5]),
        ]  # dense
        self.name = "Ball Exploration Benchmark - Level 4"


#
# Level 5
#


class Old_BallLevel5(Old_BallLevel4):
    """
    'Old' because it's a pre-gymnasium version (like gym=0.21)
        - step with 'done' instead of 'terminated' and 'truncated'
        - reset with 'observation' only, instead of 'observation' and 'info'

    Far sparse reward (as lvl 2) + dense suboptimal rewards, noisier
    """

    def __init__(self):
        Old_BallLevel4.__init__(self)
        self.sigma = 0.025
        self.name = "Ball Exploration Benchmark - Level 5"


# if __name__ == '__main__':
#     env = get_benchmark_env(1)
#     env.enable_rendering()
#     for ii in range(100):
#         # env.step(1)
#         # env.step(3)
#         # env.step(env.action_space.sample())
#         # env.step(0)
#         env.step(4)

#     env.render()
