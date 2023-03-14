import numpy as np
from rlberry.envs import SpringCartPole
from rlberry.envs.classic_control.SpringCartPole import rk4


# # actions
# LL = 0
# RR = 1
# LR = 2
# RL = 3

# action_dict = {0: "LL", 1: "RR", 2: "LR", 3: "RL"}


HORIZON = 20


def test_spring_cartpole():
    # test 1 - default
    env = SpringCartPole()

    _,info = env.reset()
    for _ in range(2):
        action = np.random.randint(0, env.action_space.n)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # if done:
        #     next_observation,info = env.reset()
        # observation = next_observation

    # test 2 - obs_trans = True and random_init = False
    env = SpringCartPole(obs_trans=True, random_init=False)

    _,info = env.reset()
    for _ in range(2):
        action = np.random.randint(0, env.action_space.n)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # if done:
        #     next_observation,info = env.reset()
        # observation = next_observation

    # # test 3 - swingup = False and random_init = False
    # env = SpringCartPole(dt=0.01, swing_up=False, random_init=False)
    # # env.enable_rendering()

    # observation,info = env.reset()
    # for tt in range(5):
    #     if observation[2] > 0:
    #         if observation[6] > 0:
    #             action = LL
    #         else:
    #             action = LR
    #     else:
    #         if observation[6] > 0:
    #             action = RL
    #         else:
    #             action = RR
    #     # print("Time: ", tt, "Action: ", action_dict[action], "Angle1: ", observation[2], "Angle2: ", observation[6])
    #     next_observation, reward, terminated, truncated, info= env.step(action)
    #     done = terminated or truncated
    #     if done:
    #         next_observation,info = env.reset()
    #     observation = next_observation

    # test 4 - swingup = False and rendering = True

    env = SpringCartPole(dt=0.02, swing_up=False, obs_trans=True)
    env.enable_rendering()

    _,info = env.reset()
    action = 0
    for _ in range(2 * HORIZON):
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            action += 1
            if action >= 4:
                action = 0
            next_observation,info = env.reset()
        _ = next_observation

    _ = env.get_video()


def test_rk4():
    """
    Test of the rk4 utils defined in speingcartpole
    """

    ## 2D system
    def derivs6(x, t):
        d1 = x[0] + 2 * x[1]
        d2 = -3 * x[0] + 4 * x[1]
        return (d1, d2)

    dt = 0.0005
    t = np.arange(0.0, 2.0, dt)
    y0 = (1, 2)
    yout = rk4(derivs6, y0, t)
    assert np.abs(yout[0][0] - 1) < 1e-2
    assert np.abs(yout[0][1] - 2) < 1e-2
    assert np.abs(yout[-1][0] + 238.087) < 1e-2
    assert np.abs(yout[-1][1] + 220.827) < 1e-2


if __name__ == "__main__":
    test_spring_cartpole()
