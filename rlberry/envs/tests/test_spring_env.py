import numpy as np
from rlberry.envs import SpringCartPole


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

    _ = env.reset()
    for _ in range(2):
        action = np.random.randint(0, env.action_space.n)
        next_state, _, done, _ = env.step(action)
        # if done:
        #     next_state = env.reset()
        # state = next_state

    # test 2 - obs_trans = True and random_init = False
    env = SpringCartPole(obs_trans=True, random_init=False)

    _ = env.reset()
    for _ in range(2):
        action = np.random.randint(0, env.action_space.n)
        next_state, _, done, _ = env.step(action)
        # if done:
        #     next_state = env.reset()
        # state = next_state

    # # test 3 - swingup = False and random_init = False
    # env = SpringCartPole(dt=0.01, swing_up=False, random_init=False)
    # # env.enable_rendering()

    # state = env.reset()
    # for tt in range(5):
    #     if state[2] > 0:
    #         if state[6] > 0:
    #             action = LL
    #         else:
    #             action = LR
    #     else:
    #         if state[6] > 0:
    #             action = RL
    #         else:
    #             action = RR
    #     # print("Time: ", tt, "Action: ", action_dict[action], "Angle1: ", state[2], "Angle2: ", state[6])
    #     next_state, reward, done, _ = env.step(action)
    #     if done:
    #         next_state = env.reset()
    #     state = next_state

    # test 4 - swingup = False and rendering = True

    env = SpringCartPole(dt=0.02, swing_up=False, obs_trans=True)
    env.enable_rendering()

    _ = env.reset()
    action = 0
    for _ in range(2 * HORIZON):
        next_state, _, done, _ = env.step(action)
        if done:
            action += 1
            if action >= 4:
                action = 0
            next_state = env.reset()
        _ = next_state

    _ = env.get_video()

    # env.render()

    # Save video
    # video = env.save_video("_video/video_plot_acrobot.mp4")
