import os
import pytest
import sys

from rlberry_research.envs.classic_control import MountainCar
from rlberry_research.envs.classic_control import Acrobot
from rlberry_research.envs.classic_control import Pendulum
from rlberry_scool.envs.finite import Chain
from rlberry_scool.envs.finite import GridWorld
from rlberry_scool.agents.dynprog import ValueIterationAgent
from rlberry_research.agents import RSUCBVIAgent
from rlberry_research.envs.benchmarks.grid_exploration.four_room import FourRoom
from rlberry_research.envs.benchmarks.grid_exploration.six_room import SixRoom
from rlberry_research.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry_research.envs.benchmarks.ball_exploration import PBall2D, SimplePBallND
from rlberry_research.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry.rendering import RenderInterface
from rlberry.rendering import RenderInterface2D
from rlberry.envs import Wrapper
from rlberry.envs import gym_make
from rlberry.seeding import Seeder

import tempfile


classes = [
    Acrobot,
    Pendulum,
    MountainCar,
    GridWorld,
    Chain,
    PBall2D,
    SimplePBallND,
    FourRoom,
    SixRoom,
    AppleGold,
    TwinRooms,
]


@pytest.mark.parametrize("ModelClass", classes)
def test_instantiation(ModelClass):
    env = ModelClass()

    if isinstance(env, RenderInterface):
        env.disable_rendering()
        assert not env.is_render_enabled()
        env.enable_rendering()
        assert env.is_render_enabled()


@pytest.mark.skipif(sys.platform != "linux", reason="bug with mac and windows???")
@pytest.mark.parametrize("ModelClass", classes)
def test_render2d_interface(ModelClass):
    env = ModelClass()

    if isinstance(env, RenderInterface2D):
        env.enable_rendering()

        if env.is_online():
            for _ in range(2):
                observation, info = env.reset()
                for _ in range(5):
                    assert env.observation_space.contains(observation)
                    action = env.action_space.sample()
                    observation, _, _, _, _ = env.step(action)
                env.render(loop=False)

            with tempfile.TemporaryDirectory() as tmpdirname:
                saving_path = tmpdirname + "/test_video.mp4"

                env.save_video(saving_path)
                env.clear_render_buffer()


@pytest.mark.skipif(sys.platform != "linux", reason="bug with mac and windows???")
@pytest.mark.parametrize("ModelClass", classes)
def test_render2d_interface_wrapped(ModelClass):
    env = Wrapper(ModelClass())

    if isinstance(env.env, RenderInterface2D):
        env.enable_rendering()
        if env.is_online():
            for _ in range(2):
                observation, info = env.reset()
                for _ in range(5):
                    assert env.observation_space.contains(observation)
                    action = env.action_space.sample()
                    observation, _, _, _, _ = env.step(action)
                env.render(loop=False)

            with tempfile.TemporaryDirectory() as tmpdirname:
                saving_path = tmpdirname + "/test_video.mp4"
                env.save_video(saving_path)
                env.clear_render_buffer()
        try:
            os.remove("test_video.mp4")
        except Exception:
            pass


def test_render_appelGold():
    env = AppleGold()
    env.render_mode = "human"
    env = Wrapper(env)

    if env.is_online():
        for _ in range(2):
            observation, info = env.reset()
            for _ in range(5):
                assert env.observation_space.contains(observation)
                action = env.action_space.sample()
                observation, _, _, _, _ = env.step(action)
            env.render(loop=False)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saving_path = tmpdirname + "/test_video.mp4"
            env.save_video(saving_path)
            env.clear_render_buffer()
    try:
        os.remove("test_video.mp4")
    except Exception:
        pass


def test_write_gif():
    env = Chain(10, 0.3)
    env.enable_rendering()
    for tt in range(20):
        env.step(env.action_space.sample())
    with tempfile.TemporaryDirectory() as tmpdirname:
        saving_path = tmpdirname + "/test_gif.mp4"
        env.save_gif(saving_path)
        assert os.path.isfile(saving_path)
        try:
            os.remove(saving_path)
        except Exception:
            pass


RENDERING_TOOL = ["pygame", "opengl"]


@pytest.mark.skipif(sys.platform == "darwin", reason="bug with Mac with pygame")
@pytest.mark.parametrize("rendering_tool", RENDERING_TOOL)
def test_gridworld_rendering_gif(rendering_tool):
    env = GridWorld(7, 10, walls=((2, 2), (3, 3)))
    env.renderer_type = rendering_tool

    agent = ValueIterationAgent(env, gamma=0.95)
    info = agent.fit()
    print(info)

    env.enable_rendering()
    observation, info = env.reset()
    for tt in range(50):
        action = agent.policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            # Warning: this will never happen in the present case because there is no terminal state.
            # See the doc of GridWorld for more information on the default parameters of GridWorld.
            break

    with tempfile.TemporaryDirectory() as tmpdirname:
        saving_path = tmpdirname + "/test_gif.gif"
        env.save_gif(saving_path)
        assert os.path.isfile(saving_path)
        try:
            os.remove(saving_path)
        except Exception:
            pass


# ##### Works only if you add a dependency to ffmpeg ############
# @pytest.mark.xfail(sys.platform == "darwin", reason="bug with Mac with pygame")
# @pytest.mark.parametrize("rendering_tool", RENDERING_TOOL)
# def test_gridworld_rendering_mp4(rendering_tool):
#     env = GridWorld(7, 10, walls=((2, 2), (3, 3)))
#     env.renderer_type = rendering_tool

#     agent = ValueIterationAgent(env, gamma=0.95)
#     info = agent.fit()
#     print(info)

#     env.enable_rendering()
#     observation, info = env.reset()
#     for tt in range(50):
#         action = agent.policy(observation)
#         observation, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         if done:
#             # Warning: this will never happen in the present case because there is no terminal state.
#             # See the doc of GridWorld for more information on the default parameters of GridWorld.
#             break

#     with tempfile.TemporaryDirectory() as tmpdirname:
#         saving_path = tmpdirname + "/test_video.mp4"
#         env.save_video(saving_path)
#         assert os.path.isfile(saving_path)
#         try:
#             os.remove(saving_path)
#         except Exception:
#             pass


@pytest.mark.skipif(sys.platform == "darwin", reason="bug with Mac with pygame")
@pytest.mark.parametrize("rendering_tool", RENDERING_TOOL)
def test_gridworld_rendering_screen(rendering_tool):
    env = GridWorld(7, 10, walls=((2, 2), (3, 3)))
    env.renderer_type = rendering_tool

    agent = ValueIterationAgent(env, gamma=0.95)
    info = agent.fit()
    print(info)

    env.enable_rendering()
    observation, info = env.reset()
    for tt in range(50):
        action = agent.policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            # Warning: this will never happen in the present case because there is no terminal state.
            # See the doc of GridWorld for more information on the default parameters of GridWorld.
            break

    env.render(loop=False)


@pytest.mark.skipif(sys.platform == "darwin", reason="bug with Mac with pygame")
@pytest.mark.parametrize("rendering_tool", RENDERING_TOOL)
def test_gym_make_rendering_gif(rendering_tool):
    seeder = Seeder(123)
    env = gym_make("MountainCar-v0", render_mode="rgb_array")
    env.reseed(seeder)
    env.renderer_type = rendering_tool

    agent = RSUCBVIAgent(
        env,
        gamma=0.99,
        horizon=200,
        bonus_scale_factor=0.1,
        copy_env=False,
        min_dist=0.1,
    )

    info = agent.fit(15)
    print(info)

    env.enable_rendering()
    observation, info = env.reset()
    for tt in range(100):
        action = agent.policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            # Warning: this will never happen in the present case.
            break

    with tempfile.TemporaryDirectory() as tmpdirname:
        saving_path = tmpdirname + "/test_gif.gif"
        env.save_gif(saving_path)
        assert os.path.isfile(saving_path)

        try:
            os.remove(saving_path)
        except Exception:
            pass


# ##### Works only if you add a dependency to ffmpeg ############
# @pytest.mark.skipif(sys.platform == "darwin", reason="bug with Mac with pygame")
# @pytest.mark.parametrize("rendering_tool", RENDERING_TOOL)
# def test_gym_make_rendering_mp4(rendering_tool):
#     seeder = Seeder(123)
#     env = gym_make("MountainCar-v0", render_mode="rgb_array")
#     env.reseed(seeder)
#     env.renderer_type = rendering_tool

#     agent = RSUCBVIAgent(
#         env,
#         gamma=0.99,
#         horizon=200,
#         bonus_scale_factor=0.1,
#         copy_env=False,
#         min_dist=0.1,
#     )

#     info = agent.fit(15)
#     print(info)

#     env.enable_rendering()
#     observation, info = env.reset()
#     for tt in range(100):
#         action = agent.policy(observation)
#         observation, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         if done:
#             # Warning: this will never happen in the present case.
#             break

#     with tempfile.TemporaryDirectory() as tmpdirname:
#         saving_path = tmpdirname + "/test_video.mp4"
#         env.save_video(saving_path)
#         assert os.path.isfile(saving_path)
#         try:
#             os.remove(saving_path)
#         except Exception:
#             pass
