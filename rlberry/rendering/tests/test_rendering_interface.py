import os
import pytest
import sys

from pyvirtualdisplay import Display
from rlberry.envs.classic_control import MountainCar
from rlberry.envs.classic_control import Acrobot
from rlberry.envs.classic_control import Pendulum
from rlberry.envs.finite import Chain
from rlberry.envs.finite import GridWorld
from rlberry.envs.benchmarks.grid_exploration.four_room import FourRoom
from rlberry.envs.benchmarks.grid_exploration.six_room import SixRoom
from rlberry.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry.envs.benchmarks.ball_exploration import PBall2D, SimplePBallND
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry.rendering import RenderInterface
from rlberry.rendering import RenderInterface2D
from rlberry.envs import Wrapper

import tempfile

try:
    display = Display(visible=0, size=(1400, 900))
    display.start()
except Exception:
    pass

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


@pytest.mark.xfail(sys.platform != "linux", reason="bug with mac and windows???")
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


@pytest.mark.xfail(sys.platform != "linux", reason="bug with mac and windows???")
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
