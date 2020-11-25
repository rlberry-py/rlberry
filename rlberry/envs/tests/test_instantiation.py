import numpy as np
import pytest

from rlberry.envs.classic_control import MountainCar, Acrobot
from rlberry.envs.finite import Chain
from rlberry.envs.finite import GridWorld
from rlberry.envs.benchmarks.ball_exploration import PBall2D, SimplePBallND
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.rendering.render_interface import RenderInterface2D


classes = [
    MountainCar,
    GridWorld,
    Chain,
    PBall2D,
    SimplePBallND,
    Acrobot
]


@pytest.mark.parametrize("ModelClass", classes)
def test_instantiation(ModelClass):
    env = ModelClass()

    if env.is_online():
        for _ in range(2):
            state = env.reset()
            for _ in range(50):
                assert env.observation_space.contains(state)
                action = env.action_space.sample()
                next_s, _, _, _ = env.step(action)
                state = next_s

    if env.is_generative():
        for _ in range(100):
            state = env.observation_space.sample()
            action = env.action_space.sample()
            next_s, _, _, _ = env.sample(state, action)
            assert env.observation_space.contains(next_s)


@pytest.mark.parametrize("ModelClass", classes)
def test_rendering_calls(ModelClass):
    env = ModelClass()
    if isinstance(env, RenderInterface2D):
        _ = env.get_background()
        _ = env.get_scene(env.observation_space.sample())


def test_gridworld_aux_functions():
    env = GridWorld(nrows=5, ncols=5, walls=((1, 1),),
                    reward_at={(4, 4): 1, (4, 3): -1})
    env.print()  # from FiniteMDP
    env.render_ascii()  # from GridWorld
    vals = np.ones(env.observation_space.n)
    env.display_values(vals)
    env.print_transition_at(0, 0, 'up')


def test_ball2d_benchmark_instantiation():
    for level in [1, 2, 3, 4, 5]:
        env = get_benchmark_env(level)
        for aa in range(env.action_space.n):
            env.step(aa)
            env.sample(env.observation_space.sample(), aa)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, np.inf])
def test_pball_env(p):
    env = PBall2D(p=p)
    env.get_reward_lipschitz_constant()
    env.get_transitions_lipschitz_constant()
