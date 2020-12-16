import numpy as np
import pytest

from rlberry.envs.classic_control import MountainCar, Acrobot, Pendulum
from rlberry.envs.finite import Chain
from rlberry.envs.finite import GridWorld
from rlberry.envs.benchmarks.ball_exploration import PBall2D, SimplePBallND
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.envs.benchmarks.grid_exploration.four_room import FourRoom
from rlberry.envs.benchmarks.grid_exploration.six_room import SixRoom
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry.rendering.render_interface import RenderInterface2D


classes = [
    MountainCar,
    GridWorld,
    Chain,
    PBall2D,
    SimplePBallND,
    Acrobot,
    Pendulum,
    FourRoom,
    SixRoom,
    AppleGold,
    NRoom
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
    env.log()  # from FiniteMDP
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


@pytest.mark.parametrize("reward_free, difficulty, array_observation",
                         [
                             (True, 0, False),
                             (False, 0, False),
                             (False, 0, True),
                             (False, 1, False),
                             (False, 1, True),
                             (False, 2, False),
                             (False, 2, True),
                         ])
def test_four_room(reward_free, difficulty, array_observation):
    env = FourRoom(reward_free=reward_free,
                   difficulty=difficulty,
                   array_observation=array_observation)

    initial_state = env.reset()
    next_state, reward, _, _ = env.step(1)

    assert env.observation_space.contains(initial_state)
    assert env.observation_space.contains(next_state)

    if reward_free:
        assert env.reward_at == {}

    if difficulty == 2:
        assert reward < 0.0

    if array_observation:
        assert isinstance(initial_state, np.ndarray)
        assert isinstance(next_state, np.ndarray)


@pytest.mark.parametrize("reward_free, array_observation",
                         [
                             (False, False),
                             (False, True),
                             (True, False),
                             (True, True),
                         ])
def test_six_room(reward_free, array_observation):
    env = SixRoom(reward_free=reward_free, array_observation=array_observation)

    initial_state = env.reset()
    next_state, reward, _, _ = env.step(1)

    assert env.observation_space.contains(initial_state)
    assert env.observation_space.contains(next_state)

    if reward_free:
        assert env.reward_at == {}

    if array_observation:
        assert isinstance(initial_state, np.ndarray)
        assert isinstance(next_state, np.ndarray)


@pytest.mark.parametrize("reward_free, array_observation",
                         [
                             (False, False),
                             (False, True),
                             (True, False),
                             (True, True),
                         ])
def test_apple_gold(reward_free, array_observation):
    env = AppleGold(reward_free=reward_free, array_observation=array_observation)

    initial_state = env.reset()
    next_state, reward, _, _ = env.step(1)
    assert env.observation_space.contains(initial_state)
    assert env.observation_space.contains(next_state)

    if reward_free:
        assert env.reward_at == {}

    if array_observation:
        assert isinstance(initial_state, np.ndarray)
        assert isinstance(next_state, np.ndarray)


@pytest.mark.parametrize("reward_free, array_observation",
                         [
                             (False, False),
                             (False, True),
                             (True, False),
                             (True, True),
                         ])
def test_n_room(reward_free, array_observation):
    env = NRoom(reward_free=reward_free, array_observation=array_observation)

    initial_state = env.reset()
    next_state, reward, _, _ = env.step(1)

    assert env.observation_space.contains(initial_state)
    assert env.observation_space.contains(next_state)

    if reward_free:
        assert env.reward_at == {}

    if array_observation:
        assert isinstance(initial_state, np.ndarray)
        assert isinstance(next_state, np.ndarray)
