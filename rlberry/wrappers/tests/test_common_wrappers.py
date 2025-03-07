import numpy as np
import pytest
from rlberry import spaces
from rlberry_research.agents import RSUCBVIAgent
from rlberry_research.envs.classic_control import MountainCar
from rlberry_research.envs.finite import GridWorld
from rlberry.envs.finite_mdp import FiniteMDP
from rlberry_research.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.seeding import Seeder
from rlberry.wrappers.autoreset import AutoResetWrapper
from rlberry.wrappers.discrete2onehot import DiscreteToOneHotWrapper
from rlberry.wrappers.discretize_state import DiscretizeStateWrapper
from rlberry.wrappers.rescale_reward import RescaleRewardWrapper
from rlberry.wrappers.uncertainty_estimator_wrapper import UncertaintyEstimatorWrapper
from rlberry_research.wrappers.vis2d import Vis2dWrapper
from rlberry.wrappers.gym_utils import OldGymCompatibilityWrapper
from rlberry.wrappers.utils import get_base_env


from rlberry.wrappers.tests.old_env.old_acrobot import Old_Acrobot
from rlberry.wrappers.tests.old_env.old_apple_gold import Old_AppleGold
from rlberry.wrappers.tests.old_env.old_four_room import Old_FourRoom
from rlberry.wrappers.tests.old_env.old_gridworld import Old_GridWorld
from rlberry.wrappers.tests.old_env.old_mountain_car import Old_MountainCar
from rlberry.wrappers.tests.old_env.old_nroom import Old_NRoom
from rlberry.wrappers.tests.old_env.old_pendulum import Old_Pendulum
from rlberry.wrappers.tests.old_env.old_pball import Old_PBall2D, Old_SimplePBallND
from rlberry.wrappers.tests.old_env.old_six_room import Old_SixRoom
from rlberry.wrappers.tests.old_env.old_twinrooms import Old_TwinRooms


classes = [
    Old_Acrobot,
    Old_AppleGold,
    Old_FourRoom,
    Old_GridWorld,
    Old_MountainCar,
    Old_NRoom,
    Old_PBall2D,
    Old_Pendulum,
    Old_SimplePBallND,
    Old_SixRoom,
    Old_TwinRooms,
]


@pytest.mark.parametrize("n_bins", list(range(1, 10)))
def test_discretizer(n_bins):
    env = DiscretizeStateWrapper(MountainCar(), n_bins)
    assert env.observation_space.n == n_bins * n_bins

    for _ in range(2):
        observation, info = env.reset()
        for _ in range(50):
            assert env.observation_space.contains(observation)
            action = env.action_space.sample()
            observation, _, _, _, _ = env.step(action)

    for _ in range(100):
        observation = env.observation_space.sample()
        action = env.action_space.sample()
        next_observation, _, _, _, _ = env.sample(observation, action)
        assert env.observation_space.contains(next_observation)

    assert env.unwrapped.name == "MountainCar"


def test_rescale_reward():
    # tolerance
    tol = 1e-14

    rng = Seeder(123).rng

    for _ in range(10):
        # generate random MDP
        S, A = 5, 2
        R = rng.uniform(0.0, 1.0, (S, A))
        P = rng.uniform(0.0, 1.0, (S, A, S))
        for ss in range(S):
            for aa in range(A):
                P[ss, aa, :] /= P[ss, aa, :].sum()
        env = FiniteMDP(R, P)

        # test
        wrapped = RescaleRewardWrapper(env, (-10, 10))
        _ = wrapped.reset()
        for _ in range(100):
            _, reward, _, _, _ = wrapped.sample(
                wrapped.observation_space.sample(), wrapped.action_space.sample()
            )
            assert reward <= 10 + tol and reward >= -10 - tol

        _ = wrapped.reset()
        for _ in range(100):
            _, reward, _, _, _ = wrapped.step(wrapped.action_space.sample())
            assert reward <= 10 + tol and reward >= -10 - tol


@pytest.mark.parametrize("rmin, rmax", [(0, 1), (-1, 1), (-5, 5), (-5, 15)])
def test_rescale_reward_2(rmin, rmax):
    # tolerance
    tol = 1e-15

    # dummy MDP
    S, A = 5, 2
    R = np.ones((S, A))
    P = np.ones((S, A, S))
    for ss in range(S):
        for aa in range(A):
            P[ss, aa, :] /= P[ss, aa, :].sum()
    env = FiniteMDP(R, P)

    # test bounded case
    env.reward_range = (-100, 50)
    wrapped = RescaleRewardWrapper(env, (rmin, rmax))
    xx = np.linspace(-100, 50, num=100)
    for x in xx:
        y = wrapped._rescale(x)
        assert y >= rmin - tol and y <= rmax + tol

    # test unbounded above
    env.reward_range = (-1.0, np.inf)
    wrapped = RescaleRewardWrapper(env, (rmin, rmax))
    xx = np.linspace(-1, 1e2, num=100)
    for x in xx:
        y = wrapped._rescale(x)
        assert y >= rmin - tol and y <= rmax + tol

    # test unbounded below
    env.reward_range = (-np.inf, 1.0)
    wrapped = RescaleRewardWrapper(env, (rmin, rmax))
    xx = np.linspace(-1e2, 1, num=100)
    for x in xx:
        y = wrapped._rescale(x)
        assert y >= rmin - tol and y <= rmax + tol

    # test unbounded
    env.reward_range = (-np.inf, np.inf)
    wrapped = RescaleRewardWrapper(env, (rmin, rmax))
    xx = np.linspace(-1e2, 1e2, num=200)
    for x in xx:
        y = wrapped._rescale(x)
        assert y >= rmin - tol and y <= rmax + tol


@pytest.mark.parametrize("horizon", list(range(1, 10)))
def test_autoreset(horizon):
    # dummy MDP
    S, A = 5, 2
    R = np.ones((S, A))
    P = np.ones((S, A, S))
    for ss in range(S):
        for aa in range(A):
            P[ss, aa, :] /= P[ss, aa, :].sum()
    # initial state = 3
    env = FiniteMDP(R, P, initial_state_distribution=3)
    env = AutoResetWrapper(env, horizon)

    env.reset()
    for tt in range(5 * horizon + 1):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if (tt + 1) % horizon == 0:
            assert observation == 3


def test_uncertainty_est_wrapper():
    env = GridWorld()

    def uncertainty_est_fn(observation_space, action_space):
        return DiscreteCounter(observation_space, action_space)

    w_env = UncertaintyEstimatorWrapper(env, uncertainty_est_fn, bonus_scale_factor=1.0)

    for ii in range(10):
        w_env.reset()
        _, _, _, _, info = w_env.step(0)
        nn = w_env.uncertainty_estimator.count(0, 0)
        assert nn == ii + 1
        assert info["exploration_bonus"] == pytest.approx(1 / np.sqrt(nn))


def test_vis2dwrapper():
    env = MountainCar()
    env = Vis2dWrapper(env, n_bins_obs=20, memory_size=200)

    agent = RSUCBVIAgent(
        env,
        gamma=0.99,
        horizon=200,
        bonus_scale_factor=0.1,
        copy_env=False,
        min_dist=0.1,
    )

    agent.fit(budget=15)
    env.plot_trajectories(show=False)
    env.plot_trajectory_actions(show=False)


def test_discrete2onehot():
    env = DiscreteToOneHotWrapper(GridWorld())
    base_env = get_base_env(env)
    base_env.reseed(123)
    assert isinstance(env.observation_space, spaces.Box)
    for ii in range(env.unwrapped.observation_space.n):
        initial_distr = np.zeros(env.unwrapped.observation_space.n)
        initial_distr[ii] = 1.0
        env.unwrapped.set_initial_state_distribution(initial_distr)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        obs, info = env.reset()
        assert np.array_equal(obs, initial_distr)


@pytest.mark.parametrize("ModelClass", classes)
def test_OldGymCompatibilityWrapper(ModelClass):
    # tester ancien environnement
    env = ModelClass()
    env.reseed(1)
    result = env.reset()
    assert not isinstance(result, tuple)
    action = env.action_space.sample()
    result = env.step(action)
    assert isinstance(result, tuple)
    assert len(result) == 4

    # tester wrapper
    env = ModelClass()
    env = OldGymCompatibilityWrapper(env)
    result = env.reset(seed=42)
    assert isinstance(result, tuple)
    observations, infos = result
    assert isinstance(infos, dict)
    for tt in range(5000):
        action = env.action_space.sample()
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5
        observation, reward, terminated, truncated, info = result
        assert env.observation_space.contains(observation)
        done = terminated or truncated
        if done:
            observation, info = env.reset(42**2)
