import pytest
import numpy as np
from rlberry.envs import GridWorld
from rlberry.envs import MountainCar
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.exploration_tools.online_discretization_counter import OnlineDiscretizationCounter


@pytest.mark.parametrize("rate_power", [0.5, 1])
def test_discrete_env(rate_power):
    env = GridWorld()
    counter = DiscreteCounter(env.observation_space, env.action_space, rate_power=rate_power)

    for N in range(10, 20):
        assert counter.get_n_visited_states() == 0
        assert counter.get_entropy() == 0.0

        for ss in range(env.observation_space.n):
            for aa in range(env.action_space.n):
                for _ in range(N):
                    ns, rr, _, _ = env.sample(ss, aa)
                    counter.update(ss, aa, ns, rr)
                assert counter.N_sa[ss, aa] == N
                assert counter.count(ss, aa) == N
                if rate_power == pytest.approx(1):
                    assert np.allclose(counter.measure(ss, aa), 1.0 / N)
                elif rate_power == pytest.approx(0.5):
                    assert np.allclose(counter.measure(ss, aa), np.sqrt(1.0 / N))

        assert counter.get_n_visited_states() == env.observation_space.n
        assert np.allclose(counter.get_entropy(), np.log2(env.observation_space.n))

        counter.reset()


@pytest.mark.parametrize("rate_power", [0.5, 1])
def test_continuous_state_env(rate_power):
    env = MountainCar()
    counter = DiscreteCounter(env.observation_space, env.action_space, rate_power=rate_power)

    for N in [10, 20]:
        for _ in range(50):
            ss = env.observation_space.sample()
            aa = env.action_space.sample()
            for _ in range(N):
                ns, rr, _, _ = env.sample(ss, aa)
                counter.update(ss, aa, ns, rr)

            dss = counter.state_discretizer.discretize(ss)
            assert counter.N_sa[dss, aa] == N
            assert counter.count(ss, aa) == N
            if rate_power == pytest.approx(1):
                assert np.allclose(counter.measure(ss, aa), 1.0 / N)
            elif rate_power == pytest.approx(0.5):
                assert np.allclose(counter.measure(ss, aa), np.sqrt(1.0 / N))
            counter.reset()


@pytest.mark.parametrize("rate_power", [True, False])
def test_continuous_state_env_2(rate_power):
    env = MountainCar()
    counter = OnlineDiscretizationCounter(env.observation_space,
                                          env.action_space,
                                          rate_power=rate_power)

    for N in [10, 20]:
        for _ in range(50):
            ss = env.observation_space.sample()
            aa = env.action_space.sample()
            for nn in range(N):
                ns, rr, _, _ = env.sample(ss, aa)
                counter.update(ss, aa, ns, rr)
            assert counter.count(ss, aa) == N
            if rate_power == pytest.approx(1):
                assert np.allclose(counter.measure(ss, aa), 1.0 / N)
            elif rate_power == pytest.approx(0.5):
                assert np.allclose(counter.measure(ss, aa), np.sqrt(1.0 / N))
            counter.reset()


def test_continuous_state_env_3():
    env = NRoom(nrooms=3, array_observation=True)
    counter = OnlineDiscretizationCounter(env.observation_space,
                                          env.action_space,
                                          rate_power=0.5,
                                          min_dist=0.0)

    for N in range(10, 20):
        assert counter.get_n_visited_states() == 0
        assert counter.get_entropy() == 0.0

        for ss in range(env.discrete_observation_space.n):
            for aa in range(env.action_space.n):
                for _ in range(N):
                    ns, rr, _, _ = env.sample(ss, aa)
                    continuous_ss = env._convert_index_to_float_coord(ss)
                    counter.update(continuous_ss, aa, None, rr)
                assert counter.N_sa[ss, aa] == N
                assert counter.count(continuous_ss, aa) == N
                assert np.allclose(counter.measure(continuous_ss, aa), np.sqrt(1.0 / N))

        assert counter.get_n_visited_states() == env.discrete_observation_space.n
        assert np.allclose(counter.get_entropy(), np.log2(env.discrete_observation_space.n))

        counter.reset()
