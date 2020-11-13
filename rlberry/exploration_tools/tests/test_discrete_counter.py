from rlberry.envs import GridWorld
from rlberry.envs import MountainCar
from rlberry.exploration_tools.discrete_counter import DiscreteCounter 
from rlberry.exploration_tools.online_discretization_counter import OnlineDiscretizationCounter


def test_discrete_env():
    env = GridWorld()
    counter = DiscreteCounter(env.observation_space, env.action_space)
    
    for N in range(10, 20):
        for ss in range(env.observation_space.n):
            for aa in range(env.action_space.n):
                for nn in range(N):
                    ns, rr, _, _ = env.sample(ss, aa)
                    counter.update(ss, aa, ns, rr)
                assert counter.N_sa[ss, aa] == N 
                assert counter.count(ss, aa) == N
        counter.reset()


def test_continuous_state_env():
    env = MountainCar()
    counter = DiscreteCounter(env.observation_space, env.action_space)

    for N in [10, 20, 30]:
        for ii in range(100):
            ss = env.observation_space.sample()
            aa = env.action_space.sample()
            for nn in range(N):
                ns, rr, _, _ = env.sample(ss, aa)
                counter.update(ss, aa, ns, rr)
            
            dss = counter.state_discretizer.discretize(ss)
            assert counter.N_sa[dss, aa] == N 
            assert counter.count(ss, aa) == N
            counter.reset()


def test_continuous_state_env_2():
    env = MountainCar()
    counter = OnlineDiscretizationCounter(env.observation_space, env.action_space)

    for N in [10, 20, 30]:
        for ii in range(100):
            ss = env.observation_space.sample()
            aa = env.action_space.sample()
            for nn in range(N):
                ns, rr, _, _ = env.sample(ss, aa)
                counter.update(ss, aa, ns, rr)
            assert counter.count(ss, aa) == N
            counter.reset()
