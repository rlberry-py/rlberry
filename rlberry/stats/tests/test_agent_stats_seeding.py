from rlberry.envs.tests.test_env_seeding import get_env_trajectory, compare_trajectories
from rlberry.envs import gym_make
from rlberry.envs.classic_control import MountainCar
from rlberry.stats import AgentStats, MultipleStats
from rlberry.agents.kernel_based import RSUCBVIAgent


def test_agent_stats_and_multiple_stats_seeding():
    for env in [MountainCar(), (gym_make, {'env_name': 'MountainCar-v0'})]:
        agent_stats = AgentStats(RSUCBVIAgent,
                                 env,
                                 init_kwargs={'n_episodes': 2, 'horizon': 10},
                                 n_fit=6,
                                 seed=3456)
        agent_stats_test = AgentStats(RSUCBVIAgent,
                                      env,
                                      init_kwargs={'n_episodes': 2, 'horizon': 10},
                                      n_fit=6,
                                      seed=3456)

        mstats = MultipleStats()
        mstats.append(agent_stats)
        mstats.append(agent_stats_test)
        mstats.run()

        stats1, stats2 = mstats.allstats

        for ii in range(2, agent_stats.n_fit):
            traj1 = get_env_trajectory(stats1.fitted_agents[ii-2].env, horizon=10)
            traj2 = get_env_trajectory(stats1.fitted_agents[ii-1].env, horizon=10)
            traj3 = get_env_trajectory(stats1.fitted_agents[ii].env, horizon=10)

            traj1_test = get_env_trajectory(stats2.fitted_agents[ii-2].env, horizon=10)
            traj2_test = get_env_trajectory(stats2.fitted_agents[ii-1].env, horizon=10)
            traj3_test = get_env_trajectory(stats2.fitted_agents[ii].env, horizon=10)

            assert not compare_trajectories(traj1, traj2)
            assert not compare_trajectories(traj1, traj3)
            assert not compare_trajectories(traj2, traj3)
            assert compare_trajectories(traj1, traj1_test)
            assert compare_trajectories(traj2, traj2_test)
            assert compare_trajectories(traj3, traj3_test)

        for ii in range(2, agent_stats.n_fit):
            rand1 = stats1.fitted_agents[ii-2].seeder.rng.integers(2**32)
            rand2 = stats1.fitted_agents[ii-1].seeder.rng.integers(2**32)
            rand3 = stats1.fitted_agents[ii].seeder.rng.integers(2**32)

            rand1_test = stats2.fitted_agents[ii-2].seeder.rng.integers(2**32)
            rand2_test = stats2.fitted_agents[ii-1].seeder.rng.integers(2**32)
            rand3_test = stats2.fitted_agents[ii].seeder.rng.integers(2**32)

            assert rand1 != rand2
            assert rand1 != rand3
            assert rand2 != rand3
            assert rand1 == rand1_test
            assert rand2 == rand2_test
            assert rand3 == rand3_test
