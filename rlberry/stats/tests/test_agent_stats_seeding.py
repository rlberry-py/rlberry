from rlberry.envs.tests.test_env_seeding import get_env_trajectory, compare_trajectories
from rlberry.envs import gym_make
from rlberry.envs.classic_control import MountainCar
from rlberry.stats import AgentStats
from rlberry.agents.kernel_based import RSUCBVIAgent
import rlberry.seeding as sd


def test_agent_stats_seeding():
    sd.set_global_seed(3456)
    for env in [MountainCar(), (gym_make, {'env_name': 'MountainCar-v0'})]:
        agent_stats = AgentStats(RSUCBVIAgent,
                                 env,
                                 init_kwargs={'n_episodes': 2, 'horizon': 10},
                                 n_fit=6)
        agent_stats.fit()

        for ii in range(2, agent_stats.n_fit):
            traj1 = get_env_trajectory(agent_stats.fitted_agents[ii-2].env, horizon=10)
            traj2 = get_env_trajectory(agent_stats.fitted_agents[ii-1].env, horizon=10)
            traj3 = get_env_trajectory(agent_stats.fitted_agents[ii].env, horizon=10)
            assert not compare_trajectories(traj1, traj2)
            assert not compare_trajectories(traj1, traj3)
            assert not compare_trajectories(traj2, traj3)
