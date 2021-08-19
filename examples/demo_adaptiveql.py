from rlberry.agents import AdaptiveQLAgent
from rlberry.agents import RSUCBVIAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.stats import MultipleStats, AgentStats, plot_episode_rewards
import matplotlib.pyplot as plt
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms


env = TwinRooms()
# env = get_benchmark_env(level=2)

N_EP = 5
HORIZON = 30

params = {}
params['adaql'] = {
    'n_episodes': N_EP,
    'horizon': HORIZON,
    'gamma': 1.0,
    'bonus_scale_factor': 1.0
}

params['rsucbvi'] = {
    'n_episodes': N_EP,
    'horizon': HORIZON,
    'gamma': 1.0,
    'bonus_scale_factor': 1.0,
    'min_dist': 0.05,
    'max_repr': 800
}

mstats = MultipleStats()
mstats.append(
    AgentStats(AdaptiveQLAgent,
               env,
               init_kwargs=params['adaql'],
               n_fit=4,
               n_jobs=4)
)
mstats.append(
    AgentStats(RSUCBVIAgent, env, init_kwargs=params['rsucbvi'], n_fit=2)
)

mstats.run(save=False)

plot_episode_rewards(mstats.allstats, cumulative=True)

for stats in mstats.allstats:
    agent = stats.fitted_agents[0]
    try:
        agent.Qtree.plot(0, 25)
    except AttributeError:
        pass
plt.show()
