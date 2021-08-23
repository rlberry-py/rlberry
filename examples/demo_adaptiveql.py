import numpy as np
from rlberry.agents import AdaptiveQLAgent
from rlberry.agents import RSUCBVIAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.stats import MultipleStats, AgentStats, plot_writer_data, evaluate_agents
import matplotlib.pyplot as plt

env = (get_benchmark_env, dict(level=2))

N_EP = 5000
HORIZON = 30

params = {}
params['adaql'] = {
    'horizon': HORIZON,
    'gamma': 1.0,
    'bonus_scale_factor': 1.0
}

params['rsucbvi'] = {
    'horizon': HORIZON,
    'gamma': 1.0,
    'bonus_scale_factor': 1.0,
    'min_dist': 0.05,
    'max_repr': 800
}

eval_kwargs = dict(eval_horizon=HORIZON, n_simulations=20)

mstats = MultipleStats()
mstats.append(
    AgentStats(AdaptiveQLAgent,
               env,
               fit_budget=N_EP,
               init_kwargs=params['adaql'],
               eval_kwargs=eval_kwargs,
               n_fit=4,
               n_jobs=4)
)
mstats.append(
    AgentStats(RSUCBVIAgent,
               env,
               fit_budget=N_EP,
               init_kwargs=params['rsucbvi'], n_fit=2)
)

mstats.run(save=False)

evaluate_agents(mstats.allstats)

plot_writer_data(mstats.allstats, tag='episode_rewards',
                 preprocess_func=np.cumsum, title='Cumulative Rewards')

for stats in mstats.allstats:
    agent = stats.fitted_agents[0]
    try:
        agent.Qtree.plot(0, 25)
    except AttributeError:
        pass
plt.show()

for stats in mstats.allstats:
    print(f'Agent = {stats.agent_name}, Eval = {stats.eval()}')
