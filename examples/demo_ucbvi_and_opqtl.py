import numpy as np
from rlberry.agents.ucbvi import UCBVIAgent
from rlberry.agents.optql import OptQLAgent
from rlberry.envs.finite import GridWorld
from rlberry.stats import AgentStats, plot_writer_data
from rlberry.stats import MultipleStats


N_EP = 3000
HORIZON = 20
GAMMA = 1.0

env = (GridWorld, dict(nrows=5, ncols=10))

params = {}

params['ucbvi'] = {
    'horizon': HORIZON,
    'stage_dependent': True,
    'gamma': GAMMA,
    'real_time_dp': True,
    'bonus_scale_factor': 1.0,
}

params['optql'] = {
    'horizon': HORIZON,
    'gamma': GAMMA,
    'bonus_scale_factor': 1.0,
}

eval_kwargs = dict(eval_horizon=HORIZON, n_simulations=20)


mstats = MultipleStats()

mstats.append(
    AgentStats(UCBVIAgent, env, fit_budget=N_EP, init_kwargs=params['ucbvi'], eval_kwargs=eval_kwargs)
)

mstats.append(
    AgentStats(OptQLAgent, env, fit_budget=N_EP, init_kwargs=params['optql'], eval_kwargs=eval_kwargs)
)

mstats.run()

plot_writer_data(mstats.allstats, tag='episode_rewards',
                 preprocess_func=np.cumsum,
                 title='Cumulative Rewards')
