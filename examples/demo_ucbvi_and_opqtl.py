import numpy as np
from rlberry.agents.ucbvi import UCBVIAgent
from rlberry.agents.optql import OptQLAgent
from rlberry.envs.finite import GridWorld
from rlberry.stats import AgentStats, plot_writer_data
from rlberry.stats import MultipleStats


N_EP = 3000
HORIZON = 20
GAMMA = 1.0

env = GridWorld(nrows=5, ncols=10)

params = {}

params['ucbvi'] = {
    'n_episodes': N_EP,
    'horizon': HORIZON,
    'stage_dependent': True,
    'gamma': GAMMA,
    'real_time_dp': True,
    'bonus_scale_factor': 1.0,
}

params['optql'] = {
    'n_episodes': N_EP,
    'horizon': HORIZON,
    'gamma': GAMMA,
    'bonus_scale_factor': 1.0,
}

mstats = MultipleStats()

mstats.append(
    AgentStats(UCBVIAgent, env, init_kwargs=params['ucbvi'])
)

mstats.append(
    AgentStats(OptQLAgent, env, init_kwargs=params['optql'])
)

mstats.run()

plot_writer_data(mstats.allstats, tag='episode_rewards',
                 preprocess_func=np.cumsum,
                 title='Cumulative Rewards')
