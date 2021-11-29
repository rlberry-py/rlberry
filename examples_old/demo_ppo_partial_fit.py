import numpy as np
from rlberry.agents.torch.ppo import PPOAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.manager import AgentManager, plot_writer_data, evaluate_agents

env = (PBall2D, dict())
n_episodes = 400
horizon = 100

ppo_params = {}
ppo_params['horizon'] = 100
ppo_params['gamma'] = 0.99
ppo_params['learning_rate'] = 0.001
ppo_params['eps_clip'] = 0.2
ppo_params['k_epochs'] = 4

eval_kwargs = dict(eval_horizon=horizon, n_simulations=20)

ppo_stats = AgentManager(
    PPOAgent, env, fit_budget=n_episodes, eval_kwargs=eval_kwargs,
    init_kwargs=ppo_params, n_fit=2)
ppo_stats.fit(n_episodes // 2)
plot_writer_data(ppo_stats, tag='episode_rewards',
                 preprocess_func=np.cumsum,
                 title='Cumulative Rewards', show=False)
evaluate_agents([ppo_stats], show=False)
ppo_stats.fit(n_episodes // 4)
plot_writer_data(ppo_stats, tag='episode_rewards',
                 preprocess_func=np.cumsum,
                 title='Cumulative Rewards', show=False)
evaluate_agents([ppo_stats], show=True)
