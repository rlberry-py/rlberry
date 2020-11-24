from rlberry.agents.ppo import PPOAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.seeding import seeding
from rlberry.stats import AgentStats, plot_episode_rewards, compare_policies
seeding.set_global_seed(1223)


env = PBall2D()
n_episodes = 400
horizon = 100

ppo_params = {}
ppo_params['n_episodes'] = 400
ppo_params['horizon'] = 100
ppo_params['gamma'] = 0.99
ppo_params['learning_rate'] = 0.001
ppo_params['eps_clip'] = 0.2
ppo_params['k_epochs'] = 4
ppo_params['verbose'] = 5

ppo_stats = AgentStats(PPOAgent, env, eval_horizon=100,
                       init_kwargs=ppo_params, n_fit=2)
ppo_stats.partial_fit(0.3)
plot_episode_rewards([ppo_stats], show=False, cumulative=True)
compare_policies([ppo_stats], show=False)
ppo_stats.partial_fit(0.2)
plot_episode_rewards([ppo_stats], show=False, cumulative=True)
compare_policies([ppo_stats], show=True)
