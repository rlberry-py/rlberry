from rlberry.envs import gym_make
from rlberry.manager import plot_writer_data, AgentManager, evaluate_agents
from rlberry.agents.experimental.torch import PPOAgent

# import gym
from stable_baselines3 import PPO
from rlberry.agents.stable_baselines import StableBaselinesAgent

# from gym.wrappers import TimeLimit
# import torch

env_name = "Acrobot-v1"

ppo = AgentManager(
    PPOAgent,
    (gym_make, dict(id=env_name)),
    fit_budget=int(1e5),
    eval_kwargs=dict(eval_horizon=500),
    n_fit=1,
    agent_name="RLB_PPO",
)
ppo.fit()
ppo_sb3 = AgentManager(
    StableBaselinesAgent,
    (gym_make, dict(id=env_name)),
    init_kwargs=dict(
        algo_cls=PPO,
        verbose=0,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,
        learning_rate=0.01,
    ),
    # policy_kwargs=policy_kwargs),
    eval_kwargs=dict(eval_horizon=500),
    n_fit=1,
    fit_budget=int(1e5),
    agent_name="SB_PPO",
)
ppo_sb3.fit()
_ = plot_writer_data(
    [ppo, ppo_sb3],
    tag="episode_rewards",
    title="Training Episode Cumulative Rewards",
    show=True,
)
evaluate_agents([ppo, ppo_sb3], n_simulations=50)
