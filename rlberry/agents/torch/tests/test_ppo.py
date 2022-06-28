from rlberry.envs import gym_make
from rlberry.agents.torch.ppo import PPOAgent


env = (gym_make, dict(id="Acrobot-v1"))
# env = gym_make(id="Acrobot-v1")
ppo = PPOAgent(env)
ppo.fit(4096)
