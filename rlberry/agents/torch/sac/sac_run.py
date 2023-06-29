import gym
from rlberry.agents.torch.sac import SACAgent
# from rlberry.envs import Pendulum
from rlberry.manager import AgentManager


def env_ctor(env, wrap_spaces=True):
    # return Wrapper(env, wrap_spaces)
    return env


env = gym.make("Hopper-v2")
# env = Pendulum()
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
env = gym.wrappers.RecordEpisodeStatistics(env)

env_kwargs = dict(env=env)
agent = AgentManager(
    SACAgent,
    (env_ctor, env_kwargs),
    fit_budget=int(1e6),
    n_fit=1,
    enable_tensorboard=True,
    agent_name="RLBuffer_BackToRLBerry_Hopper1M",
)
agent.fit()
