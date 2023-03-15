"""
 =====================
 Demo: demo_rnd
 =====================
"""
from rlberry.exploration_tools.torch.rnd import RandomNetworkDistillation
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env

# Environment
env = get_benchmark_env(level=1)

# RND
rnd = RandomNetworkDistillation(
    env.observation_space,
    env.action_space,
    learning_rate=0.1,
    update_period=100,
    embedding_dim=2,
)

# Test
observation,info = env.reset()
for ii in range(20000):
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info  = env.step(action)
    rnd.update(observation, action, next_observation, reward)
    observation = next_observation

    if ii % 500 == 0:
        state,info = env.reset()
        bonus = rnd.measure(observation, action)
        print("it = {}, bonus = {}, loss = {}".format(ii, bonus, rnd.loss.item()))
