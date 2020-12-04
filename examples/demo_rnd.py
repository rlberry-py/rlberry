from rlberry.exploration_tools.rnd import RandomNetworkDistillation
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env

# Environment
env = get_benchmark_env(level=1)

# RND
rnd = RandomNetworkDistillation(
    env.observation_space,
    env.action_space,
    learning_rate=0.01,
    update_period=10)

# Test
state = env.reset()
for ii in range(10000):
    action = env.action_space.sample()
    next_s, reward, _, _ = env.step(action)
    rnd.update(state, action, next_s, reward)
    state = next_s

    if ii % 100 == 0:
        bonus = rnd.measure(state, action)
        print("it = {}, bonus = {}, loss = {}"
              .format(ii, bonus, rnd.loss.item()))
        state = env.reset()
