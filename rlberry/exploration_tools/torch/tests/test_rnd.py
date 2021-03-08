from rlberry.exploration_tools.torch.rnd import RandomNetworkDistillation
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env


def test_rnd():
    # Environment
    env = get_benchmark_env(level=1)

    # RND
    rnd = RandomNetworkDistillation(
        env.observation_space,
        env.action_space,
        learning_rate=0.1,
        update_period=100,
        embedding_dim=2)

    # Test
    state = env.reset()
    for ii in range(1000):
        action = env.action_space.sample()
        next_s, reward, _, _ = env.step(action)
        rnd.update(state, action, next_s, reward)
        state = next_s
        # measure uncertainty
        _ = rnd.measure(state, action)
