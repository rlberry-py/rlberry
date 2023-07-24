from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.exploration_tools.torch.rnd import RandomNetworkDistillation


def test_rnd():
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
    observation, info = env.reset()
    for ii in range(1000):
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rnd.update(observation, action, next_observation, reward)
        observation = next_observation
        # measure uncertainty
        _ = rnd.measure(observation, action)
