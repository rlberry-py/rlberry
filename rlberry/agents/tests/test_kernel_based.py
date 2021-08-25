import pytest
from rlberry.agents.kernel_based import RSKernelUCBVIAgent
from rlberry.agents.kernel_based import RSUCBVIAgent
from rlberry.agents.kernel_based.kernels import _str_to_int
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env


@pytest.mark.parametrize("kernel_type", [
    "uniform",
    "triangular",
    "gaussian",
    "epanechnikov",
    "quartic",
    "triweight",
    "tricube",
    "cosine",
    "exp-2"
])
def test_rs_kernel_ucbvi(kernel_type):
    for horizon in [None, 30]:
        env = get_benchmark_env(level=1)
        agent = RSKernelUCBVIAgent(
            env,
            gamma=0.95,
            horizon=horizon,
            bonus_scale_factor=0.01,
            min_dist=0.2,
            bandwidth=0.05,
            beta=1.0,
            kernel_type=kernel_type)
        agent.fit(budget=5)
        agent.policy(env.observation_space.sample())


def test_str_to_int():
    for ii in range(100):
        assert _str_to_int(str(ii)) == ii


def test_rs_ucbvi():
    env = get_benchmark_env(level=1)
    agent = RSUCBVIAgent(env,
                         gamma=0.99,
                         horizon=30,
                         bonus_scale_factor=0.1)
    agent.fit(budget=5)
    agent.policy(env.observation_space.sample())


def test_rs_ucbvi_reward_free():
    env = get_benchmark_env(level=1)
    agent = RSUCBVIAgent(env,
                         gamma=0.99,
                         horizon=30,
                         bonus_scale_factor=0.1,
                         reward_free=True)
    agent.fit(budget=5)
    agent.policy(env.observation_space.sample())
    assert agent.R_hat.sum() == 0.0
