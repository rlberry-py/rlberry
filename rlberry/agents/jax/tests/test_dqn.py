import pytest

from rlberry.envs import gym_make
from rlberry.manager import AgentManager

# Ignoring the text in case jax_agents requirements are not installled
# TODO: better handle setup to avoid conflicts between jax_agents and tensorboard.
_IMPORT_SUCCESSFUL = True
try:
    from rlberry.agents.jax.dqn.dqn import DQNAgent
except ImportError:
    _IMPORT_SUCCESSFUL = False


@pytest.mark.parametrize("lambda_", [None, 0.1])
def test_jax_dqn(lambda_):
    if not _IMPORT_SUCCESSFUL:
        return

    env = (gym_make, dict(id='CartPole-v0'))
    params = dict(
        chunk_size=4,
        batch_size=128,
        target_update_interval=5,
        lambda_=lambda_
    )

    stats = AgentManager(
        DQNAgent,
        env,
        fit_budget=20,
        eval_env=env,
        init_kwargs=params,
        n_fit=1,
        parallelization='thread',
    )
    stats.fit()
    stats.clear_output_dir()
