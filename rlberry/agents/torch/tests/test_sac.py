import sys

import pytest
from gymnasium import make

from rlberry.agents.torch.sac import SACAgent
from rlberry.envs import Wrapper
from rlberry.manager import AgentManager, evaluate_agents


@pytest.mark.timeout(300)
@pytest.mark.xfail(sys.platform == "win32", reason="bug with windows???")
def test_sac():
    env = "Pendulum-v1"
    mdp = make(env)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    sacrlberry_stats = AgentManager(
        SACAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(132),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=24, device="cpu"),
        n_fit=1,
        agent_name="SAC_rlberry_" + env,
    )

    sacrlberry_stats.fit()

    output = evaluate_agents([sacrlberry_stats], n_simulations=2, plot=False)
    sacrlberry_stats.clear_output_dir()

    # test also non default
    env = "Pendulum-v1"
    mdp = make(env)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    sacrlberry_stats = AgentManager(
        SACAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(1024),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(
            learning_start=int(512),
            autotune_alpha=False,
            batch_size=24,
            policy_net_kwargs=dict(
                type="MultiLayerPerceptron",
                layer_sizes=(256,),
                reshape=False,
                is_policy=True,
            ),
            q_net_kwargs=dict(
                type="MultiLayerPerceptron",
                layer_sizes=[
                    512,
                ],
                reshape=False,
                out_size=1,
            ),
        ),
        n_fit=1,
        agent_name="SAC_rlberry_" + env,
    )
    sacrlberry_stats.fit()
    output = evaluate_agents([sacrlberry_stats], n_simulations=2, plot=False)
    sacrlberry_stats.clear_output_dir()
