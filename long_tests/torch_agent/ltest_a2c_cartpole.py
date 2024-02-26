from rlberry.envs import gym_make
from rlberry_research.agents.torch import A2CAgent
from rlberry.manager import ExperimentManager
from rlberry_research.agents.torch.utils.training import model_factory_from_env
import numpy as np

# Using parameters from deeprl quick start
policy_configs = {
    "type": "MultiLayerPerceptron",  # A network architecture
    "layer_sizes": (64, 64),  # Network dimensions
    "reshape": False,
    "is_policy": True,
}

critic_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": (64, 64),
    "reshape": False,
    "out_size": 1,
}


def test_a2c_cartpole():
    env_ctor = gym_make
    env_kwargs = dict(id="CartPole-v1")

    rb_xp = ExperimentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        agent_name="A2CAgent",
        init_kwargs=dict(
            policy_net_fn=model_factory_from_env,
            policy_net_kwargs=policy_configs,
            value_net_fn=model_factory_from_env,
            value_net_kwargs=critic_configs,
            entr_coef=0.0,
            batch_size=1024,
            optimizer_type="ADAM",
            learning_rate=1e-3,
        ),
        fit_budget=3e5,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=8,
        parallelization="process",
        mp_context="fork",
        seed=42,
    )

    rb_xp.fit()
    writer_data = rb_xp.get_writer_data()
    id500 = [
        writer_data[idx].loc[writer_data[idx]["tag"] == "episode_rewards", "value"]
        == 500
        for idx in writer_data
    ]  # ids of the episodes at which reward 500 is attained
    gstep500 = [
        writer_data[idx]
        .loc[writer_data[idx]["tag"] == "episode_rewards"]
        .loc[id500[idx], "global_step"]
        for idx in writer_data
    ]  # global steps of the episodes at which reward 500 is attained
    quantiles500 = [
        np.quantile(gsteps, 0.1) for gsteps in gstep500
    ]  # 10% quantile of these global steps
    assert (
        np.mean(quantiles500) < 155_000
    )  # this value corresponds to performances in v0.3.0
