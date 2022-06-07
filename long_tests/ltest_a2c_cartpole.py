from rlberry.envs import gym_make
from stable_baselines3 import A2C as A2C
from rlberry.agents.stable_baselines import StableBaselinesAgent
from rlberry.agents.torch import A2CAgent
from rlberry.manager import AgentManager, MultipleManagers, evaluate_agents
from rlberry.agents.torch.utils.training import model_factory_from_env

# Using parameters from https://github.com/araffin/rl-baselines-zoo/tree/master/hyperparams
policy_configs = {
    "type": "MultiLayerPerceptron",  # A network architecture
    "layer_sizes": (64, 64),  # Network dimensions
    "reshape": False,
    "is_policy": True,  # The network should output a distribution
    # over actions
}

critic_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": (64, 64),
    "reshape": False,
    "out_size": 1,  # The critic network is an approximator of
    # a value function V: States -> |R
}


def test_a2c_cartpole_vs_stablebaseline():
    env_ctor = gym_make
    env_kwargs = dict(id="CartPole-v1")

    # Pass the wrapper directly with init_kwargs
    sbagent = AgentManager(
        StableBaselinesAgent,
        (env_ctor, env_kwargs),
        agent_name="A2C stablebaseline",
        init_kwargs=dict(algo_cls=A2C, policy="MlpPolicy", ent_coef=0.0),
        fit_budget=5e5,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=8,
        parallelization="process",
        mp_context="fork",
        seed=42,
    )

    # Pass a subclass for hyperparameter optimization
    rbagent = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        agent_name="A2C rlberry",
        init_kwargs=dict(
            policy_net_fn=model_factory_from_env,  # A policy network constructor
            policy_net_kwargs=policy_configs,  # Policy network's architecure
            value_net_fn=model_factory_from_env,  # A Critic network constructor
            value_net_kwargs=critic_configs,  # Critic network's architecure.
            entr_coef=0.0,  # How much to force exploration.
            batch_size=1024,  # Number of interactions used to
            learning_rate=0.0007,
        ),
        fit_budget=5e5,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=8,
        parallelization="process",
        mp_context="fork",
        seed=42,
    )

    # Fit everything in parallel
    multimanagers = MultipleManagers(parallelization="thread")
    multimanagers.append(sbagent)
    multimanagers.append(rbagent)

    multimanagers.run()

    # Plot policy evaluation
    out = evaluate_agents(multimanagers.managers)
    assert np.mean(out["A2C rlberry"]) > np.mean(out["A2C stablebaseline"]) - 10
