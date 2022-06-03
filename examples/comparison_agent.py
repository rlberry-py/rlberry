from rlberry.manager import AgentManager, AgentComparer
from rlberry.agents.torch import A2CAgent
from rlberry.envs import gym_make


# GST definition

K = 10  # at most 10 interim
alpha = 0.1
n = 8  # size of a group

comparer = AgentComparer(n, K, alpha)

# DeepRL agent definition
env_ctor = gym_make
env_kwargs = dict(id="CartPole-v1")
seed = 42
budget = 1e4


if __name__ == "__main__":
    manager1 = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        fit_budget=budget,
        seed=seed,
        eval_kwargs=dict(eval_horizon=500),
        init_kwargs=dict(
            learning_rate=1e-3, entr_coef=0.0  # Size of the policy gradient
        ),
        parallelization="process",
        mp_context="forkserver",
    )
    manager2 = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        fit_budget=budget,
        seed=seed,
        init_kwargs=dict(
            learning_rate=1e-3,  # Size of the policy gradient
            entr_coef=0.0,
            batch_size=1024,
        ),
        eval_kwargs=dict(eval_horizon=500),
        agent_name="A2C_tuned",
        parallelization="process",
        mp_context="forkserver",
    )
    comparer.compare(manager1, manager2)
