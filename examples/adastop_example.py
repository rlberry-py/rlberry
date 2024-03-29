"""
===========================================
Compare PPO and A2C on Acrobot with AdaStop
===========================================

This example illustrate the use of adastop_comparator which uses adaptive multiple-testing to assess whether trained agents are
statistically different or not.

Remark that in the case where two agents are not deemed statistically different it can mean either that they are as efficient,
or it can mean that there have not been enough fits to assess the variability of the agents.

Results in

.. code-block::

    [INFO] 13:35: Test finished
    [INFO] 13:35: Results are
      Agent1 vs Agent2  mean Agent1  mean Agent2  mean diff  std Agent 1  std Agent 2 decisions
    0       A2C vs PPO     -274.274      -85.068   -189.206    185.82553      2.71784   smaller


"""

from rlberry.envs import gym_make
from stable_baselines3 import A2C, PPO
from rlberry.agents.stable_baselines import StableBaselinesAgent
from rlberry.manager import AdastopComparator

env_ctor, env_kwargs = gym_make, dict(id="Acrobot-v1")

managers = [
    {
        "agent_class": StableBaselinesAgent,
        "train_env": (env_ctor, env_kwargs),
        "fit_budget": 5e4,
        "agent_name": "A2C",
        "init_kwargs": {"algo_cls": A2C, "policy": "MlpPolicy", "verbose": 1},
    },
    {
        "agent_class": StableBaselinesAgent,
        "train_env": (env_ctor, env_kwargs),
        "agent_name": "PPO",
        "fit_budget": 5e4,
        "init_kwargs": {"algo_cls": PPO, "policy": "MlpPolicy", "verbose": 1},
    },
]

comparator = AdastopComparator()
comparator.compare(managers)
print(comparator.managers_paths)
