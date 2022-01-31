"""
 =====================
 Demo: demo_from_stable_baselines_atari
 =====================
"""
from stable_baselines3 import A2C as A2CStableBaselines
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import AgentManager
from rlberry.wrappers.scalarize import ScalarizeEnvWrapper
from pathlib import Path


class A2CAgent(AgentWithSimplePolicy):
    name = "A2C"

    def __init__(
        self,
        env,
        policy,
        learning_rate=7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log=None,
        create_eval_env=False,
        policy_kwargs=None,
        verbose: int = 0,
        seed=None,
        device="auto",
        _init_setup_model: bool = True,
        **kwargs
    ):
        # init rlberry base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        # rlberry accepts tuples (env_constructor, env_kwargs) as env
        # After a call to __init__, self.env is set as an environment instance
        env = self.env

        # Generate seed for A2CStableBaselines using rlberry seeding
        seed = self.rng.integers(2 ** 32).item()

        # init stable baselines class
        self.wrapped = A2CStableBaselines(
            policy,
            env,
            learning_rate,
            n_steps,
            gamma,
            gae_lambda,
            ent_coef,
            vf_coef,
            max_grad_norm,
            rms_prop_eps,
            use_rms_prop,
            use_sde,
            sde_sample_freq,
            normalize_advantage,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

    def fit(self, budget):
        self.wrapped.learn(total_timesteps=budget)

    def policy(self, observation):
        action, _ = self.wrapped.predict(observation, deterministic=True)
        return action

    #
    # Some agents are not pickable: in this case, they must define custom save/load methods.
    #
    def save(self, filename):
        self.wrapped.save(filename)
        return Path(filename).with_suffix(".zip")

    @classmethod
    def load(cls, filename, **kwargs):
        rlberry_a2c_wrapper = cls(**kwargs)
        rlberry_a2c_wrapper.wrapped = A2CStableBaselines.load(filename)
        return rlberry_a2c_wrapper

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)

        return {"learning_rate": learning_rate}


#
# Train and eval env constructors
#
def env_constructor(n_envs=4):
    env = make_atari_env("MontezumaRevenge-v0", n_envs=n_envs)
    env = VecFrameStack(env, n_stack=4)
    return env


def eval_env_constructor(n_envs=1):
    """
    Evaluation should be in a scalar environment.
    """
    env = make_atari_env("MontezumaRevenge-v0", n_envs=n_envs)
    env = VecFrameStack(env, n_stack=4)
    env = ScalarizeEnvWrapper(env)
    return env


#
# Testing single agent
#


if __name__ == "__main__":
    #
    # Training several agents and comparing different hyperparams
    #

    stats = AgentManager(
        A2CAgent,
        train_env=(env_constructor, None),
        eval_env=(eval_env_constructor, None),
        eval_kwargs=dict(eval_horizon=200),
        agent_name="A2C baseline",
        fit_budget=5000,
        init_kwargs=dict(policy="CnnPolicy", verbose=10),
        n_fit=4,
        parallelization="process",
        output_dir="dev/stable_baselines_atari",
        seed=123,
    )

    stats.fit()
    stats.optimize_hyperparams(timeout=60, n_fit=2)
