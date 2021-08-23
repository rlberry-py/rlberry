"""
TODO: update according to new interface
"""

from stable_baselines3 import A2C as A2CStableBaselines
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from rlberry.agents import Agent
from rlberry.stats import AgentStats, MultipleStats
from rlberry.wrappers.scalarize import ScalarizeEnvWrapper


class A2CAgent(Agent):

    name = 'A2C'

    def __init__(self,
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
                 **kwargs):

        # init rlberry base class
        Agent.__init__(self, env, **kwargs)

        # Generate seed for A2CStableBaselines using rlberry seeding
        seed = self.rng.integers(2**32).item()

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
            _init_setup_model)

    def fit(self, **kwargs):
        result = self.wrapped.learn(**kwargs)

    def policy(self, observation, **kwargs):
        action, _state = self.wrapped.predict(observation, **kwargs)
        return action

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)

        return {'learning_rate': learning_rate}


#
# Training one agent
#
def env_constructor(n_envs=4):
    env = make_atari_env('MontezumaRevenge-v0', n_envs=n_envs)
    env = VecFrameStack(env, n_stack=4)
    return env


def eval_env_constructor(n_envs=1):
    """
    Evaluation should be in a scalar environment.
    """
    env = make_atari_env('MontezumaRevenge-v0', n_envs=n_envs)
    env = VecFrameStack(env, n_stack=4)
    env = ScalarizeEnvWrapper(env)
    return env

#
# Training several agents and comparing different hyperparams
#


stats = AgentStats(
    A2CAgent,
    (env_constructor, None),
    eval_env=(eval_env_constructor, None),
    eval_horizon=200,
    agent_name='A2C baseline',
    init_kwargs={'policy': 'CnnPolicy', 'verbose': 10},
    fit_kwargs={'total_timesteps': 1000},
    policy_kwargs={'deterministic': True},
    n_fit=10,
    n_jobs=6,
    joblib_backend='threading')


stats_alternative = AgentStats(
    A2CAgent,
    (env_constructor, None),
    eval_env=(eval_env_constructor, None),
    eval_horizon=200,
    agent_name='A2C high learning rate',
    init_kwargs={'policy': 'CnnPolicy', 'verbose': 10, 'learning_rate': 0.01},
    fit_kwargs={'total_timesteps': 1000},
    policy_kwargs={'deterministic': True},
    n_fit=10,
    n_jobs=6,
    joblib_backend='threading')


# Fit everything in parallel
mstats = MultipleStats()
mstats.append(stats)
mstats.append(stats_alternative)
mstats.run()
mstats.save()


# Test hyperparam optim
print("testint a call to hyperparam optim")
mstats.allstats[0].optimize_hyperparams(timeout=60, n_fit=2, n_jobs=1)
