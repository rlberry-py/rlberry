from rlberry.envs import gym_make
from stable_baselines3 import A2C as A2CStableBaselines
from rlberry.agents import Agent


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
        info = {}  # possibly store something from results
        return info

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


env = gym_make('CartPole-v1')
agent = A2CAgent(env, 'MlpPolicy', verbose=1)
agent.fit(total_timesteps=1000)

obs = env.reset()
for i in range(1000):
    action = agent.policy(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
env.close()

#
# Training several agents and comparing different hyperparams
#
from rlberry.stats import AgentStats, MultipleStats, compare_policies

stats = AgentStats(
    A2CAgent,
    env,
    eval_horizon=200,
    agent_name='A2C baseline',
    init_kwargs={'policy': 'MlpPolicy', 'verbose': 1},
    fit_kwargs={'total_timesteps': 1000},
    policy_kwargs={'deterministic': True},
    n_fit=4,
    n_jobs=4,
    joblib_backend='loky')   # we might need 'threading' here, since stable baselines creates processes
                             # 'multiprocessing' does not work, 'loky' seems good

stats_alternative = AgentStats(
    A2CAgent,
    env,
    eval_horizon=200,
    agent_name='A2C high learning rate',
    init_kwargs={'policy': 'MlpPolicy', 'verbose': 1, 'learning_rate': 0.01},
    fit_kwargs={'total_timesteps': 1000},
    policy_kwargs={'deterministic': True},
    n_fit=4,
    n_jobs=4,
    joblib_backend='loky')

# Fit everything in parallel
mstats = MultipleStats()
mstats.append(stats)
mstats.append(stats_alternative)

mstats.run()

# Plot policy evaluation
compare_policies(mstats.allstats)

# Test hyperparam optim
print("testint a call to hyperparam optim")
mstats.allstats[0].optimize_hyperparams(timeout=60)
