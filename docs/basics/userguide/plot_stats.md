(plot_stats_page)=

# Plotting and Statistics

Similary to an RL research project or coursework, we would like to compare, the performances of PPO and A2C from [stable-baselines3](stable-baselines3.readthedocs.io/en/master/) on CartPole-v1. As well as draw statistical significant conclusions on which agent is better in terms of average evaluation performances.

## Reminder on agent training with ExperimentManager

We first start by training the agents on 10 seeds each using [ExperimentManagers](https://rlberry-py.github.io/rlberry/generated/rlberry.manager.ExperimentManager.html#rlberry.manager.ExperimentManager) and [StableBaselinesAgent](https://rlberry-py.github.io/rlberry/generated/rlberry.agents.stable_baselines.StableBaselinesAgent.html#rlberry.agents.stable_baselines.StableBaselinesAgent).


```python
from rlberry.envs import gym_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import A2C, PPO
from rlberry.manager import ExperimentManager

env_id = "CartPole-v1"  # Id of the environment

env_ctor = gym_make  # constructor for the env
env_kwargs = dict(id=env_id)  # give the id of the env inside the kwargs

first_agent = ExperimentManager(
    StableBaselinesAgent,  # Agent Class
    init_kwargs=dict(algo_cls=A2C),
    train_env=(env_ctor, env_kwargs),  # Environment as Tuple(constructor,kwargs)
    seed=42,
    fit_budget=int(1e4),
    n_fit=10,
    agent_name="Sb3-A2C",
)
second_agent = ExperimentManager(
    StableBaselinesAgent,  # Agent Class
    init_kwargs=dict(algo_cls=PPO),
    train_env=(env_ctor, env_kwargs),  # Environment as Tuple(constructor,kwargs)
    seed=42,
    fit_budget=int(1e4),
    n_fit=10,
    agent_name="Sb3-PPO",
)
first_agent.fit()
second_agent.fit()
```

    [38;21m[INFO] 14:38: Running ExperimentManager fit() for Sb3-A2C with n_fit = 10 and max_workers = None. [0m
    INFO:rlberry_logger:Running ExperimentManager fit() for Sb3-A2C with n_fit = 10 and max_workers = None.
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38:                agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      2           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      2           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38:                agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      0           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      0           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38:                agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      1           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      1           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38:                agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      5           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      5           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38:                agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      3           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      3           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38:                agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      4           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      4           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 2]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.25064384937286377 | train/entropy_loss = -0.6872977018356323 | train/policy_loss = -1.036084532737732 | train/value_loss = 22.74214744567871 | time/iterations = 100 | rollout/ep_rew_mean = 19.68 | rollout/ep_len_mean = 19.68 | time/fps = 70 | time/time_elapsed = 7 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.25064384937286377 | train/entropy_loss = -0.6872977018356323 | train/policy_loss = -1.036084532737732 | train/value_loss = 22.74214744567871 | time/iterations = 100 | rollout/ep_rew_mean = 19.68 | rollout/ep_len_mean = 19.68 | time/fps = 70 | time/time_elapsed = 7 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 0]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.38402533531188965 | train/entropy_loss = -0.5975228548049927 | train/policy_loss = 1.186388611793518 | train/value_loss = 6.420439720153809 | time/iterations = 100 | rollout/ep_rew_mean = 59.875 | rollout/ep_len_mean = 59.875 | time/fps = 68 | time/time_elapsed = 7 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.38402533531188965 | train/entropy_loss = -0.5975228548049927 | train/policy_loss = 1.186388611793518 | train/value_loss = 6.420439720153809 | time/iterations = 100 | rollout/ep_rew_mean = 59.875 | rollout/ep_len_mean = 59.875 | time/fps = 68 | time/time_elapsed = 7 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 1]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -1.010756254196167 | train/entropy_loss = -0.6501280069351196 | train/policy_loss = 1.4370367527008057 | train/value_loss = 11.72978687286377 | time/iterations = 100 | rollout/ep_rew_mean = 30.625 | rollout/ep_len_mean = 30.625 | time/fps = 69 | time/time_elapsed = 7 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -1.010756254196167 | train/entropy_loss = -0.6501280069351196 | train/policy_loss = 1.4370367527008057 | train/value_loss = 11.72978687286377 | time/iterations = 100 | rollout/ep_rew_mean = 30.625 | rollout/ep_len_mean = 30.625 | time/fps = 69 | time/time_elapsed = 7 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 5]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.0039084553718566895 | train/entropy_loss = -0.6803387403488159 | train/policy_loss = 2.1474685668945312 | train/value_loss = 9.631743431091309 | time/iterations = 100 | rollout/ep_rew_mean = 22.727272727272727 | rollout/ep_len_mean = 22.727272727272727 | time/fps = 67 | time/time_elapsed = 7 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.0039084553718566895 | train/entropy_loss = -0.6803387403488159 | train/policy_loss = 2.1474685668945312 | train/value_loss = 9.631743431091309 | time/iterations = 100 | rollout/ep_rew_mean = 22.727272727272727 | rollout/ep_len_mean = 22.727272727272727 | time/fps = 67 | time/time_elapsed = 7 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 3]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.13930046558380127 | train/entropy_loss = -0.6920222043991089 | train/policy_loss = 1.9530296325683594 | train/value_loss = 9.033199310302734 | time/iterations = 100 | rollout/ep_rew_mean = 16.379310344827587 | rollout/ep_len_mean = 16.379310344827587 | time/fps = 65 | time/time_elapsed = 7 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.13930046558380127 | train/entropy_loss = -0.6920222043991089 | train/policy_loss = 1.9530296325683594 | train/value_loss = 9.033199310302734 | time/iterations = 100 | rollout/ep_rew_mean = 16.379310344827587 | rollout/ep_len_mean = 16.379310344827587 | time/fps = 65 | time/time_elapsed = 7 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 4]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.25177425146102905 | train/entropy_loss = -0.679092288017273 | train/policy_loss = 1.6058944463729858 | train/value_loss = 5.8588714599609375 | time/iterations = 100 | rollout/ep_rew_mean = 24.95 | rollout/ep_len_mean = 24.95 | time/fps = 62 | time/time_elapsed = 7 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.25177425146102905 | train/entropy_loss = -0.679092288017273 | train/policy_loss = 1.6058944463729858 | train/value_loss = 5.8588714599609375 | time/iterations = 100 | rollout/ep_rew_mean = 24.95 | rollout/ep_len_mean = 24.95 | time/fps = 62 | time/time_elapsed = 7 | time/total_timesteps = 500 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 2]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.2021259069442749 | train/entropy_loss = -0.6800739169120789 | train/policy_loss = 1.7649768590927124 | train/value_loss = 9.171698570251465 | time/iterations = 200 | rollout/ep_rew_mean = 20.70731707317073 | rollout/ep_len_mean = 20.70731707317073 | time/fps = 68 | time/time_elapsed = 14 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.2021259069442749 | train/entropy_loss = -0.6800739169120789 | train/policy_loss = 1.7649768590927124 | train/value_loss = 9.171698570251465 | time/iterations = 200 | rollout/ep_rew_mean = 20.70731707317073 | rollout/ep_len_mean = 20.70731707317073 | time/fps = 68 | time/time_elapsed = 14 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 1]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -3.4632983207702637 | train/entropy_loss = -0.3765987753868103 | train/policy_loss = 3.91054105758667 | train/value_loss = 29.23153305053711 | time/iterations = 200 | rollout/ep_rew_mean = 32.733333333333334 | rollout/ep_len_mean = 32.733333333333334 | time/fps = 67 | time/time_elapsed = 14 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -3.4632983207702637 | train/entropy_loss = -0.3765987753868103 | train/policy_loss = 3.91054105758667 | train/value_loss = 29.23153305053711 | time/iterations = 200 | rollout/ep_rew_mean = 32.733333333333334 | rollout/ep_len_mean = 32.733333333333334 | time/fps = 67 | time/time_elapsed = 14 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 0]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.1075516939163208 | train/entropy_loss = -0.37715786695480347 | train/policy_loss = 1.9796581268310547 | train/value_loss = 4.526117324829102 | time/iterations = 200 | rollout/ep_rew_mean = 49.8 | rollout/ep_len_mean = 49.8 | time/fps = 67 | time/time_elapsed = 14 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.1075516939163208 | train/entropy_loss = -0.37715786695480347 | train/policy_loss = 1.9796581268310547 | train/value_loss = 4.526117324829102 | time/iterations = 200 | rollout/ep_rew_mean = 49.8 | rollout/ep_len_mean = 49.8 | time/fps = 67 | time/time_elapsed = 14 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 5]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.10433006286621094 | train/entropy_loss = -0.6235848665237427 | train/policy_loss = 1.1559741497039795 | train/value_loss = 7.417520999908447 | time/iterations = 200 | rollout/ep_rew_mean = 27.514285714285716 | rollout/ep_len_mean = 27.514285714285716 | time/fps = 66 | time/time_elapsed = 15 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.10433006286621094 | train/entropy_loss = -0.6235848665237427 | train/policy_loss = 1.1559741497039795 | train/value_loss = 7.417520999908447 | time/iterations = 200 | rollout/ep_rew_mean = 27.514285714285716 | rollout/ep_len_mean = 27.514285714285716 | time/fps = 66 | time/time_elapsed = 15 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 3]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.0020294785499572754 | train/entropy_loss = -0.6927657127380371 | train/policy_loss = -0.7635567784309387 | train/value_loss = 36.22486877441406 | time/iterations = 200 | rollout/ep_rew_mean = 18.69811320754717 | rollout/ep_len_mean = 18.69811320754717 | time/fps = 65 | time/time_elapsed = 15 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.0020294785499572754 | train/entropy_loss = -0.6927657127380371 | train/policy_loss = -0.7635567784309387 | train/value_loss = 36.22486877441406 | time/iterations = 200 | rollout/ep_rew_mean = 18.69811320754717 | rollout/ep_len_mean = 18.69811320754717 | time/fps = 65 | time/time_elapsed = 15 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 4]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.07203227281570435 | train/entropy_loss = -0.6211680173873901 | train/policy_loss = 1.5115278959274292 | train/value_loss = 7.235849857330322 | time/iterations = 200 | rollout/ep_rew_mean = 23.761904761904763 | rollout/ep_len_mean = 23.761904761904763 | time/fps = 63 | time/time_elapsed = 15 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.07203227281570435 | train/entropy_loss = -0.6211680173873901 | train/policy_loss = 1.5115278959274292 | train/value_loss = 7.235849857330322 | time/iterations = 200 | rollout/ep_rew_mean = 23.761904761904763 | rollout/ep_len_mean = 23.761904761904763 | time/fps = 63 | time/time_elapsed = 15 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 2]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.1774425506591797 | train/entropy_loss = -0.6873862147331238 | train/policy_loss = 1.4590141773223877 | train/value_loss = 7.6238861083984375 | time/iterations = 300 | rollout/ep_rew_mean = 23.806451612903224 | rollout/ep_len_mean = 23.806451612903224 | time/fps = 69 | time/time_elapsed = 21 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.1774425506591797 | train/entropy_loss = -0.6873862147331238 | train/policy_loss = 1.4590141773223877 | train/value_loss = 7.6238861083984375 | time/iterations = 300 | rollout/ep_rew_mean = 23.806451612903224 | rollout/ep_len_mean = 23.806451612903224 | time/fps = 69 | time/time_elapsed = 21 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 1]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.15000396966934204 | train/entropy_loss = -0.570386528968811 | train/policy_loss = 1.0264523029327393 | train/value_loss = 6.628987789154053 | time/iterations = 300 | rollout/ep_rew_mean = 32.8 | rollout/ep_len_mean = 32.8 | time/fps = 68 | time/time_elapsed = 21 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.15000396966934204 | train/entropy_loss = -0.570386528968811 | train/policy_loss = 1.0264523029327393 | train/value_loss = 6.628987789154053 | time/iterations = 300 | rollout/ep_rew_mean = 32.8 | rollout/ep_len_mean = 32.8 | time/fps = 68 | time/time_elapsed = 21 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 0]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.06708848476409912 | train/entropy_loss = -0.5172844529151917 | train/policy_loss = 1.1074496507644653 | train/value_loss = 7.0124335289001465 | time/iterations = 300 | rollout/ep_rew_mean = 44.57575757575758 | rollout/ep_len_mean = 44.57575757575758 | time/fps = 67 | time/time_elapsed = 22 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.06708848476409912 | train/entropy_loss = -0.5172844529151917 | train/policy_loss = 1.1074496507644653 | train/value_loss = 7.0124335289001465 | time/iterations = 300 | rollout/ep_rew_mean = 44.57575757575758 | rollout/ep_len_mean = 44.57575757575758 | time/fps = 67 | time/time_elapsed = 22 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 3]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.0045615434646606445 | train/entropy_loss = -0.6917933821678162 | train/policy_loss = -3.9405109882354736 | train/value_loss = 114.79180908203125 | time/iterations = 300 | rollout/ep_rew_mean = 19.376623376623378 | rollout/ep_len_mean = 19.376623376623378 | time/fps = 67 | time/time_elapsed = 22 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.0045615434646606445 | train/entropy_loss = -0.6917933821678162 | train/policy_loss = -3.9405109882354736 | train/value_loss = 114.79180908203125 | time/iterations = 300 | rollout/ep_rew_mean = 19.376623376623378 | rollout/ep_len_mean = 19.376623376623378 | time/fps = 67 | time/time_elapsed = 22 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 5]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.03214538097381592 | train/entropy_loss = -0.6267425417900085 | train/policy_loss = 1.7444145679473877 | train/value_loss = 6.495245456695557 | time/iterations = 300 | rollout/ep_rew_mean = 32.77272727272727 | rollout/ep_len_mean = 32.77272727272727 | time/fps = 67 | time/time_elapsed = 22 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.03214538097381592 | train/entropy_loss = -0.6267425417900085 | train/policy_loss = 1.7444145679473877 | train/value_loss = 6.495245456695557 | time/iterations = 300 | rollout/ep_rew_mean = 32.77272727272727 | rollout/ep_len_mean = 32.77272727272727 | time/fps = 67 | time/time_elapsed = 22 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:38: [Sb3-A2C[worker: 4]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.1877366304397583 | train/entropy_loss = -0.6559485197067261 | train/policy_loss = 1.2800041437149048 | train/value_loss = 4.668278217315674 | time/iterations = 300 | rollout/ep_rew_mean = 30.659574468085108 | rollout/ep_len_mean = 30.659574468085108 | time/fps = 64 | time/time_elapsed = 23 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.1877366304397583 | train/entropy_loss = -0.6559485197067261 | train/policy_loss = 1.2800041437149048 | train/value_loss = 4.668278217315674 | time/iterations = 300 | rollout/ep_rew_mean = 30.659574468085108 | rollout/ep_len_mean = 30.659574468085108 | time/fps = 64 | time/time_elapsed = 23 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 2]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.1263028383255005 | train/entropy_loss = -0.6847257614135742 | train/policy_loss = 1.2597534656524658 | train/value_loss = 5.19762659072876 | time/iterations = 400 | rollout/ep_rew_mean = 23.376470588235293 | rollout/ep_len_mean = 23.376470588235293 | time/fps = 69 | time/time_elapsed = 28 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.1263028383255005 | train/entropy_loss = -0.6847257614135742 | train/policy_loss = 1.2597534656524658 | train/value_loss = 5.19762659072876 | time/iterations = 400 | rollout/ep_rew_mean = 23.376470588235293 | rollout/ep_len_mean = 23.376470588235293 | time/fps = 69 | time/time_elapsed = 28 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 0]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.22821450233459473 | train/entropy_loss = -0.32986798882484436 | train/policy_loss = 0.5577487945556641 | train/value_loss = 5.3127546310424805 | time/iterations = 400 | rollout/ep_rew_mean = 45.97674418604651 | rollout/ep_len_mean = 45.97674418604651 | time/fps = 67 | time/time_elapsed = 29 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.22821450233459473 | train/entropy_loss = -0.32986798882484436 | train/policy_loss = 0.5577487945556641 | train/value_loss = 5.3127546310424805 | time/iterations = 400 | rollout/ep_rew_mean = 45.97674418604651 | rollout/ep_len_mean = 45.97674418604651 | time/fps = 67 | time/time_elapsed = 29 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 1]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.16450059413909912 | train/entropy_loss = -0.5457640886306763 | train/policy_loss = 1.5679147243499756 | train/value_loss = 6.7226433753967285 | time/iterations = 400 | rollout/ep_rew_mean = 33.333333333333336 | rollout/ep_len_mean = 33.333333333333336 | time/fps = 68 | time/time_elapsed = 29 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.16450059413909912 | train/entropy_loss = -0.5457640886306763 | train/policy_loss = 1.5679147243499756 | train/value_loss = 6.7226433753967285 | time/iterations = 400 | rollout/ep_rew_mean = 33.333333333333336 | rollout/ep_len_mean = 33.333333333333336 | time/fps = 68 | time/time_elapsed = 29 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 5]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.006972908973693848 | train/entropy_loss = -0.6762612462043762 | train/policy_loss = 1.3445746898651123 | train/value_loss = 5.743118762969971 | time/iterations = 400 | rollout/ep_rew_mean = 36.31481481481482 | rollout/ep_len_mean = 36.31481481481482 | time/fps = 67 | time/time_elapsed = 29 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.006972908973693848 | train/entropy_loss = -0.6762612462043762 | train/policy_loss = 1.3445746898651123 | train/value_loss = 5.743118762969971 | time/iterations = 400 | rollout/ep_rew_mean = 36.31481481481482 | rollout/ep_len_mean = 36.31481481481482 | time/fps = 67 | time/time_elapsed = 29 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 3]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.06557649374008179 | train/entropy_loss = -0.6592729091644287 | train/policy_loss = 1.1537708044052124 | train/value_loss = 4.507213592529297 | time/iterations = 400 | rollout/ep_rew_mean = 19.98 | rollout/ep_len_mean = 19.98 | time/fps = 67 | time/time_elapsed = 29 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.06557649374008179 | train/entropy_loss = -0.6592729091644287 | train/policy_loss = 1.1537708044052124 | train/value_loss = 4.507213592529297 | time/iterations = 400 | rollout/ep_rew_mean = 19.98 | rollout/ep_len_mean = 19.98 | time/fps = 67 | time/time_elapsed = 29 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 4]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.04809325933456421 | train/entropy_loss = -0.538925290107727 | train/policy_loss = 0.6070107221603394 | train/value_loss = 5.591319561004639 | time/iterations = 400 | rollout/ep_rew_mean = 35.472727272727276 | rollout/ep_len_mean = 35.472727272727276 | time/fps = 64 | time/time_elapsed = 30 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.04809325933456421 | train/entropy_loss = -0.538925290107727 | train/policy_loss = 0.6070107221603394 | train/value_loss = 5.591319561004639 | time/iterations = 400 | rollout/ep_rew_mean = 35.472727272727276 | rollout/ep_len_mean = 35.472727272727276 | time/fps = 64 | time/time_elapsed = 30 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 2]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.004032015800476074 | train/entropy_loss = -0.6882427334785461 | train/policy_loss = -5.229913711547852 | train/value_loss = 243.8998565673828 | time/iterations = 500 | rollout/ep_rew_mean = 24.12 | rollout/ep_len_mean = 24.12 | time/fps = 69 | time/time_elapsed = 36 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.004032015800476074 | train/entropy_loss = -0.6882427334785461 | train/policy_loss = -5.229913711547852 | train/value_loss = 243.8998565673828 | time/iterations = 500 | rollout/ep_rew_mean = 24.12 | rollout/ep_len_mean = 24.12 | time/fps = 69 | time/time_elapsed = 36 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 0]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.6509398519992828 | train/entropy_loss = -0.5585274696350098 | train/policy_loss = 0.7496447563171387 | train/value_loss = 3.716269016265869 | time/iterations = 500 | rollout/ep_rew_mean = 48.0 | rollout/ep_len_mean = 48.0 | time/fps = 68 | time/time_elapsed = 36 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.6509398519992828 | train/entropy_loss = -0.5585274696350098 | train/policy_loss = 0.7496447563171387 | train/value_loss = 3.716269016265869 | time/iterations = 500 | rollout/ep_rew_mean = 48.0 | rollout/ep_len_mean = 48.0 | time/fps = 68 | time/time_elapsed = 36 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 1]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.22020363807678223 | train/entropy_loss = -0.3855079412460327 | train/policy_loss = 2.1662728786468506 | train/value_loss = 5.747239112854004 | time/iterations = 500 | rollout/ep_rew_mean = 37.059701492537314 | rollout/ep_len_mean = 37.059701492537314 | time/fps = 68 | time/time_elapsed = 36 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.22020363807678223 | train/entropy_loss = -0.3855079412460327 | train/policy_loss = 2.1662728786468506 | train/value_loss = 5.747239112854004 | time/iterations = 500 | rollout/ep_rew_mean = 37.059701492537314 | rollout/ep_len_mean = 37.059701492537314 | time/fps = 68 | time/time_elapsed = 36 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 5]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.04794734716415405 | train/entropy_loss = -0.5888231992721558 | train/policy_loss = 1.482774257659912 | train/value_loss = 5.065307140350342 | time/iterations = 500 | rollout/ep_rew_mean = 37.19672131147541 | rollout/ep_len_mean = 37.19672131147541 | time/fps = 67 | time/time_elapsed = 37 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.04794734716415405 | train/entropy_loss = -0.5888231992721558 | train/policy_loss = 1.482774257659912 | train/value_loss = 5.065307140350342 | time/iterations = 500 | rollout/ep_rew_mean = 37.19672131147541 | rollout/ep_len_mean = 37.19672131147541 | time/fps = 67 | time/time_elapsed = 37 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 3]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.0046073198318481445 | train/entropy_loss = -0.6875606775283813 | train/policy_loss = 1.416259527206421 | train/value_loss = 5.871022701263428 | time/iterations = 500 | rollout/ep_rew_mean = 22.33 | rollout/ep_len_mean = 22.33 | time/fps = 67 | time/time_elapsed = 37 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.0046073198318481445 | train/entropy_loss = -0.6875606775283813 | train/policy_loss = 1.416259527206421 | train/value_loss = 5.871022701263428 | time/iterations = 500 | rollout/ep_rew_mean = 22.33 | rollout/ep_len_mean = 22.33 | time/fps = 67 | time/time_elapsed = 37 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 4]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.017992138862609863 | train/entropy_loss = -0.6405903100967407 | train/policy_loss = 0.9609434008598328 | train/value_loss = 5.029736042022705 | time/iterations = 500 | rollout/ep_rew_mean = 39.903225806451616 | rollout/ep_len_mean = 39.903225806451616 | time/fps = 64 | time/time_elapsed = 38 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.017992138862609863 | train/entropy_loss = -0.6405903100967407 | train/policy_loss = 0.9609434008598328 | train/value_loss = 5.029736042022705 | time/iterations = 500 | rollout/ep_rew_mean = 39.903225806451616 | rollout/ep_len_mean = 39.903225806451616 | time/fps = 64 | time/time_elapsed = 38 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 2]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0017293691635131836 | train/entropy_loss = -0.6757768392562866 | train/policy_loss = -9.245651245117188 | train/value_loss = 299.4968566894531 | time/iterations = 600 | rollout/ep_rew_mean = 24.87 | rollout/ep_len_mean = 24.87 | time/fps = 68 | time/time_elapsed = 43 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0017293691635131836 | train/entropy_loss = -0.6757768392562866 | train/policy_loss = -9.245651245117188 | train/value_loss = 299.4968566894531 | time/iterations = 600 | rollout/ep_rew_mean = 24.87 | rollout/ep_len_mean = 24.87 | time/fps = 68 | time/time_elapsed = 43 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 0]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.009956121444702148 | train/entropy_loss = -0.45464205741882324 | train/policy_loss = 0.4330309331417084 | train/value_loss = 4.5164289474487305 | time/iterations = 600 | rollout/ep_rew_mean = 53.96296296296296 | rollout/ep_len_mean = 53.96296296296296 | time/fps = 67 | time/time_elapsed = 44 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.009956121444702148 | train/entropy_loss = -0.45464205741882324 | train/policy_loss = 0.4330309331417084 | train/value_loss = 4.5164289474487305 | time/iterations = 600 | rollout/ep_rew_mean = 53.96296296296296 | rollout/ep_len_mean = 53.96296296296296 | time/fps = 67 | time/time_elapsed = 44 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 1]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.004995226860046387 | train/entropy_loss = -0.5594584941864014 | train/policy_loss = 0.8933206796646118 | train/value_loss = 4.778811931610107 | time/iterations = 600 | rollout/ep_rew_mean = 39.70666666666666 | rollout/ep_len_mean = 39.70666666666666 | time/fps = 67 | time/time_elapsed = 44 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.004995226860046387 | train/entropy_loss = -0.5594584941864014 | train/policy_loss = 0.8933206796646118 | train/value_loss = 4.778811931610107 | time/iterations = 600 | rollout/ep_rew_mean = 39.70666666666666 | rollout/ep_len_mean = 39.70666666666666 | time/fps = 67 | time/time_elapsed = 44 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 5]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.004312396049499512 | train/entropy_loss = -0.5958601832389832 | train/policy_loss = 0.65104079246521 | train/value_loss = 4.5853095054626465 | time/iterations = 600 | rollout/ep_rew_mean = 44.298507462686565 | rollout/ep_len_mean = 44.298507462686565 | time/fps = 67 | time/time_elapsed = 44 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.004312396049499512 | train/entropy_loss = -0.5958601832389832 | train/policy_loss = 0.65104079246521 | train/value_loss = 4.5853095054626465 | time/iterations = 600 | rollout/ep_rew_mean = 44.298507462686565 | rollout/ep_len_mean = 44.298507462686565 | time/fps = 67 | time/time_elapsed = 44 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 3]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.01231539249420166 | train/entropy_loss = -0.6158283948898315 | train/policy_loss = 1.280914545059204 | train/value_loss = 5.312915802001953 | time/iterations = 600 | rollout/ep_rew_mean = 25.67 | rollout/ep_len_mean = 25.67 | time/fps = 66 | time/time_elapsed = 45 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.01231539249420166 | train/entropy_loss = -0.6158283948898315 | train/policy_loss = 1.280914545059204 | train/value_loss = 5.312915802001953 | time/iterations = 600 | rollout/ep_rew_mean = 25.67 | rollout/ep_len_mean = 25.67 | time/fps = 66 | time/time_elapsed = 45 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 4]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.0011767745018005371 | train/entropy_loss = -0.6301224827766418 | train/policy_loss = 1.1250263452529907 | train/value_loss = 4.637922286987305 | time/iterations = 600 | rollout/ep_rew_mean = 43.44117647058823 | rollout/ep_len_mean = 43.44117647058823 | time/fps = 64 | time/time_elapsed = 46 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.0011767745018005371 | train/entropy_loss = -0.6301224827766418 | train/policy_loss = 1.1250263452529907 | train/value_loss = 4.637922286987305 | time/iterations = 600 | rollout/ep_rew_mean = 43.44117647058823 | rollout/ep_len_mean = 43.44117647058823 | time/fps = 64 | time/time_elapsed = 46 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 2]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.04061317443847656 | train/entropy_loss = -0.6422535181045532 | train/policy_loss = 0.807817816734314 | train/value_loss = 5.01497745513916 | time/iterations = 700 | rollout/ep_rew_mean = 26.58 | rollout/ep_len_mean = 26.58 | time/fps = 68 | time/time_elapsed = 50 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.04061317443847656 | train/entropy_loss = -0.6422535181045532 | train/policy_loss = 0.807817816734314 | train/value_loss = 5.01497745513916 | time/iterations = 700 | rollout/ep_rew_mean = 26.58 | rollout/ep_len_mean = 26.58 | time/fps = 68 | time/time_elapsed = 50 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 0]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.004462122917175293 | train/entropy_loss = -0.3419960141181946 | train/policy_loss = 1.5238847732543945 | train/value_loss = 3.983145236968994 | time/iterations = 700 | rollout/ep_rew_mean = 57.96666666666667 | rollout/ep_len_mean = 57.96666666666667 | time/fps = 68 | time/time_elapsed = 51 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.004462122917175293 | train/entropy_loss = -0.3419960141181946 | train/policy_loss = 1.5238847732543945 | train/value_loss = 3.983145236968994 | time/iterations = 700 | rollout/ep_rew_mean = 57.96666666666667 | rollout/ep_len_mean = 57.96666666666667 | time/fps = 68 | time/time_elapsed = 51 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 1]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.0026279687881469727 | train/entropy_loss = -0.5771624445915222 | train/policy_loss = 0.6436672806739807 | train/value_loss = 4.39661979675293 | time/iterations = 700 | rollout/ep_rew_mean = 42.024096385542165 | rollout/ep_len_mean = 42.024096385542165 | time/fps = 67 | time/time_elapsed = 51 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.0026279687881469727 | train/entropy_loss = -0.5771624445915222 | train/policy_loss = 0.6436672806739807 | train/value_loss = 4.39661979675293 | time/iterations = 700 | rollout/ep_rew_mean = 42.024096385542165 | rollout/ep_len_mean = 42.024096385542165 | time/fps = 67 | time/time_elapsed = 51 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 5]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.0034325122833251953 | train/entropy_loss = -0.4122610092163086 | train/policy_loss = 0.6861094236373901 | train/value_loss = 4.001700401306152 | time/iterations = 700 | rollout/ep_rew_mean = 49.214285714285715 | rollout/ep_len_mean = 49.214285714285715 | time/fps = 67 | time/time_elapsed = 51 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.0034325122833251953 | train/entropy_loss = -0.4122610092163086 | train/policy_loss = 0.6861094236373901 | train/value_loss = 4.001700401306152 | time/iterations = 700 | rollout/ep_rew_mean = 49.214285714285715 | rollout/ep_len_mean = 49.214285714285715 | time/fps = 67 | time/time_elapsed = 51 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 3]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.0011927485466003418 | train/entropy_loss = -0.637627124786377 | train/policy_loss = -13.983449935913086 | train/value_loss = 762.1309204101562 | time/iterations = 700 | rollout/ep_rew_mean = 28.81 | rollout/ep_len_mean = 28.81 | time/fps = 66 | time/time_elapsed = 52 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.0011927485466003418 | train/entropy_loss = -0.637627124786377 | train/policy_loss = -13.983449935913086 | train/value_loss = 762.1309204101562 | time/iterations = 700 | rollout/ep_rew_mean = 28.81 | rollout/ep_len_mean = 28.81 | time/fps = 66 | time/time_elapsed = 52 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 4]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.009778618812561035 | train/entropy_loss = -0.5846667885780334 | train/policy_loss = 1.3682843446731567 | train/value_loss = 4.1664557456970215 | time/iterations = 700 | rollout/ep_rew_mean = 47.57746478873239 | rollout/ep_len_mean = 47.57746478873239 | time/fps = 64 | time/time_elapsed = 54 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.009778618812561035 | train/entropy_loss = -0.5846667885780334 | train/policy_loss = 1.3682843446731567 | train/value_loss = 4.1664557456970215 | time/iterations = 700 | rollout/ep_rew_mean = 47.57746478873239 | rollout/ep_len_mean = 47.57746478873239 | time/fps = 64 | time/time_elapsed = 54 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 0]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.001515209674835205 | train/entropy_loss = -0.4833572804927826 | train/policy_loss = 0.44404298067092896 | train/value_loss = 3.415248155593872 | time/iterations = 800 | rollout/ep_rew_mean = 61.765625 | rollout/ep_len_mean = 61.765625 | time/fps = 68 | time/time_elapsed = 58 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.001515209674835205 | train/entropy_loss = -0.4833572804927826 | train/policy_loss = 0.44404298067092896 | train/value_loss = 3.415248155593872 | time/iterations = 800 | rollout/ep_rew_mean = 61.765625 | rollout/ep_len_mean = 61.765625 | time/fps = 68 | time/time_elapsed = 58 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 2]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.008747398853302002 | train/entropy_loss = -0.653870701789856 | train/policy_loss = 1.1805390119552612 | train/value_loss = 4.307283401489258 | time/iterations = 800 | rollout/ep_rew_mean = 27.93 | rollout/ep_len_mean = 27.93 | time/fps = 68 | time/time_elapsed = 58 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.008747398853302002 | train/entropy_loss = -0.653870701789856 | train/policy_loss = 1.1805390119552612 | train/value_loss = 4.307283401489258 | time/iterations = 800 | rollout/ep_rew_mean = 27.93 | rollout/ep_len_mean = 27.93 | time/fps = 68 | time/time_elapsed = 58 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 1]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0025467872619628906 | train/entropy_loss = -0.5890483856201172 | train/policy_loss = 0.8449739217758179 | train/value_loss = 3.893800735473633 | time/iterations = 800 | rollout/ep_rew_mean = 42.81720430107527 | rollout/ep_len_mean = 42.81720430107527 | time/fps = 68 | time/time_elapsed = 58 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0025467872619628906 | train/entropy_loss = -0.5890483856201172 | train/policy_loss = 0.8449739217758179 | train/value_loss = 3.893800735473633 | time/iterations = 800 | rollout/ep_rew_mean = 42.81720430107527 | rollout/ep_len_mean = 42.81720430107527 | time/fps = 68 | time/time_elapsed = 58 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 5]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.00012814998626708984 | train/entropy_loss = -0.662704348564148 | train/policy_loss = 0.8989217877388 | train/value_loss = 3.424931049346924 | time/iterations = 800 | rollout/ep_rew_mean = 52.605633802816904 | rollout/ep_len_mean = 52.605633802816904 | time/fps = 67 | time/time_elapsed = 58 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.00012814998626708984 | train/entropy_loss = -0.662704348564148 | train/policy_loss = 0.8989217877388 | train/value_loss = 3.424931049346924 | time/iterations = 800 | rollout/ep_rew_mean = 52.605633802816904 | rollout/ep_len_mean = 52.605633802816904 | time/fps = 67 | time/time_elapsed = 58 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 3]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0008755922317504883 | train/entropy_loss = -0.5480612516403198 | train/policy_loss = 0.7346829175949097 | train/value_loss = 4.210129261016846 | time/iterations = 800 | rollout/ep_rew_mean = 32.31 | rollout/ep_len_mean = 32.31 | time/fps = 67 | time/time_elapsed = 59 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0008755922317504883 | train/entropy_loss = -0.5480612516403198 | train/policy_loss = 0.7346829175949097 | train/value_loss = 4.210129261016846 | time/iterations = 800 | rollout/ep_rew_mean = 32.31 | rollout/ep_len_mean = 32.31 | time/fps = 67 | time/time_elapsed = 59 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 4]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.0003160834312438965 | train/entropy_loss = -0.5749378800392151 | train/policy_loss = 1.0486208200454712 | train/value_loss = 3.5246429443359375 | time/iterations = 800 | rollout/ep_rew_mean = 51.12820512820513 | rollout/ep_len_mean = 51.12820512820513 | time/fps = 64 | time/time_elapsed = 61 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.0003160834312438965 | train/entropy_loss = -0.5749378800392151 | train/policy_loss = 1.0486208200454712 | train/value_loss = 3.5246429443359375 | time/iterations = 800 | rollout/ep_rew_mean = 51.12820512820513 | rollout/ep_len_mean = 51.12820512820513 | time/fps = 64 | time/time_elapsed = 61 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 0]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0004139542579650879 | train/entropy_loss = -0.4873366355895996 | train/policy_loss = 0.5109798312187195 | train/value_loss = 2.9177780151367188 | time/iterations = 900 | rollout/ep_rew_mean = 66.01515151515152 | rollout/ep_len_mean = 66.01515151515152 | time/fps = 68 | time/time_elapsed = 65 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0004139542579650879 | train/entropy_loss = -0.4873366355895996 | train/policy_loss = 0.5109798312187195 | train/value_loss = 2.9177780151367188 | time/iterations = 900 | rollout/ep_rew_mean = 66.01515151515152 | rollout/ep_len_mean = 66.01515151515152 | time/fps = 68 | time/time_elapsed = 65 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 2]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 7.927417755126953e-05 | train/entropy_loss = -0.6721600294113159 | train/policy_loss = 1.0471904277801514 | train/value_loss = 3.822793483734131 | time/iterations = 900 | rollout/ep_rew_mean = 31.05 | rollout/ep_len_mean = 31.05 | time/fps = 68 | time/time_elapsed = 65 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 7.927417755126953e-05 | train/entropy_loss = -0.6721600294113159 | train/policy_loss = 1.0471904277801514 | train/value_loss = 3.822793483734131 | time/iterations = 900 | rollout/ep_rew_mean = 31.05 | rollout/ep_len_mean = 31.05 | time/fps = 68 | time/time_elapsed = 65 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 1]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.007180690765380859 | train/entropy_loss = -0.4910888671875 | train/policy_loss = 0.6905101537704468 | train/value_loss = 3.4290027618408203 | time/iterations = 900 | rollout/ep_rew_mean = 44.02 | rollout/ep_len_mean = 44.02 | time/fps = 67 | time/time_elapsed = 66 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.007180690765380859 | train/entropy_loss = -0.4910888671875 | train/policy_loss = 0.6905101537704468 | train/value_loss = 3.4290027618408203 | time/iterations = 900 | rollout/ep_rew_mean = 44.02 | rollout/ep_len_mean = 44.02 | time/fps = 67 | time/time_elapsed = 66 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 5]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0003559589385986328 | train/entropy_loss = -0.6319179534912109 | train/policy_loss = 0.8419055938720703 | train/value_loss = 2.928614854812622 | time/iterations = 900 | rollout/ep_rew_mean = 57.567567567567565 | rollout/ep_len_mean = 57.567567567567565 | time/fps = 67 | time/time_elapsed = 66 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0003559589385986328 | train/entropy_loss = -0.6319179534912109 | train/policy_loss = 0.8419055938720703 | train/value_loss = 2.928614854812622 | time/iterations = 900 | rollout/ep_rew_mean = 57.567567567567565 | rollout/ep_len_mean = 57.567567567567565 | time/fps = 67 | time/time_elapsed = 66 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 3]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0006763935089111328 | train/entropy_loss = -0.5977659225463867 | train/policy_loss = 0.6310566663742065 | train/value_loss = 3.6793556213378906 | time/iterations = 900 | rollout/ep_rew_mean = 34.79 | rollout/ep_len_mean = 34.79 | time/fps = 66 | time/time_elapsed = 67 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0006763935089111328 | train/entropy_loss = -0.5977659225463867 | train/policy_loss = 0.6310566663742065 | train/value_loss = 3.6793556213378906 | time/iterations = 900 | rollout/ep_rew_mean = 34.79 | rollout/ep_len_mean = 34.79 | time/fps = 66 | time/time_elapsed = 67 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 4]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 7.742643356323242e-05 | train/entropy_loss = -0.5841665267944336 | train/policy_loss = 0.5374836921691895 | train/value_loss = 3.0442893505096436 | time/iterations = 900 | rollout/ep_rew_mean = 53.63855421686747 | rollout/ep_len_mean = 53.63855421686747 | time/fps = 64 | time/time_elapsed = 70 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 7.742643356323242e-05 | train/entropy_loss = -0.5841665267944336 | train/policy_loss = 0.5374836921691895 | train/value_loss = 3.0442893505096436 | time/iterations = 900 | rollout/ep_rew_mean = 53.63855421686747 | rollout/ep_len_mean = 53.63855421686747 | time/fps = 64 | time/time_elapsed = 70 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 2]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0011212825775146484 | train/entropy_loss = -0.6825903654098511 | train/policy_loss = 1.1762949228286743 | train/value_loss = 3.394505739212036 | time/iterations = 1000 | rollout/ep_rew_mean = 33.72 | rollout/ep_len_mean = 33.72 | time/fps = 68 | time/time_elapsed = 73 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0011212825775146484 | train/entropy_loss = -0.6825903654098511 | train/policy_loss = 1.1762949228286743 | train/value_loss = 3.394505739212036 | time/iterations = 1000 | rollout/ep_rew_mean = 33.72 | rollout/ep_len_mean = 33.72 | time/fps = 68 | time/time_elapsed = 73 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 0]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 2.3365020751953125e-05 | train/entropy_loss = -0.554356575012207 | train/policy_loss = 0.5222793221473694 | train/value_loss = 2.462502956390381 | time/iterations = 1000 | rollout/ep_rew_mean = 72.23188405797102 | rollout/ep_len_mean = 72.23188405797102 | time/fps = 68 | time/time_elapsed = 73 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 2.3365020751953125e-05 | train/entropy_loss = -0.554356575012207 | train/policy_loss = 0.5222793221473694 | train/value_loss = 2.462502956390381 | time/iterations = 1000 | rollout/ep_rew_mean = 72.23188405797102 | rollout/ep_len_mean = 72.23188405797102 | time/fps = 68 | time/time_elapsed = 73 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 5]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.00018936395645141602 | train/entropy_loss = -0.5015659332275391 | train/policy_loss = 0.5901550650596619 | train/value_loss = 2.4704835414886475 | time/iterations = 1000 | rollout/ep_rew_mean = 64.51948051948052 | rollout/ep_len_mean = 64.51948051948052 | time/fps = 67 | time/time_elapsed = 74 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.00018936395645141602 | train/entropy_loss = -0.5015659332275391 | train/policy_loss = 0.5901550650596619 | train/value_loss = 2.4704835414886475 | time/iterations = 1000 | rollout/ep_rew_mean = 64.51948051948052 | rollout/ep_len_mean = 64.51948051948052 | time/fps = 67 | time/time_elapsed = 74 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 1]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 1.8894672393798828e-05 | train/entropy_loss = -0.5667680501937866 | train/policy_loss = 0.8399535417556763 | train/value_loss = 2.9642319679260254 | time/iterations = 1000 | rollout/ep_rew_mean = 46.77 | rollout/ep_len_mean = 46.77 | time/fps = 67 | time/time_elapsed = 74 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 1.8894672393798828e-05 | train/entropy_loss = -0.5667680501937866 | train/policy_loss = 0.8399535417556763 | train/value_loss = 2.9642319679260254 | time/iterations = 1000 | rollout/ep_rew_mean = 46.77 | rollout/ep_len_mean = 46.77 | time/fps = 67 | time/time_elapsed = 74 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 3]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -8.928775787353516e-05 | train/entropy_loss = -0.4371188282966614 | train/policy_loss = -23.172534942626953 | train/value_loss = 1790.7884521484375 | time/iterations = 1000 | rollout/ep_rew_mean = 40.04 | rollout/ep_len_mean = 40.04 | time/fps = 66 | time/time_elapsed = 75 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -8.928775787353516e-05 | train/entropy_loss = -0.4371188282966614 | train/policy_loss = -23.172534942626953 | train/value_loss = 1790.7884521484375 | time/iterations = 1000 | rollout/ep_rew_mean = 40.04 | rollout/ep_len_mean = 40.04 | time/fps = 66 | time/time_elapsed = 75 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 4]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -8.225440979003906e-06 | train/entropy_loss = -0.476095974445343 | train/policy_loss = -29.084692001342773 | train/value_loss = 1837.828125 | time/iterations = 1000 | rollout/ep_rew_mean = 54.28260869565217 | rollout/ep_len_mean = 54.28260869565217 | time/fps = 63 | time/time_elapsed = 78 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -8.225440979003906e-06 | train/entropy_loss = -0.476095974445343 | train/policy_loss = -29.084692001342773 | train/value_loss = 1837.828125 | time/iterations = 1000 | rollout/ep_rew_mean = 54.28260869565217 | rollout/ep_len_mean = 54.28260869565217 | time/fps = 63 | time/time_elapsed = 78 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 2]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.003865480422973633 | train/entropy_loss = -0.6579142808914185 | train/policy_loss = 0.7392672300338745 | train/value_loss = 2.969383716583252 | time/iterations = 1100 | rollout/ep_rew_mean = 36.57 | rollout/ep_len_mean = 36.57 | time/fps = 67 | time/time_elapsed = 81 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.003865480422973633 | train/entropy_loss = -0.6579142808914185 | train/policy_loss = 0.7392672300338745 | train/value_loss = 2.969383716583252 | time/iterations = 1100 | rollout/ep_rew_mean = 36.57 | rollout/ep_len_mean = 36.57 | time/fps = 67 | time/time_elapsed = 81 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 0]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 7.039308547973633e-05 | train/entropy_loss = -0.49048203229904175 | train/policy_loss = 0.27631622552871704 | train/value_loss = 2.034088134765625 | time/iterations = 1100 | rollout/ep_rew_mean = 76.67605633802818 | rollout/ep_len_mean = 76.67605633802818 | time/fps = 67 | time/time_elapsed = 81 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 7.039308547973633e-05 | train/entropy_loss = -0.49048203229904175 | train/policy_loss = 0.27631622552871704 | train/value_loss = 2.034088134765625 | time/iterations = 1100 | rollout/ep_rew_mean = 76.67605633802818 | rollout/ep_len_mean = 76.67605633802818 | time/fps = 67 | time/time_elapsed = 81 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 1]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -3.0994415283203125e-05 | train/entropy_loss = -0.39993545413017273 | train/policy_loss = -16.06704330444336 | train/value_loss = 1474.594970703125 | time/iterations = 1100 | rollout/ep_rew_mean = 51.07 | rollout/ep_len_mean = 51.07 | time/fps = 66 | time/time_elapsed = 82 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -3.0994415283203125e-05 | train/entropy_loss = -0.39993545413017273 | train/policy_loss = -16.06704330444336 | train/value_loss = 1474.594970703125 | time/iterations = 1100 | rollout/ep_rew_mean = 51.07 | rollout/ep_len_mean = 51.07 | time/fps = 66 | time/time_elapsed = 82 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 5]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 3.1948089599609375e-05 | train/entropy_loss = -0.5845779776573181 | train/policy_loss = 0.4587177336215973 | train/value_loss = 2.050629138946533 | time/iterations = 1100 | rollout/ep_rew_mean = 67.9625 | rollout/ep_len_mean = 67.9625 | time/fps = 66 | time/time_elapsed = 82 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 3.1948089599609375e-05 | train/entropy_loss = -0.5845779776573181 | train/policy_loss = 0.4587177336215973 | train/value_loss = 2.050629138946533 | time/iterations = 1100 | rollout/ep_rew_mean = 67.9625 | rollout/ep_len_mean = 67.9625 | time/fps = 66 | time/time_elapsed = 82 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 3]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.002168416976928711 | train/entropy_loss = -0.45545658469200134 | train/policy_loss = 1.5087124109268188 | train/value_loss = 2.739410638809204 | time/iterations = 1100 | rollout/ep_rew_mean = 43.29 | rollout/ep_len_mean = 43.29 | time/fps = 66 | time/time_elapsed = 83 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.002168416976928711 | train/entropy_loss = -0.45545658469200134 | train/policy_loss = 1.5087124109268188 | train/value_loss = 2.739410638809204 | time/iterations = 1100 | rollout/ep_rew_mean = 43.29 | rollout/ep_len_mean = 43.29 | time/fps = 66 | time/time_elapsed = 83 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:39: [Sb3-A2C[worker: 4]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0017911791801452637 | train/entropy_loss = -0.6262950301170349 | train/policy_loss = 0.6619774699211121 | train/value_loss = 2.246936321258545 | time/iterations = 1100 | rollout/ep_rew_mean = 53.75 | rollout/ep_len_mean = 53.75 | time/fps = 63 | time/time_elapsed = 86 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0017911791801452637 | train/entropy_loss = -0.6262950301170349 | train/policy_loss = 0.6619774699211121 | train/value_loss = 2.246936321258545 | time/iterations = 1100 | rollout/ep_rew_mean = 53.75 | rollout/ep_len_mean = 53.75 | time/fps = 63 | time/time_elapsed = 86 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 0]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -2.6702880859375e-05 | train/entropy_loss = -0.48699846863746643 | train/policy_loss = 0.6468709707260132 | train/value_loss = 1.654481291770935 | time/iterations = 1200 | rollout/ep_rew_mean = 80.94594594594595 | rollout/ep_len_mean = 80.94594594594595 | time/fps = 67 | time/time_elapsed = 89 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -2.6702880859375e-05 | train/entropy_loss = -0.48699846863746643 | train/policy_loss = 0.6468709707260132 | train/value_loss = 1.654481291770935 | time/iterations = 1200 | rollout/ep_rew_mean = 80.94594594594595 | rollout/ep_len_mean = 80.94594594594595 | time/fps = 67 | time/time_elapsed = 89 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 2]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.00011426210403442383 | train/entropy_loss = -0.646999716758728 | train/policy_loss = 0.9169305562973022 | train/value_loss = 2.554244041442871 | time/iterations = 1200 | rollout/ep_rew_mean = 39.6 | rollout/ep_len_mean = 39.6 | time/fps = 67 | time/time_elapsed = 89 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.00011426210403442383 | train/entropy_loss = -0.646999716758728 | train/policy_loss = 0.9169305562973022 | train/value_loss = 2.554244041442871 | time/iterations = 1200 | rollout/ep_rew_mean = 39.6 | rollout/ep_len_mean = 39.6 | time/fps = 67 | time/time_elapsed = 89 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 1]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00034868717193603516 | train/entropy_loss = -0.3801347017288208 | train/policy_loss = 0.9221994280815125 | train/value_loss = 2.0918474197387695 | time/iterations = 1200 | rollout/ep_rew_mean = 53.92 | rollout/ep_len_mean = 53.92 | time/fps = 66 | time/time_elapsed = 89 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00034868717193603516 | train/entropy_loss = -0.3801347017288208 | train/policy_loss = 0.9221994280815125 | train/value_loss = 2.0918474197387695 | time/iterations = 1200 | rollout/ep_rew_mean = 53.92 | rollout/ep_len_mean = 53.92 | time/fps = 66 | time/time_elapsed = 89 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 5]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 3.9517879486083984e-05 | train/entropy_loss = -0.4274832606315613 | train/policy_loss = 0.8704540133476257 | train/value_loss = 1.6697683334350586 | time/iterations = 1200 | rollout/ep_rew_mean = 71.72289156626506 | rollout/ep_len_mean = 71.72289156626506 | time/fps = 66 | time/time_elapsed = 90 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 3.9517879486083984e-05 | train/entropy_loss = -0.4274832606315613 | train/policy_loss = 0.8704540133476257 | train/value_loss = 1.6697683334350586 | time/iterations = 1200 | rollout/ep_rew_mean = 71.72289156626506 | rollout/ep_len_mean = 71.72289156626506 | time/fps = 66 | time/time_elapsed = 90 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 3]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.00013047456741333008 | train/entropy_loss = -0.4630466401576996 | train/policy_loss = 1.4156584739685059 | train/value_loss = 2.311117172241211 | time/iterations = 1200 | rollout/ep_rew_mean = 46.29 | rollout/ep_len_mean = 46.29 | time/fps = 66 | time/time_elapsed = 90 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.00013047456741333008 | train/entropy_loss = -0.4630466401576996 | train/policy_loss = 1.4156584739685059 | train/value_loss = 2.311117172241211 | time/iterations = 1200 | rollout/ep_rew_mean = 46.29 | rollout/ep_len_mean = 46.29 | time/fps = 66 | time/time_elapsed = 90 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 4]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.0005003213882446289 | train/entropy_loss = -0.5199692845344543 | train/policy_loss = 0.4090266823768616 | train/value_loss = 1.8674787282943726 | time/iterations = 1200 | rollout/ep_rew_mean = 58.45 | rollout/ep_len_mean = 58.45 | time/fps = 63 | time/time_elapsed = 94 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.0005003213882446289 | train/entropy_loss = -0.5199692845344543 | train/policy_loss = 0.4090266823768616 | train/value_loss = 1.8674787282943726 | time/iterations = 1200 | rollout/ep_rew_mean = 58.45 | rollout/ep_len_mean = 58.45 | time/fps = 63 | time/time_elapsed = 94 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 0]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -5.841255187988281e-06 | train/entropy_loss = -0.5114079117774963 | train/policy_loss = 0.4429708421230316 | train/value_loss = 1.3007886409759521 | time/iterations = 1300 | rollout/ep_rew_mean = 84.97333333333333 | rollout/ep_len_mean = 84.97333333333333 | time/fps = 67 | time/time_elapsed = 96 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -5.841255187988281e-06 | train/entropy_loss = -0.5114079117774963 | train/policy_loss = 0.4429708421230316 | train/value_loss = 1.3007886409759521 | time/iterations = 1300 | rollout/ep_rew_mean = 84.97333333333333 | rollout/ep_len_mean = 84.97333333333333 | time/fps = 67 | time/time_elapsed = 96 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 2]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.000665128231048584 | train/entropy_loss = -0.6706067323684692 | train/policy_loss = 0.8240481615066528 | train/value_loss = 2.1847362518310547 | time/iterations = 1300 | rollout/ep_rew_mean = 42.45 | rollout/ep_len_mean = 42.45 | time/fps = 67 | time/time_elapsed = 96 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.000665128231048584 | train/entropy_loss = -0.6706067323684692 | train/policy_loss = 0.8240481615066528 | train/value_loss = 2.1847362518310547 | time/iterations = 1300 | rollout/ep_rew_mean = 42.45 | rollout/ep_len_mean = 42.45 | time/fps = 67 | time/time_elapsed = 96 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 5]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 1.33514404296875e-05 | train/entropy_loss = -0.4819595217704773 | train/policy_loss = 0.38840028643608093 | train/value_loss = 1.3202183246612549 | time/iterations = 1300 | rollout/ep_rew_mean = 75.17647058823529 | rollout/ep_len_mean = 75.17647058823529 | time/fps = 66 | time/time_elapsed = 97 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 1.33514404296875e-05 | train/entropy_loss = -0.4819595217704773 | train/policy_loss = 0.38840028643608093 | train/value_loss = 1.3202183246612549 | time/iterations = 1300 | rollout/ep_rew_mean = 75.17647058823529 | rollout/ep_len_mean = 75.17647058823529 | time/fps = 66 | time/time_elapsed = 97 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 1]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -7.3909759521484375e-06 | train/entropy_loss = -0.5249566435813904 | train/policy_loss = 0.30444851517677307 | train/value_loss = 1.690023422241211 | time/iterations = 1300 | rollout/ep_rew_mean = 56.39 | rollout/ep_len_mean = 56.39 | time/fps = 66 | time/time_elapsed = 97 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -7.3909759521484375e-06 | train/entropy_loss = -0.5249566435813904 | train/policy_loss = 0.30444851517677307 | train/value_loss = 1.690023422241211 | time/iterations = 1300 | rollout/ep_rew_mean = 56.39 | rollout/ep_len_mean = 56.39 | time/fps = 66 | time/time_elapsed = 97 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 3]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.00018131732940673828 | train/entropy_loss = -0.6450713872909546 | train/policy_loss = 0.6046493053436279 | train/value_loss = 1.9229116439819336 | time/iterations = 1300 | rollout/ep_rew_mean = 50.96 | rollout/ep_len_mean = 50.96 | time/fps = 66 | time/time_elapsed = 97 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.00018131732940673828 | train/entropy_loss = -0.6450713872909546 | train/policy_loss = 0.6046493053436279 | train/value_loss = 1.9229116439819336 | time/iterations = 1300 | rollout/ep_rew_mean = 50.96 | rollout/ep_len_mean = 50.96 | time/fps = 66 | time/time_elapsed = 97 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 4]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 1.436471939086914e-05 | train/entropy_loss = -0.570276141166687 | train/policy_loss = 0.5083378553390503 | train/value_loss = 1.499513030052185 | time/iterations = 1300 | rollout/ep_rew_mean = 62.93 | rollout/ep_len_mean = 62.93 | time/fps = 63 | time/time_elapsed = 102 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 1.436471939086914e-05 | train/entropy_loss = -0.570276141166687 | train/policy_loss = 0.5083378553390503 | train/value_loss = 1.499513030052185 | time/iterations = 1300 | rollout/ep_rew_mean = 62.93 | rollout/ep_len_mean = 62.93 | time/fps = 63 | time/time_elapsed = 102 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 0]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.00040137767791748047 | train/entropy_loss = -0.406001478433609 | train/policy_loss = 0.3112906813621521 | train/value_loss = 0.9832593202590942 | time/iterations = 1400 | rollout/ep_rew_mean = 90.4342105263158 | rollout/ep_len_mean = 90.4342105263158 | time/fps = 67 | time/time_elapsed = 103 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.00040137767791748047 | train/entropy_loss = -0.406001478433609 | train/policy_loss = 0.3112906813621521 | train/value_loss = 0.9832593202590942 | time/iterations = 1400 | rollout/ep_rew_mean = 90.4342105263158 | rollout/ep_len_mean = 90.4342105263158 | time/fps = 67 | time/time_elapsed = 103 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 2]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.0024064183235168457 | train/entropy_loss = -0.6756181716918945 | train/policy_loss = 0.6367989778518677 | train/value_loss = 1.8146251440048218 | time/iterations = 1400 | rollout/ep_rew_mean = 45.68 | rollout/ep_len_mean = 45.68 | time/fps = 67 | time/time_elapsed = 103 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.0024064183235168457 | train/entropy_loss = -0.6756181716918945 | train/policy_loss = 0.6367989778518677 | train/value_loss = 1.8146251440048218 | time/iterations = 1400 | rollout/ep_rew_mean = 45.68 | rollout/ep_len_mean = 45.68 | time/fps = 67 | time/time_elapsed = 103 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 1]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 7.414817810058594e-05 | train/entropy_loss = -0.4900560975074768 | train/policy_loss = 0.2898872494697571 | train/value_loss = 1.346577763557434 | time/iterations = 1400 | rollout/ep_rew_mean = 62.44 | rollout/ep_len_mean = 62.44 | time/fps = 66 | time/time_elapsed = 104 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 7.414817810058594e-05 | train/entropy_loss = -0.4900560975074768 | train/policy_loss = 0.2898872494697571 | train/value_loss = 1.346577763557434 | time/iterations = 1400 | rollout/ep_rew_mean = 62.44 | rollout/ep_len_mean = 62.44 | time/fps = 66 | time/time_elapsed = 104 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 5]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 4.094839096069336e-05 | train/entropy_loss = -0.5060238838195801 | train/policy_loss = 0.3872297704219818 | train/value_loss = 1.0126216411590576 | time/iterations = 1400 | rollout/ep_rew_mean = 79.0 | rollout/ep_len_mean = 79.0 | time/fps = 66 | time/time_elapsed = 104 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 4.094839096069336e-05 | train/entropy_loss = -0.5060238838195801 | train/policy_loss = 0.3872297704219818 | train/value_loss = 1.0126216411590576 | time/iterations = 1400 | rollout/ep_rew_mean = 79.0 | rollout/ep_len_mean = 79.0 | time/fps = 66 | time/time_elapsed = 104 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 3]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 5.334615707397461e-05 | train/entropy_loss = -0.6004511117935181 | train/policy_loss = 0.5048932433128357 | train/value_loss = 1.576467752456665 | time/iterations = 1400 | rollout/ep_rew_mean = 55.69 | rollout/ep_len_mean = 55.69 | time/fps = 66 | time/time_elapsed = 105 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 5.334615707397461e-05 | train/entropy_loss = -0.6004511117935181 | train/policy_loss = 0.5048932433128357 | train/value_loss = 1.576467752456665 | time/iterations = 1400 | rollout/ep_rew_mean = 55.69 | rollout/ep_len_mean = 55.69 | time/fps = 66 | time/time_elapsed = 105 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 4]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.00019466876983642578 | train/entropy_loss = -0.39071124792099 | train/policy_loss = 1.198372721672058 | train/value_loss = 1.1705316305160522 | time/iterations = 1400 | rollout/ep_rew_mean = 64.93 | rollout/ep_len_mean = 64.93 | time/fps = 63 | time/time_elapsed = 109 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.00019466876983642578 | train/entropy_loss = -0.39071124792099 | train/policy_loss = 1.198372721672058 | train/value_loss = 1.1705316305160522 | time/iterations = 1400 | rollout/ep_rew_mean = 64.93 | rollout/ep_len_mean = 64.93 | time/fps = 63 | time/time_elapsed = 109 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 0]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0001443624496459961 | train/entropy_loss = -0.44979768991470337 | train/policy_loss = 0.3629647195339203 | train/value_loss = 0.7255966663360596 | time/iterations = 1500 | rollout/ep_rew_mean = 94.68354430379746 | rollout/ep_len_mean = 94.68354430379746 | time/fps = 67 | time/time_elapsed = 110 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0001443624496459961 | train/entropy_loss = -0.44979768991470337 | train/policy_loss = 0.3629647195339203 | train/value_loss = 0.7255966663360596 | time/iterations = 1500 | rollout/ep_rew_mean = 94.68354430379746 | rollout/ep_len_mean = 94.68354430379746 | time/fps = 67 | time/time_elapsed = 110 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 2]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.00021398067474365234 | train/entropy_loss = -0.6540223956108093 | train/policy_loss = 0.6754752397537231 | train/value_loss = 1.4870054721832275 | time/iterations = 1500 | rollout/ep_rew_mean = 49.47 | rollout/ep_len_mean = 49.47 | time/fps = 67 | time/time_elapsed = 110 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.00021398067474365234 | train/entropy_loss = -0.6540223956108093 | train/policy_loss = 0.6754752397537231 | train/value_loss = 1.4870054721832275 | time/iterations = 1500 | rollout/ep_rew_mean = 49.47 | rollout/ep_len_mean = 49.47 | time/fps = 67 | time/time_elapsed = 110 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 1]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.00013190507888793945 | train/entropy_loss = -0.3193588852882385 | train/policy_loss = 1.1181831359863281 | train/value_loss = 1.045384168624878 | time/iterations = 1500 | rollout/ep_rew_mean = 66.36 | rollout/ep_len_mean = 66.36 | time/fps = 67 | time/time_elapsed = 111 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.00013190507888793945 | train/entropy_loss = -0.3193588852882385 | train/policy_loss = 1.1181831359863281 | train/value_loss = 1.045384168624878 | time/iterations = 1500 | rollout/ep_rew_mean = 66.36 | rollout/ep_len_mean = 66.36 | time/fps = 67 | time/time_elapsed = 111 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 5]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.0001785755157470703 | train/entropy_loss = -0.5382242202758789 | train/policy_loss = 0.283080518245697 | train/value_loss = 0.740989089012146 | time/iterations = 1500 | rollout/ep_rew_mean = 82.13636363636364 | rollout/ep_len_mean = 82.13636363636364 | time/fps = 66 | time/time_elapsed = 112 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.0001785755157470703 | train/entropy_loss = -0.5382242202758789 | train/policy_loss = 0.283080518245697 | train/value_loss = 0.740989089012146 | time/iterations = 1500 | rollout/ep_rew_mean = 82.13636363636364 | rollout/ep_len_mean = 82.13636363636364 | time/fps = 66 | time/time_elapsed = 112 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 3]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 9.000301361083984e-06 | train/entropy_loss = -0.635063886642456 | train/policy_loss = 0.5013632774353027 | train/value_loss = 1.2514545917510986 | time/iterations = 1500 | rollout/ep_rew_mean = 59.3 | rollout/ep_len_mean = 59.3 | time/fps = 66 | time/time_elapsed = 112 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 9.000301361083984e-06 | train/entropy_loss = -0.635063886642456 | train/policy_loss = 0.5013632774353027 | train/value_loss = 1.2514545917510986 | time/iterations = 1500 | rollout/ep_rew_mean = 59.3 | rollout/ep_len_mean = 59.3 | time/fps = 66 | time/time_elapsed = 112 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 0]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 4.214048385620117e-05 | train/entropy_loss = -0.40551096200942993 | train/policy_loss = 0.31980475783348083 | train/value_loss = 0.5067461133003235 | time/iterations = 1600 | rollout/ep_rew_mean = 97.41463414634147 | rollout/ep_len_mean = 97.41463414634147 | time/fps = 68 | time/time_elapsed = 117 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 4.214048385620117e-05 | train/entropy_loss = -0.40551096200942993 | train/policy_loss = 0.31980475783348083 | train/value_loss = 0.5067461133003235 | time/iterations = 1600 | rollout/ep_rew_mean = 97.41463414634147 | rollout/ep_len_mean = 97.41463414634147 | time/fps = 68 | time/time_elapsed = 117 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 4]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.00018680095672607422 | train/entropy_loss = -0.4876507818698883 | train/policy_loss = 0.30482232570648193 | train/value_loss = 0.8702136278152466 | time/iterations = 1500 | rollout/ep_rew_mean = 69.75 | rollout/ep_len_mean = 69.75 | time/fps = 63 | time/time_elapsed = 117 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.00018680095672607422 | train/entropy_loss = -0.4876507818698883 | train/policy_loss = 0.30482232570648193 | train/value_loss = 0.8702136278152466 | time/iterations = 1500 | rollout/ep_rew_mean = 69.75 | rollout/ep_len_mean = 69.75 | time/fps = 63 | time/time_elapsed = 117 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 2]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -3.2067298889160156e-05 | train/entropy_loss = -0.683731198310852 | train/policy_loss = 0.5812050104141235 | train/value_loss = 1.1604154109954834 | time/iterations = 1600 | rollout/ep_rew_mean = 52.33 | rollout/ep_len_mean = 52.33 | time/fps = 68 | time/time_elapsed = 117 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -3.2067298889160156e-05 | train/entropy_loss = -0.683731198310852 | train/policy_loss = 0.5812050104141235 | train/value_loss = 1.1604154109954834 | time/iterations = 1600 | rollout/ep_rew_mean = 52.33 | rollout/ep_len_mean = 52.33 | time/fps = 68 | time/time_elapsed = 117 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 1]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.00016170740127563477 | train/entropy_loss = -0.4693725109100342 | train/policy_loss = 0.2079867422580719 | train/value_loss = 0.7884647250175476 | time/iterations = 1600 | rollout/ep_rew_mean = 69.36 | rollout/ep_len_mean = 69.36 | time/fps = 67 | time/time_elapsed = 118 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.00016170740127563477 | train/entropy_loss = -0.4693725109100342 | train/policy_loss = 0.2079867422580719 | train/value_loss = 0.7884647250175476 | time/iterations = 1600 | rollout/ep_rew_mean = 69.36 | rollout/ep_len_mean = 69.36 | time/fps = 67 | time/time_elapsed = 118 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 5]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 2.1338462829589844e-05 | train/entropy_loss = -0.5131866335868835 | train/policy_loss = 0.2927188277244568 | train/value_loss = 0.5115782618522644 | time/iterations = 1600 | rollout/ep_rew_mean = 88.31111111111112 | rollout/ep_len_mean = 88.31111111111112 | time/fps = 67 | time/time_elapsed = 119 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 2.1338462829589844e-05 | train/entropy_loss = -0.5131866335868835 | train/policy_loss = 0.2927188277244568 | train/value_loss = 0.5115782618522644 | time/iterations = 1600 | rollout/ep_rew_mean = 88.31111111111112 | rollout/ep_len_mean = 88.31111111111112 | time/fps = 67 | time/time_elapsed = 119 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 3]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -2.0503997802734375e-05 | train/entropy_loss = -0.5763643980026245 | train/policy_loss = 0.36284443736076355 | train/value_loss = 0.9643011093139648 | time/iterations = 1600 | rollout/ep_rew_mean = 63.91 | rollout/ep_len_mean = 63.91 | time/fps = 66 | time/time_elapsed = 119 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -2.0503997802734375e-05 | train/entropy_loss = -0.5763643980026245 | train/policy_loss = 0.36284443736076355 | train/value_loss = 0.9643011093139648 | time/iterations = 1600 | rollout/ep_rew_mean = 63.91 | rollout/ep_len_mean = 63.91 | time/fps = 66 | time/time_elapsed = 119 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 0]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -1.7881393432617188e-05 | train/entropy_loss = -0.3717404007911682 | train/policy_loss = 0.13068489730358124 | train/value_loss = 0.3302380442619324 | time/iterations = 1700 | rollout/ep_rew_mean = 97.9186046511628 | rollout/ep_len_mean = 97.9186046511628 | time/fps = 68 | time/time_elapsed = 124 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -1.7881393432617188e-05 | train/entropy_loss = -0.3717404007911682 | train/policy_loss = 0.13068489730358124 | train/value_loss = 0.3302380442619324 | time/iterations = 1700 | rollout/ep_rew_mean = 97.9186046511628 | rollout/ep_len_mean = 97.9186046511628 | time/fps = 68 | time/time_elapsed = 124 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 2]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 2.1696090698242188e-05 | train/entropy_loss = -0.6055792570114136 | train/policy_loss = 0.5443727970123291 | train/value_loss = 0.8782593011856079 | time/iterations = 1700 | rollout/ep_rew_mean = 57.29 | rollout/ep_len_mean = 57.29 | time/fps = 68 | time/time_elapsed = 124 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 2.1696090698242188e-05 | train/entropy_loss = -0.6055792570114136 | train/policy_loss = 0.5443727970123291 | train/value_loss = 0.8782593011856079 | time/iterations = 1700 | rollout/ep_rew_mean = 57.29 | rollout/ep_len_mean = 57.29 | time/fps = 68 | time/time_elapsed = 124 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 4]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0010666847229003906 | train/entropy_loss = -0.3762313425540924 | train/policy_loss = 0.3912211060523987 | train/value_loss = 0.6194579601287842 | time/iterations = 1600 | rollout/ep_rew_mean = 74.24 | rollout/ep_len_mean = 74.24 | time/fps = 64 | time/time_elapsed = 124 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0010666847229003906 | train/entropy_loss = -0.3762313425540924 | train/policy_loss = 0.3912211060523987 | train/value_loss = 0.6194579601287842 | time/iterations = 1600 | rollout/ep_rew_mean = 74.24 | rollout/ep_len_mean = 74.24 | time/fps = 64 | time/time_elapsed = 124 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 1]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 1.1920928955078125e-06 | train/entropy_loss = -0.27201372385025024 | train/policy_loss = 0.4847390651702881 | train/value_loss = 0.5658591389656067 | time/iterations = 1700 | rollout/ep_rew_mean = 73.6 | rollout/ep_len_mean = 73.6 | time/fps = 67 | time/time_elapsed = 125 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 1.1920928955078125e-06 | train/entropy_loss = -0.27201372385025024 | train/policy_loss = 0.4847390651702881 | train/value_loss = 0.5658591389656067 | time/iterations = 1700 | rollout/ep_rew_mean = 73.6 | rollout/ep_len_mean = 73.6 | time/fps = 67 | time/time_elapsed = 125 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 5]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 3.600120544433594e-05 | train/entropy_loss = -0.43315523862838745 | train/policy_loss = 0.21360301971435547 | train/value_loss = 0.32767051458358765 | time/iterations = 1700 | rollout/ep_rew_mean = 91.16304347826087 | rollout/ep_len_mean = 91.16304347826087 | time/fps = 67 | time/time_elapsed = 126 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 3.600120544433594e-05 | train/entropy_loss = -0.43315523862838745 | train/policy_loss = 0.21360301971435547 | train/value_loss = 0.32767051458358765 | time/iterations = 1700 | rollout/ep_rew_mean = 91.16304347826087 | rollout/ep_len_mean = 91.16304347826087 | time/fps = 67 | time/time_elapsed = 126 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 3]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -6.67572021484375e-05 | train/entropy_loss = -0.4283131957054138 | train/policy_loss = 0.6259448528289795 | train/value_loss = 0.7141826748847961 | time/iterations = 1700 | rollout/ep_rew_mean = 67.55 | rollout/ep_len_mean = 67.55 | time/fps = 67 | time/time_elapsed = 126 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -6.67572021484375e-05 | train/entropy_loss = -0.4283131957054138 | train/policy_loss = 0.6259448528289795 | train/value_loss = 0.7141826748847961 | time/iterations = 1700 | rollout/ep_rew_mean = 67.55 | rollout/ep_len_mean = 67.55 | time/fps = 67 | time/time_elapsed = 126 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 0]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -8.189678192138672e-05 | train/entropy_loss = -0.37357187271118164 | train/policy_loss = 0.17119437456130981 | train/value_loss = 0.19176273047924042 | time/iterations = 1800 | rollout/ep_rew_mean = 99.55555555555556 | rollout/ep_len_mean = 99.55555555555556 | time/fps = 68 | time/time_elapsed = 131 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -8.189678192138672e-05 | train/entropy_loss = -0.37357187271118164 | train/policy_loss = 0.17119437456130981 | train/value_loss = 0.19176273047924042 | time/iterations = 1800 | rollout/ep_rew_mean = 99.55555555555556 | rollout/ep_len_mean = 99.55555555555556 | time/fps = 68 | time/time_elapsed = 131 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 2]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 3.898143768310547e-05 | train/entropy_loss = -0.5119116306304932 | train/policy_loss = 0.5816401243209839 | train/value_loss = 0.6309575438499451 | time/iterations = 1800 | rollout/ep_rew_mean = 61.73 | rollout/ep_len_mean = 61.73 | time/fps = 68 | time/time_elapsed = 131 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 3.898143768310547e-05 | train/entropy_loss = -0.5119116306304932 | train/policy_loss = 0.5816401243209839 | train/value_loss = 0.6309575438499451 | time/iterations = 1800 | rollout/ep_rew_mean = 61.73 | rollout/ep_len_mean = 61.73 | time/fps = 68 | time/time_elapsed = 131 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 4]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 0.00019037723541259766 | train/entropy_loss = -0.4400525689125061 | train/policy_loss = 0.23170113563537598 | train/value_loss = 0.4154384136199951 | time/iterations = 1700 | rollout/ep_rew_mean = 81.44 | rollout/ep_len_mean = 81.44 | time/fps = 64 | time/time_elapsed = 132 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 0.00019037723541259766 | train/entropy_loss = -0.4400525689125061 | train/policy_loss = 0.23170113563537598 | train/value_loss = 0.4154384136199951 | time/iterations = 1700 | rollout/ep_rew_mean = 81.44 | rollout/ep_len_mean = 81.44 | time/fps = 64 | time/time_elapsed = 132 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 1]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -5.960464477539062e-07 | train/entropy_loss = -0.2922981381416321 | train/policy_loss = 0.05332003906369209 | train/value_loss = 0.3782663643360138 | time/iterations = 1800 | rollout/ep_rew_mean = 76.79 | rollout/ep_len_mean = 76.79 | time/fps = 67 | time/time_elapsed = 132 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -5.960464477539062e-07 | train/entropy_loss = -0.2922981381416321 | train/policy_loss = 0.05332003906369209 | train/value_loss = 0.3782663643360138 | time/iterations = 1800 | rollout/ep_rew_mean = 76.79 | rollout/ep_len_mean = 76.79 | time/fps = 67 | time/time_elapsed = 132 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 5]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 1.1444091796875e-05 | train/entropy_loss = -0.5097273588180542 | train/policy_loss = 0.2061690092086792 | train/value_loss = 0.18395781517028809 | time/iterations = 1800 | rollout/ep_rew_mean = 93.68085106382979 | rollout/ep_len_mean = 93.68085106382979 | time/fps = 67 | time/time_elapsed = 133 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 1.1444091796875e-05 | train/entropy_loss = -0.5097273588180542 | train/policy_loss = 0.2061690092086792 | train/value_loss = 0.18395781517028809 | time/iterations = 1800 | rollout/ep_rew_mean = 93.68085106382979 | rollout/ep_len_mean = 93.68085106382979 | time/fps = 67 | time/time_elapsed = 133 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 3]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 2.9146671295166016e-05 | train/entropy_loss = -0.5711411833763123 | train/policy_loss = 0.4059692919254303 | train/value_loss = 0.4975046217441559 | time/iterations = 1800 | rollout/ep_rew_mean = 71.86 | rollout/ep_len_mean = 71.86 | time/fps = 67 | time/time_elapsed = 133 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 2.9146671295166016e-05 | train/entropy_loss = -0.5711411833763123 | train/policy_loss = 0.4059692919254303 | train/value_loss = 0.4975046217441559 | time/iterations = 1800 | rollout/ep_rew_mean = 71.86 | rollout/ep_len_mean = 71.86 | time/fps = 67 | time/time_elapsed = 133 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 0]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 3.3974647521972656e-05 | train/entropy_loss = -0.3485861122608185 | train/policy_loss = 0.04477228969335556 | train/value_loss = 0.08724518120288849 | time/iterations = 1900 | rollout/ep_rew_mean = 102.29347826086956 | rollout/ep_len_mean = 102.29347826086956 | time/fps = 68 | time/time_elapsed = 138 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 0]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 3.3974647521972656e-05 | train/entropy_loss = -0.3485861122608185 | train/policy_loss = 0.04477228969335556 | train/value_loss = 0.08724518120288849 | time/iterations = 1900 | rollout/ep_rew_mean = 102.29347826086956 | rollout/ep_len_mean = 102.29347826086956 | time/fps = 68 | time/time_elapsed = 138 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 2]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 1.531839370727539e-05 | train/entropy_loss = -0.5296542048454285 | train/policy_loss = 0.27210894227027893 | train/value_loss = 0.42589372396469116 | time/iterations = 1900 | rollout/ep_rew_mean = 66.69 | rollout/ep_len_mean = 66.69 | time/fps = 68 | time/time_elapsed = 138 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 2]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 1.531839370727539e-05 | train/entropy_loss = -0.5296542048454285 | train/policy_loss = 0.27210894227027893 | train/value_loss = 0.42589372396469116 | time/iterations = 1900 | rollout/ep_rew_mean = 66.69 | rollout/ep_len_mean = 66.69 | time/fps = 68 | time/time_elapsed = 138 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 1]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.0001074075698852539 | train/entropy_loss = -0.3498842120170593 | train/policy_loss = 0.08007973432540894 | train/value_loss = 0.22523203492164612 | time/iterations = 1900 | rollout/ep_rew_mean = 80.44 | rollout/ep_len_mean = 80.44 | time/fps = 67 | time/time_elapsed = 139 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 1]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.0001074075698852539 | train/entropy_loss = -0.3498842120170593 | train/policy_loss = 0.08007973432540894 | train/value_loss = 0.22523203492164612 | time/iterations = 1900 | rollout/ep_rew_mean = 80.44 | rollout/ep_len_mean = 80.44 | time/fps = 67 | time/time_elapsed = 139 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 4]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -9.775161743164062e-06 | train/entropy_loss = -0.45993930101394653 | train/policy_loss = 0.10064633190631866 | train/value_loss = 0.25532856583595276 | time/iterations = 1800 | rollout/ep_rew_mean = 85.02 | rollout/ep_len_mean = 85.02 | time/fps = 64 | time/time_elapsed = 139 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -9.775161743164062e-06 | train/entropy_loss = -0.45993930101394653 | train/policy_loss = 0.10064633190631866 | train/value_loss = 0.25532856583595276 | time/iterations = 1800 | rollout/ep_rew_mean = 85.02 | rollout/ep_len_mean = 85.02 | time/fps = 64 | time/time_elapsed = 139 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 5]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 5.3048133850097656e-05 | train/entropy_loss = -0.5710594654083252 | train/policy_loss = 0.12612342834472656 | train/value_loss = 0.08341091871261597 | time/iterations = 1900 | rollout/ep_rew_mean = 97.42268041237114 | rollout/ep_len_mean = 97.42268041237114 | time/fps = 67 | time/time_elapsed = 140 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 5]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 5.3048133850097656e-05 | train/entropy_loss = -0.5710594654083252 | train/policy_loss = 0.12612342834472656 | train/value_loss = 0.08341091871261597 | time/iterations = 1900 | rollout/ep_rew_mean = 97.42268041237114 | rollout/ep_len_mean = 97.42268041237114 | time/fps = 67 | time/time_elapsed = 140 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:40: [Sb3-A2C[worker: 3]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -9.655952453613281e-06 | train/entropy_loss = -0.5530790090560913 | train/policy_loss = -39.35932159423828 | train/value_loss = 6342.2353515625 | time/iterations = 1900 | rollout/ep_rew_mean = 76.38 | rollout/ep_len_mean = 76.38 | time/fps = 67 | time/time_elapsed = 140 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 3]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -9.655952453613281e-06 | train/entropy_loss = -0.5530790090560913 | train/policy_loss = -39.35932159423828 | train/value_loss = 6342.2353515625 | time/iterations = 1900 | rollout/ep_rew_mean = 76.38 | rollout/ep_len_mean = 76.38 | time/fps = 67 | time/time_elapsed = 140 | time/total_timesteps = 9500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:40:                agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      6           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      6           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:41:                agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      7           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      7           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:41:                agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      8           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      8           0.001               500
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 4]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 2.777576446533203e-05 | train/entropy_loss = -0.4514874517917633 | train/policy_loss = 0.13972558081150055 | train/value_loss = 0.13162270188331604 | time/iterations = 1900 | rollout/ep_rew_mean = 87.91 | rollout/ep_len_mean = 87.91 | time/fps = 63 | time/time_elapsed = 148 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 4]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 2.777576446533203e-05 | train/entropy_loss = -0.4514874517917633 | train/policy_loss = 0.13972558081150055 | train/value_loss = 0.13162270188331604 | time/iterations = 1900 | rollout/ep_rew_mean = 87.91 | rollout/ep_len_mean = 87.91 | time/fps = 63 | time/time_elapsed = 148 | time/total_timesteps = 9500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:41:                agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      9           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                  Sb3-A2C      9           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 6]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -2.71675181388855 | train/entropy_loss = -0.6158754229545593 | train/policy_loss = 2.138756513595581 | train/value_loss = 18.840530395507812 | time/iterations = 100 | rollout/ep_rew_mean = 29.625 | rollout/ep_len_mean = 29.625 | time/fps = 56 | time/time_elapsed = 8 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -2.71675181388855 | train/entropy_loss = -0.6158754229545593 | train/policy_loss = 2.138756513595581 | train/value_loss = 18.840530395507812 | time/iterations = 100 | rollout/ep_rew_mean = 29.625 | rollout/ep_len_mean = 29.625 | time/fps = 56 | time/time_elapsed = 8 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 7]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.08270537853240967 | train/entropy_loss = -0.4707508981227875 | train/policy_loss = 1.633626937866211 | train/value_loss = 4.9868998527526855 | time/iterations = 100 | rollout/ep_rew_mean = 68.42857142857143 | rollout/ep_len_mean = 68.42857142857143 | time/fps = 62 | time/time_elapsed = 8 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.08270537853240967 | train/entropy_loss = -0.4707508981227875 | train/policy_loss = 1.633626937866211 | train/value_loss = 4.9868998527526855 | time/iterations = 100 | rollout/ep_rew_mean = 68.42857142857143 | rollout/ep_len_mean = 68.42857142857143 | time/fps = 62 | time/time_elapsed = 8 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 8]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.32480061054229736 | train/entropy_loss = -0.6904557943344116 | train/policy_loss = -2.418275833129883 | train/value_loss = 27.017505645751953 | time/iterations = 100 | rollout/ep_rew_mean = 22.40909090909091 | rollout/ep_len_mean = 22.40909090909091 | time/fps = 68 | time/time_elapsed = 7 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.32480061054229736 | train/entropy_loss = -0.6904557943344116 | train/policy_loss = -2.418275833129883 | train/value_loss = 27.017505645751953 | time/iterations = 100 | rollout/ep_rew_mean = 22.40909090909091 | rollout/ep_len_mean = 22.40909090909091 | time/fps = 68 | time/time_elapsed = 7 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 9]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.6140213906764984 | train/entropy_loss = -0.6469623446464539 | train/policy_loss = 0.4475821852684021 | train/value_loss = 1.1537326574325562 | time/iterations = 100 | rollout/ep_rew_mean = 38.25 | rollout/ep_len_mean = 38.25 | time/fps = 71 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.6140213906764984 | train/entropy_loss = -0.6469623446464539 | train/policy_loss = 0.4475821852684021 | train/value_loss = 1.1537326574325562 | time/iterations = 100 | rollout/ep_rew_mean = 38.25 | rollout/ep_len_mean = 38.25 | time/fps = 71 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 6]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.17767810821533203 | train/entropy_loss = -0.555406391620636 | train/policy_loss = -0.8751899600028992 | train/value_loss = 42.9910774230957 | time/iterations = 200 | rollout/ep_rew_mean = 20.645833333333332 | rollout/ep_len_mean = 20.645833333333332 | time/fps = 67 | time/time_elapsed = 14 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.17767810821533203 | train/entropy_loss = -0.555406391620636 | train/policy_loss = -0.8751899600028992 | train/value_loss = 42.9910774230957 | time/iterations = 200 | rollout/ep_rew_mean = 20.645833333333332 | rollout/ep_len_mean = 20.645833333333332 | time/fps = 67 | time/time_elapsed = 14 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 7]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.053591251373291016 | train/entropy_loss = -0.5700623393058777 | train/policy_loss = 1.0375254154205322 | train/value_loss = 5.342561721801758 | time/iterations = 200 | rollout/ep_rew_mean = 61.375 | rollout/ep_len_mean = 61.375 | time/fps = 72 | time/time_elapsed = 13 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.053591251373291016 | train/entropy_loss = -0.5700623393058777 | train/policy_loss = 1.0375254154205322 | train/value_loss = 5.342561721801758 | time/iterations = 200 | rollout/ep_rew_mean = 61.375 | rollout/ep_len_mean = 61.375 | time/fps = 72 | time/time_elapsed = 13 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 8]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.07512807846069336 | train/entropy_loss = -0.6822347640991211 | train/policy_loss = 1.5891327857971191 | train/value_loss = 8.7285795211792 | time/iterations = 200 | rollout/ep_rew_mean = 23.232558139534884 | rollout/ep_len_mean = 23.232558139534884 | time/fps = 78 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.07512807846069336 | train/entropy_loss = -0.6822347640991211 | train/policy_loss = 1.5891327857971191 | train/value_loss = 8.7285795211792 | time/iterations = 200 | rollout/ep_rew_mean = 23.232558139534884 | rollout/ep_len_mean = 23.232558139534884 | time/fps = 78 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 9]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.10931622982025146 | train/entropy_loss = -0.48297566175460815 | train/policy_loss = 1.47257399559021 | train/value_loss = 7.106955528259277 | time/iterations = 200 | rollout/ep_rew_mean = 36.888888888888886 | rollout/ep_len_mean = 36.888888888888886 | time/fps = 83 | time/time_elapsed = 11 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.10931622982025146 | train/entropy_loss = -0.48297566175460815 | train/policy_loss = 1.47257399559021 | train/value_loss = 7.106955528259277 | time/iterations = 200 | rollout/ep_rew_mean = 36.888888888888886 | rollout/ep_len_mean = 36.888888888888886 | time/fps = 83 | time/time_elapsed = 11 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 6]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.02092808485031128 | train/entropy_loss = -0.6707273721694946 | train/policy_loss = 1.397865653038025 | train/value_loss = 7.061285495758057 | time/iterations = 300 | rollout/ep_rew_mean = 18.5375 | rollout/ep_len_mean = 18.5375 | time/fps = 75 | time/time_elapsed = 19 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.02092808485031128 | train/entropy_loss = -0.6707273721694946 | train/policy_loss = 1.397865653038025 | train/value_loss = 7.061285495758057 | time/iterations = 300 | rollout/ep_rew_mean = 18.5375 | rollout/ep_len_mean = 18.5375 | time/fps = 75 | time/time_elapsed = 19 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 7]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.11130237579345703 | train/entropy_loss = -0.5758782625198364 | train/policy_loss = 0.9389469027519226 | train/value_loss = 6.164398193359375 | time/iterations = 300 | rollout/ep_rew_mean = 51.92857142857143 | rollout/ep_len_mean = 51.92857142857143 | time/fps = 78 | time/time_elapsed = 19 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.11130237579345703 | train/entropy_loss = -0.5758782625198364 | train/policy_loss = 0.9389469027519226 | train/value_loss = 6.164398193359375 | time/iterations = 300 | rollout/ep_rew_mean = 51.92857142857143 | rollout/ep_len_mean = 51.92857142857143 | time/fps = 78 | time/time_elapsed = 19 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 8]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.04214823246002197 | train/entropy_loss = -0.44407591223716736 | train/policy_loss = 2.661494731903076 | train/value_loss = 6.8763322830200195 | time/iterations = 300 | rollout/ep_rew_mean = 26.8 | rollout/ep_len_mean = 26.8 | time/fps = 83 | time/time_elapsed = 17 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.04214823246002197 | train/entropy_loss = -0.44407591223716736 | train/policy_loss = 2.661494731903076 | train/value_loss = 6.8763322830200195 | time/iterations = 300 | rollout/ep_rew_mean = 26.8 | rollout/ep_len_mean = 26.8 | time/fps = 83 | time/time_elapsed = 17 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 9]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.015047788619995117 | train/entropy_loss = -0.607729971408844 | train/policy_loss = 1.0239261388778687 | train/value_loss = 6.512820243835449 | time/iterations = 300 | rollout/ep_rew_mean = 37.15 | rollout/ep_len_mean = 37.15 | time/fps = 88 | time/time_elapsed = 17 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.015047788619995117 | train/entropy_loss = -0.607729971408844 | train/policy_loss = 1.0239261388778687 | train/value_loss = 6.512820243835449 | time/iterations = 300 | rollout/ep_rew_mean = 37.15 | rollout/ep_len_mean = 37.15 | time/fps = 88 | time/time_elapsed = 17 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 6]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.03501218557357788 | train/entropy_loss = -0.666698157787323 | train/policy_loss = 1.2882025241851807 | train/value_loss = 6.51578426361084 | time/iterations = 400 | rollout/ep_rew_mean = 17.5 | rollout/ep_len_mean = 17.5 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.03501218557357788 | train/entropy_loss = -0.666698157787323 | train/policy_loss = 1.2882025241851807 | train/value_loss = 6.51578426361084 | time/iterations = 400 | rollout/ep_rew_mean = 17.5 | rollout/ep_len_mean = 17.5 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 7]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.20154619216918945 | train/entropy_loss = -0.4624057710170746 | train/policy_loss = 0.5498118996620178 | train/value_loss = 4.5275044441223145 | time/iterations = 400 | rollout/ep_rew_mean = 51.23076923076923 | rollout/ep_len_mean = 51.23076923076923 | time/fps = 83 | time/time_elapsed = 24 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.20154619216918945 | train/entropy_loss = -0.4624057710170746 | train/policy_loss = 0.5498118996620178 | train/value_loss = 4.5275044441223145 | time/iterations = 400 | rollout/ep_rew_mean = 51.23076923076923 | rollout/ep_len_mean = 51.23076923076923 | time/fps = 83 | time/time_elapsed = 24 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 8]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.02004718780517578 | train/entropy_loss = -0.6341758370399475 | train/policy_loss = 1.7571347951889038 | train/value_loss = 6.295714378356934 | time/iterations = 400 | rollout/ep_rew_mean = 30.28787878787879 | rollout/ep_len_mean = 30.28787878787879 | time/fps = 87 | time/time_elapsed = 22 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.02004718780517578 | train/entropy_loss = -0.6341758370399475 | train/policy_loss = 1.7571347951889038 | train/value_loss = 6.295714378356934 | time/iterations = 400 | rollout/ep_rew_mean = 30.28787878787879 | rollout/ep_len_mean = 30.28787878787879 | time/fps = 87 | time/time_elapsed = 22 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 9]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.04991710186004639 | train/entropy_loss = -0.6398876905441284 | train/policy_loss = 1.207918643951416 | train/value_loss = 6.237287521362305 | time/iterations = 400 | rollout/ep_rew_mean = 36.648148148148145 | rollout/ep_len_mean = 36.648148148148145 | time/fps = 91 | time/time_elapsed = 21 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.04991710186004639 | train/entropy_loss = -0.6398876905441284 | train/policy_loss = 1.207918643951416 | train/value_loss = 6.237287521362305 | time/iterations = 400 | rollout/ep_rew_mean = 36.648148148148145 | rollout/ep_len_mean = 36.648148148148145 | time/fps = 91 | time/time_elapsed = 21 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 6]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.053789734840393066 | train/entropy_loss = -0.4820387363433838 | train/policy_loss = -5.565389156341553 | train/value_loss = 236.264404296875 | time/iterations = 500 | rollout/ep_rew_mean = 17.5 | rollout/ep_len_mean = 17.5 | time/fps = 83 | time/time_elapsed = 29 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.053789734840393066 | train/entropy_loss = -0.4820387363433838 | train/policy_loss = -5.565389156341553 | train/value_loss = 236.264404296875 | time/iterations = 500 | rollout/ep_rew_mean = 17.5 | rollout/ep_len_mean = 17.5 | time/fps = 83 | time/time_elapsed = 29 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 7]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.005694389343261719 | train/entropy_loss = -0.468455970287323 | train/policy_loss = 0.8615595698356628 | train/value_loss = 5.028215408325195 | time/iterations = 500 | rollout/ep_rew_mean = 53.77777777777778 | rollout/ep_len_mean = 53.77777777777778 | time/fps = 86 | time/time_elapsed = 28 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.005694389343261719 | train/entropy_loss = -0.468455970287323 | train/policy_loss = 0.8615595698356628 | train/value_loss = 5.028215408325195 | time/iterations = 500 | rollout/ep_rew_mean = 53.77777777777778 | rollout/ep_len_mean = 53.77777777777778 | time/fps = 86 | time/time_elapsed = 28 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 8]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.11713647842407227 | train/entropy_loss = -0.5442498326301575 | train/policy_loss = 0.6234768033027649 | train/value_loss = 5.992794513702393 | time/iterations = 500 | rollout/ep_rew_mean = 32.064935064935064 | rollout/ep_len_mean = 32.064935064935064 | time/fps = 90 | time/time_elapsed = 27 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.11713647842407227 | train/entropy_loss = -0.5442498326301575 | train/policy_loss = 0.6234768033027649 | train/value_loss = 5.992794513702393 | time/iterations = 500 | rollout/ep_rew_mean = 32.064935064935064 | rollout/ep_len_mean = 32.064935064935064 | time/fps = 90 | time/time_elapsed = 27 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 9]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.04225122928619385 | train/entropy_loss = -0.5498809814453125 | train/policy_loss = 0.8411962389945984 | train/value_loss = 5.283140659332275 | time/iterations = 500 | rollout/ep_rew_mean = 37.014925373134325 | rollout/ep_len_mean = 37.014925373134325 | time/fps = 92 | time/time_elapsed = 27 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.04225122928619385 | train/entropy_loss = -0.5498809814453125 | train/policy_loss = 0.8411962389945984 | train/value_loss = 5.283140659332275 | time/iterations = 500 | rollout/ep_rew_mean = 37.014925373134325 | rollout/ep_len_mean = 37.014925373134325 | time/fps = 92 | time/time_elapsed = 27 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 6]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.3308345079421997 | train/entropy_loss = -0.48477330803871155 | train/policy_loss = 1.7074702978134155 | train/value_loss = 5.769753456115723 | time/iterations = 600 | rollout/ep_rew_mean = 19.62 | rollout/ep_len_mean = 19.62 | time/fps = 84 | time/time_elapsed = 35 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.3308345079421997 | train/entropy_loss = -0.48477330803871155 | train/policy_loss = 1.7074702978134155 | train/value_loss = 5.769753456115723 | time/iterations = 600 | rollout/ep_rew_mean = 19.62 | rollout/ep_len_mean = 19.62 | time/fps = 84 | time/time_elapsed = 35 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 7]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.004166603088378906 | train/entropy_loss = -0.49889516830444336 | train/policy_loss = 0.43069711327552795 | train/value_loss = 4.43619966506958 | time/iterations = 600 | rollout/ep_rew_mean = 59.7 | rollout/ep_len_mean = 59.7 | time/fps = 86 | time/time_elapsed = 34 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.004166603088378906 | train/entropy_loss = -0.49889516830444336 | train/policy_loss = 0.43069711327552795 | train/value_loss = 4.43619966506958 | time/iterations = 600 | rollout/ep_rew_mean = 59.7 | rollout/ep_len_mean = 59.7 | time/fps = 86 | time/time_elapsed = 34 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 8]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.002001047134399414 | train/entropy_loss = -0.5621349215507507 | train/policy_loss = 1.666232705116272 | train/value_loss = 4.912238121032715 | time/iterations = 600 | rollout/ep_rew_mean = 34.036144578313255 | rollout/ep_len_mean = 34.036144578313255 | time/fps = 89 | time/time_elapsed = 33 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.002001047134399414 | train/entropy_loss = -0.5621349215507507 | train/policy_loss = 1.666232705116272 | train/value_loss = 4.912238121032715 | time/iterations = 600 | rollout/ep_rew_mean = 34.036144578313255 | rollout/ep_len_mean = 34.036144578313255 | time/fps = 89 | time/time_elapsed = 33 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 9]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.05419325828552246 | train/entropy_loss = -0.4803670048713684 | train/policy_loss = 0.39603370428085327 | train/value_loss = 4.836783409118652 | time/iterations = 600 | rollout/ep_rew_mean = 38.32051282051282 | rollout/ep_len_mean = 38.32051282051282 | time/fps = 90 | time/time_elapsed = 33 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.05419325828552246 | train/entropy_loss = -0.4803670048713684 | train/policy_loss = 0.39603370428085327 | train/value_loss = 4.836783409118652 | time/iterations = 600 | rollout/ep_rew_mean = 38.32051282051282 | rollout/ep_len_mean = 38.32051282051282 | time/fps = 90 | time/time_elapsed = 33 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 6]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.0006102323532104492 | train/entropy_loss = -0.6232984662055969 | train/policy_loss = 0.8614702224731445 | train/value_loss = 5.50843620300293 | time/iterations = 700 | rollout/ep_rew_mean = 22.03 | rollout/ep_len_mean = 22.03 | time/fps = 83 | time/time_elapsed = 42 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.0006102323532104492 | train/entropy_loss = -0.6232984662055969 | train/policy_loss = 0.8614702224731445 | train/value_loss = 5.50843620300293 | time/iterations = 700 | rollout/ep_rew_mean = 22.03 | rollout/ep_len_mean = 22.03 | time/fps = 83 | time/time_elapsed = 42 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 7]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.004096627235412598 | train/entropy_loss = -0.4337450861930847 | train/policy_loss = 0.42405691742897034 | train/value_loss = 3.8866965770721436 | time/iterations = 700 | rollout/ep_rew_mean = 62.03636363636364 | rollout/ep_len_mean = 62.03636363636364 | time/fps = 84 | time/time_elapsed = 41 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.004096627235412598 | train/entropy_loss = -0.4337450861930847 | train/policy_loss = 0.42405691742897034 | train/value_loss = 3.8866965770721436 | time/iterations = 700 | rollout/ep_rew_mean = 62.03636363636364 | rollout/ep_len_mean = 62.03636363636364 | time/fps = 84 | time/time_elapsed = 41 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 8]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.008156061172485352 | train/entropy_loss = -0.5573519468307495 | train/policy_loss = 0.6111973524093628 | train/value_loss = 4.364705562591553 | time/iterations = 700 | rollout/ep_rew_mean = 38.62921348314607 | rollout/ep_len_mean = 38.62921348314607 | time/fps = 87 | time/time_elapsed = 40 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.008156061172485352 | train/entropy_loss = -0.5573519468307495 | train/policy_loss = 0.6111973524093628 | train/value_loss = 4.364705562591553 | time/iterations = 700 | rollout/ep_rew_mean = 38.62921348314607 | rollout/ep_len_mean = 38.62921348314607 | time/fps = 87 | time/time_elapsed = 40 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 9]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.00363922119140625 | train/entropy_loss = -0.5214232206344604 | train/policy_loss = 0.5244942307472229 | train/value_loss = 4.293979167938232 | time/iterations = 700 | rollout/ep_rew_mean = 40.651162790697676 | rollout/ep_len_mean = 40.651162790697676 | time/fps = 88 | time/time_elapsed = 39 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.00363922119140625 | train/entropy_loss = -0.5214232206344604 | train/policy_loss = 0.5244942307472229 | train/value_loss = 4.293979167938232 | time/iterations = 700 | rollout/ep_rew_mean = 40.651162790697676 | rollout/ep_len_mean = 40.651162790697676 | time/fps = 88 | time/time_elapsed = 39 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 6]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.024046480655670166 | train/entropy_loss = -0.4965330958366394 | train/policy_loss = 1.076313853263855 | train/value_loss = 5.120299339294434 | time/iterations = 800 | rollout/ep_rew_mean = 25.47 | rollout/ep_len_mean = 25.47 | time/fps = 83 | time/time_elapsed = 48 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.024046480655670166 | train/entropy_loss = -0.4965330958366394 | train/policy_loss = 1.076313853263855 | train/value_loss = 5.120299339294434 | time/iterations = 800 | rollout/ep_rew_mean = 25.47 | rollout/ep_len_mean = 25.47 | time/fps = 83 | time/time_elapsed = 48 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 7]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.00973045825958252 | train/entropy_loss = -0.34203213453292847 | train/policy_loss = 1.0284227132797241 | train/value_loss = 3.3231494426727295 | time/iterations = 800 | rollout/ep_rew_mean = 66.56896551724138 | rollout/ep_len_mean = 66.56896551724138 | time/fps = 84 | time/time_elapsed = 47 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.00973045825958252 | train/entropy_loss = -0.34203213453292847 | train/policy_loss = 1.0284227132797241 | train/value_loss = 3.3231494426727295 | time/iterations = 800 | rollout/ep_rew_mean = 66.56896551724138 | rollout/ep_len_mean = 66.56896551724138 | time/fps = 84 | time/time_elapsed = 47 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 8]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.00015747547149658203 | train/entropy_loss = -0.47525152564048767 | train/policy_loss = 0.7872891426086426 | train/value_loss = 3.783045530319214 | time/iterations = 800 | rollout/ep_rew_mean = 41.62765957446808 | rollout/ep_len_mean = 41.62765957446808 | time/fps = 86 | time/time_elapsed = 46 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.00015747547149658203 | train/entropy_loss = -0.47525152564048767 | train/policy_loss = 0.7872891426086426 | train/value_loss = 3.783045530319214 | time/iterations = 800 | rollout/ep_rew_mean = 41.62765957446808 | rollout/ep_len_mean = 41.62765957446808 | time/fps = 86 | time/time_elapsed = 46 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 9]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.026750504970550537 | train/entropy_loss = -0.28369319438934326 | train/policy_loss = 1.9601703882217407 | train/value_loss = 3.7504372596740723 | time/iterations = 800 | rollout/ep_rew_mean = 42.376344086021504 | rollout/ep_len_mean = 42.376344086021504 | time/fps = 87 | time/time_elapsed = 45 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.026750504970550537 | train/entropy_loss = -0.28369319438934326 | train/policy_loss = 1.9601703882217407 | train/value_loss = 3.7504372596740723 | time/iterations = 800 | rollout/ep_rew_mean = 42.376344086021504 | rollout/ep_len_mean = 42.376344086021504 | time/fps = 87 | time/time_elapsed = 45 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 6]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.00037419795989990234 | train/entropy_loss = -0.19497562944889069 | train/policy_loss = 2.522071361541748 | train/value_loss = 4.5219316482543945 | time/iterations = 900 | rollout/ep_rew_mean = 28.38 | rollout/ep_len_mean = 28.38 | time/fps = 83 | time/time_elapsed = 54 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.00037419795989990234 | train/entropy_loss = -0.19497562944889069 | train/policy_loss = 2.522071361541748 | train/value_loss = 4.5219316482543945 | time/iterations = 900 | rollout/ep_rew_mean = 28.38 | rollout/ep_len_mean = 28.38 | time/fps = 83 | time/time_elapsed = 54 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 7]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0002410411834716797 | train/entropy_loss = -0.4730106294155121 | train/policy_loss = 0.47295433282852173 | train/value_loss = 2.848735809326172 | time/iterations = 900 | rollout/ep_rew_mean = 72.91803278688525 | rollout/ep_len_mean = 72.91803278688525 | time/fps = 84 | time/time_elapsed = 53 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0002410411834716797 | train/entropy_loss = -0.4730106294155121 | train/policy_loss = 0.47295433282852173 | train/value_loss = 2.848735809326172 | time/iterations = 900 | rollout/ep_rew_mean = 72.91803278688525 | rollout/ep_len_mean = 72.91803278688525 | time/fps = 84 | time/time_elapsed = 53 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 8]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -8.380413055419922e-05 | train/entropy_loss = -0.5312173366546631 | train/policy_loss = 0.8744239807128906 | train/value_loss = 3.28973388671875 | time/iterations = 900 | rollout/ep_rew_mean = 45.21212121212121 | rollout/ep_len_mean = 45.21212121212121 | time/fps = 86 | time/time_elapsed = 52 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -8.380413055419922e-05 | train/entropy_loss = -0.5312173366546631 | train/policy_loss = 0.8744239807128906 | train/value_loss = 3.28973388671875 | time/iterations = 900 | rollout/ep_rew_mean = 45.21212121212121 | rollout/ep_len_mean = 45.21212121212121 | time/fps = 86 | time/time_elapsed = 52 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 9]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.003941774368286133 | train/entropy_loss = -0.48425158858299255 | train/policy_loss = 0.729732096195221 | train/value_loss = 3.2272536754608154 | time/iterations = 900 | rollout/ep_rew_mean = 45.69387755102041 | rollout/ep_len_mean = 45.69387755102041 | time/fps = 87 | time/time_elapsed = 51 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.003941774368286133 | train/entropy_loss = -0.48425158858299255 | train/policy_loss = 0.729732096195221 | train/value_loss = 3.2272536754608154 | time/iterations = 900 | rollout/ep_rew_mean = 45.69387755102041 | rollout/ep_len_mean = 45.69387755102041 | time/fps = 87 | time/time_elapsed = 51 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 6]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 8.481740951538086e-05 | train/entropy_loss = -0.6253606081008911 | train/policy_loss = -11.025751113891602 | train/value_loss = 816.6910400390625 | time/iterations = 1000 | rollout/ep_rew_mean = 32.26 | rollout/ep_len_mean = 32.26 | time/fps = 82 | time/time_elapsed = 60 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 8.481740951538086e-05 | train/entropy_loss = -0.6253606081008911 | train/policy_loss = -11.025751113891602 | train/value_loss = 816.6910400390625 | time/iterations = 1000 | rollout/ep_rew_mean = 32.26 | rollout/ep_len_mean = 32.26 | time/fps = 82 | time/time_elapsed = 60 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 7]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.00040328502655029297 | train/entropy_loss = -0.5151488780975342 | train/policy_loss = 0.6045753359794617 | train/value_loss = 2.386056423187256 | time/iterations = 1000 | rollout/ep_rew_mean = 77.44444444444444 | rollout/ep_len_mean = 77.44444444444444 | time/fps = 84 | time/time_elapsed = 59 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.00040328502655029297 | train/entropy_loss = -0.5151488780975342 | train/policy_loss = 0.6045753359794617 | train/value_loss = 2.386056423187256 | time/iterations = 1000 | rollout/ep_rew_mean = 77.44444444444444 | rollout/ep_len_mean = 77.44444444444444 | time/fps = 84 | time/time_elapsed = 59 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 8]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 7.653236389160156e-05 | train/entropy_loss = -0.4422871470451355 | train/policy_loss = 0.5032252073287964 | train/value_loss = 2.7981576919555664 | time/iterations = 1000 | rollout/ep_rew_mean = 49.19 | rollout/ep_len_mean = 49.19 | time/fps = 85 | time/time_elapsed = 58 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 7.653236389160156e-05 | train/entropy_loss = -0.4422871470451355 | train/policy_loss = 0.5032252073287964 | train/value_loss = 2.7981576919555664 | time/iterations = 1000 | rollout/ep_rew_mean = 49.19 | rollout/ep_len_mean = 49.19 | time/fps = 85 | time/time_elapsed = 58 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:41: [Sb3-A2C[worker: 9]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.0032252073287963867 | train/entropy_loss = -0.43446826934814453 | train/policy_loss = 0.2745859622955322 | train/value_loss = 2.7972793579101562 | time/iterations = 1000 | rollout/ep_rew_mean = 48.08 | rollout/ep_len_mean = 48.08 | time/fps = 86 | time/time_elapsed = 57 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.0032252073287963867 | train/entropy_loss = -0.43446826934814453 | train/policy_loss = 0.2745859622955322 | train/value_loss = 2.7972793579101562 | time/iterations = 1000 | rollout/ep_rew_mean = 48.08 | rollout/ep_len_mean = 48.08 | time/fps = 86 | time/time_elapsed = 57 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 6]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.011163830757141113 | train/entropy_loss = -0.5054067373275757 | train/policy_loss = 1.1785739660263062 | train/value_loss = 3.4740662574768066 | time/iterations = 1100 | rollout/ep_rew_mean = 35.69 | rollout/ep_len_mean = 35.69 | time/fps = 83 | time/time_elapsed = 65 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.011163830757141113 | train/entropy_loss = -0.5054067373275757 | train/policy_loss = 1.1785739660263062 | train/value_loss = 3.4740662574768066 | time/iterations = 1100 | rollout/ep_rew_mean = 35.69 | rollout/ep_len_mean = 35.69 | time/fps = 83 | time/time_elapsed = 65 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 7]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.00027871131896972656 | train/entropy_loss = -0.5161913633346558 | train/policy_loss = 0.43842649459838867 | train/value_loss = 1.981288194656372 | time/iterations = 1100 | rollout/ep_rew_mean = 80.5223880597015 | rollout/ep_len_mean = 80.5223880597015 | time/fps = 84 | time/time_elapsed = 64 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.00027871131896972656 | train/entropy_loss = -0.5161913633346558 | train/policy_loss = 0.43842649459838867 | train/value_loss = 1.981288194656372 | time/iterations = 1100 | rollout/ep_rew_mean = 80.5223880597015 | rollout/ep_len_mean = 80.5223880597015 | time/fps = 84 | time/time_elapsed = 64 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 8]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.00014543533325195312 | train/entropy_loss = -0.392616331577301 | train/policy_loss = 0.9541427493095398 | train/value_loss = 2.357178211212158 | time/iterations = 1100 | rollout/ep_rew_mean = 53.13 | rollout/ep_len_mean = 53.13 | time/fps = 86 | time/time_elapsed = 63 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.00014543533325195312 | train/entropy_loss = -0.392616331577301 | train/policy_loss = 0.9541427493095398 | train/value_loss = 2.357178211212158 | time/iterations = 1100 | rollout/ep_rew_mean = 53.13 | rollout/ep_len_mean = 53.13 | time/fps = 86 | time/time_elapsed = 63 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 9]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.00012612342834472656 | train/entropy_loss = -0.43642401695251465 | train/policy_loss = 0.30986231565475464 | train/value_loss = 2.3565428256988525 | time/iterations = 1100 | rollout/ep_rew_mean = 51.31 | rollout/ep_len_mean = 51.31 | time/fps = 87 | time/time_elapsed = 63 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.00012612342834472656 | train/entropy_loss = -0.43642401695251465 | train/policy_loss = 0.30986231565475464 | train/value_loss = 2.3565428256988525 | time/iterations = 1100 | rollout/ep_rew_mean = 51.31 | rollout/ep_len_mean = 51.31 | time/fps = 87 | time/time_elapsed = 63 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 6]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.004240214824676514 | train/entropy_loss = -0.43997782468795776 | train/policy_loss = 1.5866310596466064 | train/value_loss = 2.9799153804779053 | time/iterations = 1200 | rollout/ep_rew_mean = 40.33 | rollout/ep_len_mean = 40.33 | time/fps = 84 | time/time_elapsed = 70 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.004240214824676514 | train/entropy_loss = -0.43997782468795776 | train/policy_loss = 1.5866310596466064 | train/value_loss = 2.9799153804779053 | time/iterations = 1200 | rollout/ep_rew_mean = 40.33 | rollout/ep_len_mean = 40.33 | time/fps = 84 | time/time_elapsed = 70 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 7]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00010669231414794922 | train/entropy_loss = -0.4234845042228699 | train/policy_loss = 0.23966078460216522 | train/value_loss = 1.6140520572662354 | time/iterations = 1200 | rollout/ep_rew_mean = 83.47887323943662 | rollout/ep_len_mean = 83.47887323943662 | time/fps = 86 | time/time_elapsed = 69 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00010669231414794922 | train/entropy_loss = -0.4234845042228699 | train/policy_loss = 0.23966078460216522 | train/value_loss = 1.6140520572662354 | time/iterations = 1200 | rollout/ep_rew_mean = 83.47887323943662 | rollout/ep_len_mean = 83.47887323943662 | time/fps = 86 | time/time_elapsed = 69 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 8]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -6.222724914550781e-05 | train/entropy_loss = -0.5098690986633301 | train/policy_loss = 0.42751389741897583 | train/value_loss = 1.9521992206573486 | time/iterations = 1200 | rollout/ep_rew_mean = 56.93 | rollout/ep_len_mean = 56.93 | time/fps = 87 | time/time_elapsed = 68 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -6.222724914550781e-05 | train/entropy_loss = -0.5098690986633301 | train/policy_loss = 0.42751389741897583 | train/value_loss = 1.9521992206573486 | time/iterations = 1200 | rollout/ep_rew_mean = 56.93 | rollout/ep_len_mean = 56.93 | time/fps = 87 | time/time_elapsed = 68 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 9]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.0010905265808105469 | train/entropy_loss = -0.4501417279243469 | train/policy_loss = 0.9220317006111145 | train/value_loss = 1.9484237432479858 | time/iterations = 1200 | rollout/ep_rew_mean = 53.76 | rollout/ep_len_mean = 53.76 | time/fps = 87 | time/time_elapsed = 68 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.0010905265808105469 | train/entropy_loss = -0.4501417279243469 | train/policy_loss = 0.9220317006111145 | train/value_loss = 1.9484237432479858 | time/iterations = 1200 | rollout/ep_rew_mean = 53.76 | rollout/ep_len_mean = 53.76 | time/fps = 87 | time/time_elapsed = 68 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 6]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0018908977508544922 | train/entropy_loss = -0.614406943321228 | train/policy_loss = 0.5966942310333252 | train/value_loss = 2.546813488006592 | time/iterations = 1300 | rollout/ep_rew_mean = 44.6 | rollout/ep_len_mean = 44.6 | time/fps = 85 | time/time_elapsed = 76 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0018908977508544922 | train/entropy_loss = -0.614406943321228 | train/policy_loss = 0.5966942310333252 | train/value_loss = 2.546813488006592 | time/iterations = 1300 | rollout/ep_rew_mean = 44.6 | rollout/ep_len_mean = 44.6 | time/fps = 85 | time/time_elapsed = 76 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 7]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.00032138824462890625 | train/entropy_loss = -0.29264599084854126 | train/policy_loss = 1.0775235891342163 | train/value_loss = 1.2779738903045654 | time/iterations = 1300 | rollout/ep_rew_mean = 86.79729729729729 | rollout/ep_len_mean = 86.79729729729729 | time/fps = 86 | time/time_elapsed = 74 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.00032138824462890625 | train/entropy_loss = -0.29264599084854126 | train/policy_loss = 1.0775235891342163 | train/value_loss = 1.2779738903045654 | time/iterations = 1300 | rollout/ep_rew_mean = 86.79729729729729 | rollout/ep_len_mean = 86.79729729729729 | time/fps = 86 | time/time_elapsed = 74 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 8]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0001499652862548828 | train/entropy_loss = -0.4738292694091797 | train/policy_loss = 0.3032841682434082 | train/value_loss = 1.5808780193328857 | time/iterations = 1300 | rollout/ep_rew_mean = 61.16 | rollout/ep_len_mean = 61.16 | time/fps = 87 | time/time_elapsed = 74 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0001499652862548828 | train/entropy_loss = -0.4738292694091797 | train/policy_loss = 0.3032841682434082 | train/value_loss = 1.5808780193328857 | time/iterations = 1300 | rollout/ep_rew_mean = 61.16 | rollout/ep_len_mean = 61.16 | time/fps = 87 | time/time_elapsed = 74 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 9]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -3.933906555175781e-06 | train/entropy_loss = -0.489692360162735 | train/policy_loss = 0.49198657274246216 | train/value_loss = 1.598267912864685 | time/iterations = 1300 | rollout/ep_rew_mean = 57.88 | rollout/ep_len_mean = 57.88 | time/fps = 88 | time/time_elapsed = 73 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -3.933906555175781e-06 | train/entropy_loss = -0.489692360162735 | train/policy_loss = 0.49198657274246216 | train/value_loss = 1.598267912864685 | time/iterations = 1300 | rollout/ep_rew_mean = 57.88 | rollout/ep_len_mean = 57.88 | time/fps = 88 | time/time_elapsed = 73 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 6]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -5.1856040954589844e-05 | train/entropy_loss = -0.5534188747406006 | train/policy_loss = 0.8610655069351196 | train/value_loss = 2.091984272003174 | time/iterations = 1400 | rollout/ep_rew_mean = 49.38 | rollout/ep_len_mean = 49.38 | time/fps = 85 | time/time_elapsed = 81 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -5.1856040954589844e-05 | train/entropy_loss = -0.5534188747406006 | train/policy_loss = 0.8610655069351196 | train/value_loss = 2.091984272003174 | time/iterations = 1400 | rollout/ep_rew_mean = 49.38 | rollout/ep_len_mean = 49.38 | time/fps = 85 | time/time_elapsed = 81 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 7]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.00044083595275878906 | train/entropy_loss = -0.44299402832984924 | train/policy_loss = 0.16189515590667725 | train/value_loss = 0.9880059957504272 | time/iterations = 1400 | rollout/ep_rew_mean = 89.56410256410257 | rollout/ep_len_mean = 89.56410256410257 | time/fps = 87 | time/time_elapsed = 80 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.00044083595275878906 | train/entropy_loss = -0.44299402832984924 | train/policy_loss = 0.16189515590667725 | train/value_loss = 0.9880059957504272 | time/iterations = 1400 | rollout/ep_rew_mean = 89.56410256410257 | rollout/ep_len_mean = 89.56410256410257 | time/fps = 87 | time/time_elapsed = 80 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 8]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -9.298324584960938e-06 | train/entropy_loss = -0.4460598826408386 | train/policy_loss = 0.5154594779014587 | train/value_loss = 1.2356077432632446 | time/iterations = 1400 | rollout/ep_rew_mean = 64.5 | rollout/ep_len_mean = 64.5 | time/fps = 88 | time/time_elapsed = 79 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -9.298324584960938e-06 | train/entropy_loss = -0.4460598826408386 | train/policy_loss = 0.5154594779014587 | train/value_loss = 1.2356077432632446 | time/iterations = 1400 | rollout/ep_rew_mean = 64.5 | rollout/ep_len_mean = 64.5 | time/fps = 88 | time/time_elapsed = 79 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 9]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.00014925003051757812 | train/entropy_loss = -0.3381018042564392 | train/policy_loss = 0.7999730706214905 | train/value_loss = 1.277901291847229 | time/iterations = 1400 | rollout/ep_rew_mean = 61.04 | rollout/ep_len_mean = 61.04 | time/fps = 88 | time/time_elapsed = 79 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.00014925003051757812 | train/entropy_loss = -0.3381018042564392 | train/policy_loss = 0.7999730706214905 | train/value_loss = 1.277901291847229 | time/iterations = 1400 | rollout/ep_rew_mean = 61.04 | rollout/ep_len_mean = 61.04 | time/fps = 88 | time/time_elapsed = 79 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 6]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0006810426712036133 | train/entropy_loss = -0.46632084250450134 | train/policy_loss = 0.8761663436889648 | train/value_loss = 1.705613136291504 | time/iterations = 1500 | rollout/ep_rew_mean = 52.7 | rollout/ep_len_mean = 52.7 | time/fps = 86 | time/time_elapsed = 86 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0006810426712036133 | train/entropy_loss = -0.46632084250450134 | train/policy_loss = 0.8761663436889648 | train/value_loss = 1.705613136291504 | time/iterations = 1500 | rollout/ep_rew_mean = 52.7 | rollout/ep_len_mean = 52.7 | time/fps = 86 | time/time_elapsed = 86 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 7]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.00012695789337158203 | train/entropy_loss = -0.46482545137405396 | train/policy_loss = 0.3362646698951721 | train/value_loss = 0.7206336259841919 | time/iterations = 1500 | rollout/ep_rew_mean = 92.48101265822785 | rollout/ep_len_mean = 92.48101265822785 | time/fps = 87 | time/time_elapsed = 85 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.00012695789337158203 | train/entropy_loss = -0.46482545137405396 | train/policy_loss = 0.3362646698951721 | train/value_loss = 0.7206336259841919 | time/iterations = 1500 | rollout/ep_rew_mean = 92.48101265822785 | rollout/ep_len_mean = 92.48101265822785 | time/fps = 87 | time/time_elapsed = 85 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 8]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.0008360743522644043 | train/entropy_loss = -0.3936169743537903 | train/policy_loss = 0.252361536026001 | train/value_loss = 0.9269996881484985 | time/iterations = 1500 | rollout/ep_rew_mean = 69.34 | rollout/ep_len_mean = 69.34 | time/fps = 88 | time/time_elapsed = 84 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.0008360743522644043 | train/entropy_loss = -0.3936169743537903 | train/policy_loss = 0.252361536026001 | train/value_loss = 0.9269996881484985 | time/iterations = 1500 | rollout/ep_rew_mean = 69.34 | rollout/ep_len_mean = 69.34 | time/fps = 88 | time/time_elapsed = 84 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 9]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 1.7881393432617188e-07 | train/entropy_loss = -0.4175400137901306 | train/policy_loss = 0.13957342505455017 | train/value_loss = 0.993108868598938 | time/iterations = 1500 | rollout/ep_rew_mean = 64.42 | rollout/ep_len_mean = 64.42 | time/fps = 89 | time/time_elapsed = 84 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 1.7881393432617188e-07 | train/entropy_loss = -0.4175400137901306 | train/policy_loss = 0.13957342505455017 | train/value_loss = 0.993108868598938 | time/iterations = 1500 | rollout/ep_rew_mean = 64.42 | rollout/ep_len_mean = 64.42 | time/fps = 89 | time/time_elapsed = 84 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 6]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.000768125057220459 | train/entropy_loss = -0.434145450592041 | train/policy_loss = 1.0435283184051514 | train/value_loss = 1.3513857126235962 | time/iterations = 1600 | rollout/ep_rew_mean = 56.37 | rollout/ep_len_mean = 56.37 | time/fps = 86 | time/time_elapsed = 91 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.000768125057220459 | train/entropy_loss = -0.434145450592041 | train/policy_loss = 1.0435283184051514 | train/value_loss = 1.3513857126235962 | time/iterations = 1600 | rollout/ep_rew_mean = 56.37 | rollout/ep_len_mean = 56.37 | time/fps = 86 | time/time_elapsed = 91 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 7]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -4.267692565917969e-05 | train/entropy_loss = -0.5674619674682617 | train/policy_loss = 0.2382907122373581 | train/value_loss = 0.5067436695098877 | time/iterations = 1600 | rollout/ep_rew_mean = 95.34939759036145 | rollout/ep_len_mean = 95.34939759036145 | time/fps = 88 | time/time_elapsed = 90 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -4.267692565917969e-05 | train/entropy_loss = -0.5674619674682617 | train/policy_loss = 0.2382907122373581 | train/value_loss = 0.5067436695098877 | time/iterations = 1600 | rollout/ep_rew_mean = 95.34939759036145 | rollout/ep_len_mean = 95.34939759036145 | time/fps = 88 | time/time_elapsed = 90 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 8]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 5.21540641784668e-05 | train/entropy_loss = -0.3435831069946289 | train/policy_loss = 0.41466569900512695 | train/value_loss = 0.6674198508262634 | time/iterations = 1600 | rollout/ep_rew_mean = 75.75 | rollout/ep_len_mean = 75.75 | time/fps = 88 | time/time_elapsed = 90 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 5.21540641784668e-05 | train/entropy_loss = -0.3435831069946289 | train/policy_loss = 0.41466569900512695 | train/value_loss = 0.6674198508262634 | time/iterations = 1600 | rollout/ep_rew_mean = 75.75 | rollout/ep_len_mean = 75.75 | time/fps = 88 | time/time_elapsed = 90 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 9]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -7.712841033935547e-05 | train/entropy_loss = -0.427565336227417 | train/policy_loss = 0.2844379246234894 | train/value_loss = 0.7333279848098755 | time/iterations = 1600 | rollout/ep_rew_mean = 68.13 | rollout/ep_len_mean = 68.13 | time/fps = 89 | time/time_elapsed = 89 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -7.712841033935547e-05 | train/entropy_loss = -0.427565336227417 | train/policy_loss = 0.2844379246234894 | train/value_loss = 0.7333279848098755 | time/iterations = 1600 | rollout/ep_rew_mean = 68.13 | rollout/ep_len_mean = 68.13 | time/fps = 89 | time/time_elapsed = 89 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 6]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 0.0 | train/entropy_loss = -0.6492987871170044 | train/policy_loss = -26.272314071655273 | train/value_loss = 2680.147216796875 | time/iterations = 1700 | rollout/ep_rew_mean = 62.85 | rollout/ep_len_mean = 62.85 | time/fps = 87 | time/time_elapsed = 97 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 0.0 | train/entropy_loss = -0.6492987871170044 | train/policy_loss = -26.272314071655273 | train/value_loss = 2680.147216796875 | time/iterations = 1700 | rollout/ep_rew_mean = 62.85 | rollout/ep_len_mean = 62.85 | time/fps = 87 | time/time_elapsed = 97 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 7]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -2.6464462280273438e-05 | train/entropy_loss = -0.4914538860321045 | train/policy_loss = 0.1266205906867981 | train/value_loss = 0.33009687066078186 | time/iterations = 1700 | rollout/ep_rew_mean = 96.64367816091954 | rollout/ep_len_mean = 96.64367816091954 | time/fps = 88 | time/time_elapsed = 96 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -2.6464462280273438e-05 | train/entropy_loss = -0.4914538860321045 | train/policy_loss = 0.1266205906867981 | train/value_loss = 0.33009687066078186 | time/iterations = 1700 | rollout/ep_rew_mean = 96.64367816091954 | rollout/ep_len_mean = 96.64367816091954 | time/fps = 88 | time/time_elapsed = 96 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 8]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 1.5854835510253906e-05 | train/entropy_loss = -0.38072577118873596 | train/policy_loss = 0.15771885216236115 | train/value_loss = 0.4588068127632141 | time/iterations = 1700 | rollout/ep_rew_mean = 80.45 | rollout/ep_len_mean = 80.45 | time/fps = 88 | time/time_elapsed = 95 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 1.5854835510253906e-05 | train/entropy_loss = -0.38072577118873596 | train/policy_loss = 0.15771885216236115 | train/value_loss = 0.4588068127632141 | time/iterations = 1700 | rollout/ep_rew_mean = 80.45 | rollout/ep_len_mean = 80.45 | time/fps = 88 | time/time_elapsed = 95 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 9]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 0.00011461973190307617 | train/entropy_loss = -0.3123507499694824 | train/policy_loss = 0.5541085004806519 | train/value_loss = 0.5087783336639404 | time/iterations = 1700 | rollout/ep_rew_mean = 71.22 | rollout/ep_len_mean = 71.22 | time/fps = 88 | time/time_elapsed = 95 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 0.00011461973190307617 | train/entropy_loss = -0.3123507499694824 | train/policy_loss = 0.5541085004806519 | train/value_loss = 0.5087783336639404 | time/iterations = 1700 | rollout/ep_rew_mean = 71.22 | rollout/ep_len_mean = 71.22 | time/fps = 88 | time/time_elapsed = 95 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 7]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 4.762411117553711e-05 | train/entropy_loss = -0.3774551451206207 | train/policy_loss = 0.15526577830314636 | train/value_loss = 0.1846795529127121 | time/iterations = 1800 | rollout/ep_rew_mean = 97.93181818181819 | rollout/ep_len_mean = 97.93181818181819 | time/fps = 87 | time/time_elapsed = 102 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 4.762411117553711e-05 | train/entropy_loss = -0.3774551451206207 | train/policy_loss = 0.15526577830314636 | train/value_loss = 0.1846795529127121 | time/iterations = 1800 | rollout/ep_rew_mean = 97.93181818181819 | rollout/ep_len_mean = 97.93181818181819 | time/fps = 87 | time/time_elapsed = 102 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 6]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -0.00028836727142333984 | train/entropy_loss = -0.4327360987663269 | train/policy_loss = 0.4542310833930969 | train/value_loss = 0.787033200263977 | time/iterations = 1800 | rollout/ep_rew_mean = 66.43 | rollout/ep_len_mean = 66.43 | time/fps = 86 | time/time_elapsed = 103 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -0.00028836727142333984 | train/entropy_loss = -0.4327360987663269 | train/policy_loss = 0.4542310833930969 | train/value_loss = 0.787033200263977 | time/iterations = 1800 | rollout/ep_rew_mean = 66.43 | rollout/ep_len_mean = 66.43 | time/fps = 86 | time/time_elapsed = 103 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 8]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 9.548664093017578e-05 | train/entropy_loss = -0.43599724769592285 | train/policy_loss = 0.2910926043987274 | train/value_loss = 0.28519725799560547 | time/iterations = 1800 | rollout/ep_rew_mean = 84.58 | rollout/ep_len_mean = 84.58 | time/fps = 88 | time/time_elapsed = 102 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 9.548664093017578e-05 | train/entropy_loss = -0.43599724769592285 | train/policy_loss = 0.2910926043987274 | train/value_loss = 0.28519725799560547 | time/iterations = 1800 | rollout/ep_rew_mean = 84.58 | rollout/ep_len_mean = 84.58 | time/fps = 88 | time/time_elapsed = 102 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 9]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 1.1920928955078125e-07 | train/entropy_loss = -0.37260276079177856 | train/policy_loss = 0.19338969886302948 | train/value_loss = 0.3356333374977112 | time/iterations = 1800 | rollout/ep_rew_mean = 75.6 | rollout/ep_len_mean = 75.6 | time/fps = 88 | time/time_elapsed = 101 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 1.1920928955078125e-07 | train/entropy_loss = -0.37260276079177856 | train/policy_loss = 0.19338969886302948 | train/value_loss = 0.3356333374977112 | time/iterations = 1800 | rollout/ep_rew_mean = 75.6 | rollout/ep_len_mean = 75.6 | time/fps = 88 | time/time_elapsed = 101 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 7]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.0 | train/entropy_loss = -0.4421364665031433 | train/policy_loss = 0.07432230561971664 | train/value_loss = 0.0822327733039856 | time/iterations = 1900 | rollout/ep_rew_mean = 102.68888888888888 | rollout/ep_len_mean = 102.68888888888888 | time/fps = 87 | time/time_elapsed = 107 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 7]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.0 | train/entropy_loss = -0.4421364665031433 | train/policy_loss = 0.07432230561971664 | train/value_loss = 0.0822327733039856 | time/iterations = 1900 | rollout/ep_rew_mean = 102.68888888888888 | rollout/ep_len_mean = 102.68888888888888 | time/fps = 87 | time/time_elapsed = 107 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 6]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.0005055069923400879 | train/entropy_loss = -0.6454207897186279 | train/policy_loss = 0.3315656781196594 | train/value_loss = 0.5571736097335815 | time/iterations = 1900 | rollout/ep_rew_mean = 69.77 | rollout/ep_len_mean = 69.77 | time/fps = 86 | time/time_elapsed = 109 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 6]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.0005055069923400879 | train/entropy_loss = -0.6454207897186279 | train/policy_loss = 0.3315656781196594 | train/value_loss = 0.5571736097335815 | time/iterations = 1900 | rollout/ep_rew_mean = 69.77 | rollout/ep_len_mean = 69.77 | time/fps = 86 | time/time_elapsed = 109 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 8]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.00022411346435546875 | train/entropy_loss = -0.37226131558418274 | train/policy_loss = 0.14946675300598145 | train/value_loss = 0.15059608221054077 | time/iterations = 1900 | rollout/ep_rew_mean = 88.73 | rollout/ep_len_mean = 88.73 | time/fps = 88 | time/time_elapsed = 107 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 8]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.00022411346435546875 | train/entropy_loss = -0.37226131558418274 | train/policy_loss = 0.14946675300598145 | train/value_loss = 0.15059608221054077 | time/iterations = 1900 | rollout/ep_rew_mean = 88.73 | rollout/ep_len_mean = 88.73 | time/fps = 88 | time/time_elapsed = 107 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:42: [Sb3-A2C[worker: 9]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -3.445148468017578e-05 | train/entropy_loss = -0.4596175253391266 | train/policy_loss = 0.1725001335144043 | train/value_loss = 0.19814392924308777 | time/iterations = 1900 | rollout/ep_rew_mean = 78.96 | rollout/ep_len_mean = 78.96 | time/fps = 88 | time/time_elapsed = 106 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[Sb3-A2C[worker: 9]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -3.445148468017578e-05 | train/entropy_loss = -0.4596175253391266 | train/policy_loss = 0.1725001335144043 | train/value_loss = 0.19814392924308777 | time/iterations = 1900 | rollout/ep_rew_mean = 78.96 | rollout/ep_len_mean = 78.96 | time/fps = 88 | time/time_elapsed = 106 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:42: ... trained! [0m
    INFO:rlberry_logger:... trained!
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:42: Saved ExperimentManager(Sb3-A2C) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(Sb3-A2C) using pickle.
    [38;21m[INFO] 14:42: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/Sb3-A2C_2024-04-03_14-38-22_ac8e2514/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/Sb3-A2C_2024-04-03_14-38-22_ac8e2514/manager_obj.pickle'
    [38;21m[INFO] 14:42: Running ExperimentManager fit() for Sb3-PPO with n_fit = 10 and max_workers = None. [0m
    INFO:rlberry_logger:Running ExperimentManager fit() for Sb3-PPO with n_fit = 10 and max_workers = None.
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43:                  agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      0           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      0           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43:                  agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      4           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      4           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43:                  agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      5           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      5           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43:                  agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      2           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      2           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43:                  agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      1           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      1           1               2048
    [38;21m[INFO] 14:43:                  agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      3           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      3           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43: [Sb3-PPO[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.0 | rollout/ep_len_mean = 22.0 | time/fps = 117 | time/time_elapsed = 17 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.0 | rollout/ep_len_mean = 22.0 | time/fps = 117 | time/time_elapsed = 17 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43: [Sb3-PPO[worker: 5]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.51764705882353 | rollout/ep_len_mean = 23.51764705882353 | time/fps = 115 | time/time_elapsed = 17 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 5]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.51764705882353 | rollout/ep_len_mean = 23.51764705882353 | time/fps = 115 | time/time_elapsed = 17 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43: [Sb3-PPO[worker: 4]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.75531914893617 | rollout/ep_len_mean = 21.75531914893617 | time/fps = 116 | time/time_elapsed = 17 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 4]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.75531914893617 | rollout/ep_len_mean = 21.75531914893617 | time/fps = 116 | time/time_elapsed = 17 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43: [Sb3-PPO[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.45263157894737 | rollout/ep_len_mean = 21.45263157894737 | time/fps = 113 | time/time_elapsed = 18 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.45263157894737 | rollout/ep_len_mean = 21.45263157894737 | time/fps = 113 | time/time_elapsed = 18 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43: [Sb3-PPO[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.01086956521739 | rollout/ep_len_mean = 22.01086956521739 | time/fps = 111 | time/time_elapsed = 18 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.01086956521739 | rollout/ep_len_mean = 22.01086956521739 | time/fps = 111 | time/time_elapsed = 18 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:43: [Sb3-PPO[worker: 3]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.193181818181817 | rollout/ep_len_mean = 23.193181818181817 | time/fps = 111 | time/time_elapsed = 18 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 3]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.193181818181817 | rollout/ep_len_mean = 23.193181818181817 | time/fps = 111 | time/time_elapsed = 18 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 4]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.93 | rollout/ep_len_mean = 25.93 | time/fps = 75 | time/time_elapsed = 54 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6868883330374956 | train/policy_gradient_loss = -0.011192558324546553 | train/value_loss = 52.31638530790806 | train/approx_kl = 0.009392955340445042 | train/clip_fraction = 0.083447265625 | train/loss = 6.202040672302246 | train/explained_variance = -0.0003072023391723633 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 4]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.93 | rollout/ep_len_mean = 25.93 | time/fps = 75 | time/time_elapsed = 54 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6868883330374956 | train/policy_gradient_loss = -0.011192558324546553 | train/value_loss = 52.31638530790806 | train/approx_kl = 0.009392955340445042 | train/clip_fraction = 0.083447265625 | train/loss = 6.202040672302246 | train/explained_variance = -0.0003072023391723633 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 5]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 29.3 | rollout/ep_len_mean = 29.3 | time/fps = 75 | time/time_elapsed = 54 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6867934368550778 | train/policy_gradient_loss = -0.0121891417686129 | train/value_loss = 53.787508270144464 | train/approx_kl = 0.007793963886797428 | train/clip_fraction = 0.0779296875 | train/loss = 7.109466075897217 | train/explained_variance = -8.654594421386719e-05 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 5]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 29.3 | rollout/ep_len_mean = 29.3 | time/fps = 75 | time/time_elapsed = 54 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6867934368550778 | train/policy_gradient_loss = -0.0121891417686129 | train/value_loss = 53.787508270144464 | train/approx_kl = 0.007793963886797428 | train/clip_fraction = 0.0779296875 | train/loss = 7.109466075897217 | train/explained_variance = -8.654594421386719e-05 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.22 | rollout/ep_len_mean = 25.22 | time/fps = 75 | time/time_elapsed = 53 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6858226422220468 | train/policy_gradient_loss = -0.017927717691054567 | train/value_loss = 49.82425828278065 | train/approx_kl = 0.00978387612849474 | train/clip_fraction = 0.11357421875 | train/loss = 6.644534111022949 | train/explained_variance = 0.002265334129333496 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.22 | rollout/ep_len_mean = 25.22 | time/fps = 75 | time/time_elapsed = 53 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6858226422220468 | train/policy_gradient_loss = -0.017927717691054567 | train/value_loss = 49.82425828278065 | train/approx_kl = 0.00978387612849474 | train/clip_fraction = 0.11357421875 | train/loss = 6.644534111022949 | train/explained_variance = 0.002265334129333496 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 2]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.89 | rollout/ep_len_mean = 26.89 | time/fps = 75 | time/time_elapsed = 54 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6869663989171386 | train/policy_gradient_loss = -0.010359835267445305 | train/value_loss = 50.284837391972545 | train/approx_kl = 0.007852415554225445 | train/clip_fraction = 0.076171875 | train/loss = 5.862208366394043 | train/explained_variance = 0.0027590394020080566 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 2]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.89 | rollout/ep_len_mean = 26.89 | time/fps = 75 | time/time_elapsed = 54 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6869663989171386 | train/policy_gradient_loss = -0.010359835267445305 | train/value_loss = 50.284837391972545 | train/approx_kl = 0.007852415554225445 | train/clip_fraction = 0.076171875 | train/loss = 5.862208366394043 | train/explained_variance = 0.0027590394020080566 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.81 | rollout/ep_len_mean = 27.81 | time/fps = 74 | time/time_elapsed = 55 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6864938329905271 | train/policy_gradient_loss = -0.011870424351945985 | train/value_loss = 51.04571032524109 | train/approx_kl = 0.008346617221832275 | train/clip_fraction = 0.08037109375 | train/loss = 6.404996871948242 | train/explained_variance = -0.008775472640991211 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.81 | rollout/ep_len_mean = 27.81 | time/fps = 74 | time/time_elapsed = 55 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6864938329905271 | train/policy_gradient_loss = -0.011870424351945985 | train/value_loss = 51.04571032524109 | train/approx_kl = 0.008346617221832275 | train/clip_fraction = 0.08037109375 | train/loss = 6.404996871948242 | train/explained_variance = -0.008775472640991211 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 3]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 29.41 | rollout/ep_len_mean = 29.41 | time/fps = 74 | time/time_elapsed = 55 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.686557256616652 | train/policy_gradient_loss = -0.014514252974186093 | train/value_loss = 59.32745496034622 | train/approx_kl = 0.008458458818495274 | train/clip_fraction = 0.0962890625 | train/loss = 8.542673110961914 | train/explained_variance = -0.005324125289916992 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 3]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 29.41 | rollout/ep_len_mean = 29.41 | time/fps = 74 | time/time_elapsed = 55 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.686557256616652 | train/policy_gradient_loss = -0.014514252974186093 | train/value_loss = 59.32745496034622 | train/approx_kl = 0.008458458818495274 | train/clip_fraction = 0.0962890625 | train/loss = 8.542673110961914 | train/explained_variance = -0.005324125289916992 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 4]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 35.94 | rollout/ep_len_mean = 35.94 | time/fps = 74 | time/time_elapsed = 81 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6681853344663977 | train/policy_gradient_loss = -0.017905895704461727 | train/value_loss = 33.303678047657016 | train/approx_kl = 0.010633700527250767 | train/clip_fraction = 0.066796875 | train/loss = 9.377118110656738 | train/explained_variance = 0.12500154972076416 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 4]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 35.94 | rollout/ep_len_mean = 35.94 | time/fps = 74 | time/time_elapsed = 81 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6681853344663977 | train/policy_gradient_loss = -0.017905895704461727 | train/value_loss = 33.303678047657016 | train/approx_kl = 0.010633700527250767 | train/clip_fraction = 0.066796875 | train/loss = 9.377118110656738 | train/explained_variance = 0.12500154972076416 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 5]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 37.18 | rollout/ep_len_mean = 37.18 | time/fps = 74 | time/time_elapsed = 82 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6685162043198943 | train/policy_gradient_loss = -0.01912627555866493 | train/value_loss = 36.63949154615402 | train/approx_kl = 0.009746195748448372 | train/clip_fraction = 0.06875 | train/loss = 11.260287284851074 | train/explained_variance = 0.02012157440185547 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 5]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 37.18 | rollout/ep_len_mean = 37.18 | time/fps = 74 | time/time_elapsed = 82 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6685162043198943 | train/policy_gradient_loss = -0.01912627555866493 | train/value_loss = 36.63949154615402 | train/approx_kl = 0.009746195748448372 | train/clip_fraction = 0.06875 | train/loss = 11.260287284851074 | train/explained_variance = 0.02012157440185547 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 1]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 36.91 | rollout/ep_len_mean = 36.91 | time/fps = 73 | time/time_elapsed = 83 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6680108092725277 | train/policy_gradient_loss = -0.011856390519824345 | train/value_loss = 37.08693499565125 | train/approx_kl = 0.00897466391324997 | train/clip_fraction = 0.03994140625 | train/loss = 12.149518966674805 | train/explained_variance = 0.09349644184112549 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 1]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 36.91 | rollout/ep_len_mean = 36.91 | time/fps = 73 | time/time_elapsed = 83 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6680108092725277 | train/policy_gradient_loss = -0.011856390519824345 | train/value_loss = 37.08693499565125 | train/approx_kl = 0.00897466391324997 | train/clip_fraction = 0.03994140625 | train/loss = 12.149518966674805 | train/explained_variance = 0.09349644184112549 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 2]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 35.91 | rollout/ep_len_mean = 35.91 | time/fps = 74 | time/time_elapsed = 82 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6708333157002926 | train/policy_gradient_loss = -0.012362689161091112 | train/value_loss = 35.24872809648514 | train/approx_kl = 0.008301440626382828 | train/clip_fraction = 0.039794921875 | train/loss = 11.0582857131958 | train/explained_variance = 0.09413212537765503 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 2]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 35.91 | rollout/ep_len_mean = 35.91 | time/fps = 74 | time/time_elapsed = 82 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6708333157002926 | train/policy_gradient_loss = -0.012362689161091112 | train/value_loss = 35.24872809648514 | train/approx_kl = 0.008301440626382828 | train/clip_fraction = 0.039794921875 | train/loss = 11.0582857131958 | train/explained_variance = 0.09413212537765503 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 0]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.19 | rollout/ep_len_mean = 33.19 | time/fps = 74 | time/time_elapsed = 82 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.666435769572854 | train/policy_gradient_loss = -0.016837025916902348 | train/value_loss = 32.14510078430176 | train/approx_kl = 0.010450788773596287 | train/clip_fraction = 0.063232421875 | train/loss = 13.860794067382812 | train/explained_variance = 0.07467210292816162 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 0]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.19 | rollout/ep_len_mean = 33.19 | time/fps = 74 | time/time_elapsed = 82 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.666435769572854 | train/policy_gradient_loss = -0.016837025916902348 | train/value_loss = 32.14510078430176 | train/approx_kl = 0.010450788773596287 | train/clip_fraction = 0.063232421875 | train/loss = 13.860794067382812 | train/explained_variance = 0.07467210292816162 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:44: [Sb3-PPO[worker: 3]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 38.99 | rollout/ep_len_mean = 38.99 | time/fps = 73 | time/time_elapsed = 83 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6654575899243355 | train/policy_gradient_loss = -0.016497393150348216 | train/value_loss = 37.94590861797333 | train/approx_kl = 0.008675235323607922 | train/clip_fraction = 0.0568359375 | train/loss = 13.926454544067383 | train/explained_variance = 0.040100157260894775 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 3]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 38.99 | rollout/ep_len_mean = 38.99 | time/fps = 73 | time/time_elapsed = 83 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6654575899243355 | train/policy_gradient_loss = -0.016497393150348216 | train/value_loss = 37.94590861797333 | train/approx_kl = 0.008675235323607922 | train/clip_fraction = 0.0568359375 | train/loss = 13.926454544067383 | train/explained_variance = 0.040100157260894775 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:45: [Sb3-PPO[worker: 4]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 45.84 | rollout/ep_len_mean = 45.84 | time/fps = 73 | time/time_elapsed = 110 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6408562453463673 | train/policy_gradient_loss = -0.01784787670476362 | train/value_loss = 50.44305979013443 | train/approx_kl = 0.00931798666715622 | train/clip_fraction = 0.072705078125 | train/loss = 19.413591384887695 | train/explained_variance = 0.2960416078567505 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 4]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 45.84 | rollout/ep_len_mean = 45.84 | time/fps = 73 | time/time_elapsed = 110 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6408562453463673 | train/policy_gradient_loss = -0.01784787670476362 | train/value_loss = 50.44305979013443 | train/approx_kl = 0.00931798666715622 | train/clip_fraction = 0.072705078125 | train/loss = 19.413591384887695 | train/explained_variance = 0.2960416078567505 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:45: [Sb3-PPO[worker: 5]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 49.45 | rollout/ep_len_mean = 49.45 | time/fps = 73 | time/time_elapsed = 111 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6319150125607849 | train/policy_gradient_loss = -0.02007916850998299 | train/value_loss = 55.70775477290154 | train/approx_kl = 0.009037584997713566 | train/clip_fraction = 0.103125 | train/loss = 28.75029945373535 | train/explained_variance = 0.244806170463562 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 5]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 49.45 | rollout/ep_len_mean = 49.45 | time/fps = 73 | time/time_elapsed = 111 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6319150125607849 | train/policy_gradient_loss = -0.02007916850998299 | train/value_loss = 55.70775477290154 | train/approx_kl = 0.009037584997713566 | train/clip_fraction = 0.103125 | train/loss = 28.75029945373535 | train/explained_variance = 0.244806170463562 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:45: [Sb3-PPO[worker: 1]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 43.63 | rollout/ep_len_mean = 43.63 | time/fps = 73 | time/time_elapsed = 111 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6363960897549987 | train/policy_gradient_loss = -0.017436703966814092 | train/value_loss = 53.993737179040906 | train/approx_kl = 0.008265029639005661 | train/clip_fraction = 0.07626953125 | train/loss = 24.675281524658203 | train/explained_variance = 0.21792006492614746 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 1]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 43.63 | rollout/ep_len_mean = 43.63 | time/fps = 73 | time/time_elapsed = 111 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6363960897549987 | train/policy_gradient_loss = -0.017436703966814092 | train/value_loss = 53.993737179040906 | train/approx_kl = 0.008265029639005661 | train/clip_fraction = 0.07626953125 | train/loss = 24.675281524658203 | train/explained_variance = 0.21792006492614746 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:45: [Sb3-PPO[worker: 2]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 46.92 | rollout/ep_len_mean = 46.92 | time/fps = 73 | time/time_elapsed = 112 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6380577769130469 | train/policy_gradient_loss = -0.02162545901082922 | train/value_loss = 53.42646481990814 | train/approx_kl = 0.008550059050321579 | train/clip_fraction = 0.0958984375 | train/loss = 20.76878547668457 | train/explained_variance = 0.26671403646469116 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 2]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 46.92 | rollout/ep_len_mean = 46.92 | time/fps = 73 | time/time_elapsed = 112 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6380577769130469 | train/policy_gradient_loss = -0.02162545901082922 | train/value_loss = 53.42646481990814 | train/approx_kl = 0.008550059050321579 | train/clip_fraction = 0.0958984375 | train/loss = 20.76878547668457 | train/explained_variance = 0.26671403646469116 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:45: [Sb3-PPO[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 44.87 | rollout/ep_len_mean = 44.87 | time/fps = 72 | time/time_elapsed = 112 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6302148325368762 | train/policy_gradient_loss = -0.021342786648892796 | train/value_loss = 52.719143885374066 | train/approx_kl = 0.011043318547308445 | train/clip_fraction = 0.107470703125 | train/loss = 22.691471099853516 | train/explained_variance = 0.2502027153968811 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 44.87 | rollout/ep_len_mean = 44.87 | time/fps = 72 | time/time_elapsed = 112 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6302148325368762 | train/policy_gradient_loss = -0.021342786648892796 | train/value_loss = 52.719143885374066 | train/approx_kl = 0.011043318547308445 | train/clip_fraction = 0.107470703125 | train/loss = 22.691471099853516 | train/explained_variance = 0.2502027153968811 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:45: [Sb3-PPO[worker: 3]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 51.78 | rollout/ep_len_mean = 51.78 | time/fps = 72 | time/time_elapsed = 113 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6307819658890367 | train/policy_gradient_loss = -0.018105181005375927 | train/value_loss = 58.36709864735603 | train/approx_kl = 0.008882740512490273 | train/clip_fraction = 0.084912109375 | train/loss = 23.922922134399414 | train/explained_variance = 0.1844729781150818 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 3]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 51.78 | rollout/ep_len_mean = 51.78 | time/fps = 72 | time/time_elapsed = 113 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6307819658890367 | train/policy_gradient_loss = -0.018105181005375927 | train/value_loss = 58.36709864735603 | train/approx_kl = 0.008882740512490273 | train/clip_fraction = 0.084912109375 | train/loss = 23.922922134399414 | train/explained_variance = 0.1844729781150818 | train/n_updates = 30 | train/clip_range = 0.2 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:45:                  agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      6           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      6           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:45:                  agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      7           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      7           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:45:                  agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      8           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      8           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:45:                  agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      9           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                    Sb3-PPO      9           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:45: [Sb3-PPO[worker: 6]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.329787234042552 | rollout/ep_len_mean = 21.329787234042552 | time/fps = 144 | time/time_elapsed = 14 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 6]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.329787234042552 | rollout/ep_len_mean = 21.329787234042552 | time/fps = 144 | time/time_elapsed = 14 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:45: [Sb3-PPO[worker: 7]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.322916666666668 | rollout/ep_len_mean = 21.322916666666668 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 7]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.322916666666668 | rollout/ep_len_mean = 21.322916666666668 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:45: [Sb3-PPO[worker: 8]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.395604395604394 | rollout/ep_len_mean = 22.395604395604394 | time/fps = 160 | time/time_elapsed = 12 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 8]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.395604395604394 | rollout/ep_len_mean = 22.395604395604394 | time/fps = 160 | time/time_elapsed = 12 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:45: [Sb3-PPO[worker: 9]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 20.76530612244898 | rollout/ep_len_mean = 20.76530612244898 | time/fps = 167 | time/time_elapsed = 12 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 9]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 20.76530612244898 | rollout/ep_len_mean = 20.76530612244898 | time/fps = 167 | time/time_elapsed = 12 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 6]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.51 | rollout/ep_len_mean = 27.51 | time/fps = 123 | time/time_elapsed = 33 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6856363097205758 | train/policy_gradient_loss = -0.01955040743050631 | train/value_loss = 50.88062896132469 | train/approx_kl = 0.0089765265583992 | train/clip_fraction = 0.120947265625 | train/loss = 7.111379623413086 | train/explained_variance = -0.0021555423736572266 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 6]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.51 | rollout/ep_len_mean = 27.51 | time/fps = 123 | time/time_elapsed = 33 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6856363097205758 | train/policy_gradient_loss = -0.01955040743050631 | train/value_loss = 50.88062896132469 | train/approx_kl = 0.0089765265583992 | train/clip_fraction = 0.120947265625 | train/loss = 7.111379623413086 | train/explained_variance = -0.0021555423736572266 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 7]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.21 | rollout/ep_len_mean = 26.21 | time/fps = 126 | time/time_elapsed = 32 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6860836455598474 | train/policy_gradient_loss = -0.014052454034390394 | train/value_loss = 45.97430849969387 | train/approx_kl = 0.008620580658316612 | train/clip_fraction = 0.0908203125 | train/loss = 5.718726634979248 | train/explained_variance = 0.005562543869018555 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 7]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.21 | rollout/ep_len_mean = 26.21 | time/fps = 126 | time/time_elapsed = 32 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6860836455598474 | train/policy_gradient_loss = -0.014052454034390394 | train/value_loss = 45.97430849969387 | train/approx_kl = 0.008620580658316612 | train/clip_fraction = 0.0908203125 | train/loss = 5.718726634979248 | train/explained_variance = 0.005562543869018555 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 8]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.17 | rollout/ep_len_mean = 26.17 | time/fps = 128 | time/time_elapsed = 31 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6859566856175661 | train/policy_gradient_loss = -0.0166478871498839 | train/value_loss = 52.89832760989666 | train/approx_kl = 0.00855167768895626 | train/clip_fraction = 0.101025390625 | train/loss = 7.985254764556885 | train/explained_variance = 0.0052613019943237305 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 8]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.17 | rollout/ep_len_mean = 26.17 | time/fps = 128 | time/time_elapsed = 31 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6859566856175661 | train/policy_gradient_loss = -0.0166478871498839 | train/value_loss = 52.89832760989666 | train/approx_kl = 0.00855167768895626 | train/clip_fraction = 0.101025390625 | train/loss = 7.985254764556885 | train/explained_variance = 0.0052613019943237305 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 9]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 23.85 | rollout/ep_len_mean = 23.85 | time/fps = 131 | time/time_elapsed = 31 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6865209612995387 | train/policy_gradient_loss = -0.01159217145468574 | train/value_loss = 47.86252434551716 | train/approx_kl = 0.008216005750000477 | train/clip_fraction = 0.0794921875 | train/loss = 7.031304359436035 | train/explained_variance = 0.0007323622703552246 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 9]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 23.85 | rollout/ep_len_mean = 23.85 | time/fps = 131 | time/time_elapsed = 31 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6865209612995387 | train/policy_gradient_loss = -0.01159217145468574 | train/value_loss = 47.86252434551716 | train/approx_kl = 0.008216005750000477 | train/clip_fraction = 0.0794921875 | train/loss = 7.031304359436035 | train/explained_variance = 0.0007323622703552246 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 6]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 34.75 | rollout/ep_len_mean = 34.75 | time/fps = 118 | time/time_elapsed = 51 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6680946618318557 | train/policy_gradient_loss = -0.015845490466745105 | train/value_loss = 40.543832015991214 | train/approx_kl = 0.00919223204255104 | train/clip_fraction = 0.051318359375 | train/loss = 18.082538604736328 | train/explained_variance = 0.08105307817459106 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 6]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 34.75 | rollout/ep_len_mean = 34.75 | time/fps = 118 | time/time_elapsed = 51 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6680946618318557 | train/policy_gradient_loss = -0.015845490466745105 | train/value_loss = 40.543832015991214 | train/approx_kl = 0.00919223204255104 | train/clip_fraction = 0.051318359375 | train/loss = 18.082538604736328 | train/explained_variance = 0.08105307817459106 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 7]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.99 | rollout/ep_len_mean = 33.99 | time/fps = 120 | time/time_elapsed = 51 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6703325387090444 | train/policy_gradient_loss = -0.017256467255356255 | train/value_loss = 36.56526944041252 | train/approx_kl = 0.0104983514174819 | train/clip_fraction = 0.073046875 | train/loss = 10.88027286529541 | train/explained_variance = 0.10745066404342651 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 7]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.99 | rollout/ep_len_mean = 33.99 | time/fps = 120 | time/time_elapsed = 51 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6703325387090444 | train/policy_gradient_loss = -0.017256467255356255 | train/value_loss = 36.56526944041252 | train/approx_kl = 0.0104983514174819 | train/clip_fraction = 0.073046875 | train/loss = 10.88027286529541 | train/explained_variance = 0.10745066404342651 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 9]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 34.17 | rollout/ep_len_mean = 34.17 | time/fps = 122 | time/time_elapsed = 50 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6635754937306046 | train/policy_gradient_loss = -0.01833372879191302 | train/value_loss = 28.669793429970742 | train/approx_kl = 0.011582517996430397 | train/clip_fraction = 0.076171875 | train/loss = 10.092113494873047 | train/explained_variance = 0.12253385782241821 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 9]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 34.17 | rollout/ep_len_mean = 34.17 | time/fps = 122 | time/time_elapsed = 50 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6635754937306046 | train/policy_gradient_loss = -0.01833372879191302 | train/value_loss = 28.669793429970742 | train/approx_kl = 0.011582517996430397 | train/clip_fraction = 0.076171875 | train/loss = 10.092113494873047 | train/explained_variance = 0.12253385782241821 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 8]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 35.56 | rollout/ep_len_mean = 35.56 | time/fps = 121 | time/time_elapsed = 50 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6636657716706396 | train/policy_gradient_loss = -0.020468915918900165 | train/value_loss = 34.0020552277565 | train/approx_kl = 0.012107600457966328 | train/clip_fraction = 0.0880859375 | train/loss = 12.507184028625488 | train/explained_variance = 0.12036782503128052 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 8]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 35.56 | rollout/ep_len_mean = 35.56 | time/fps = 121 | time/time_elapsed = 50 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6636657716706396 | train/policy_gradient_loss = -0.020468915918900165 | train/value_loss = 34.0020552277565 | train/approx_kl = 0.012107600457966328 | train/clip_fraction = 0.0880859375 | train/loss = 12.507184028625488 | train/explained_variance = 0.12036782503128052 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 7]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 44.64 | rollout/ep_len_mean = 44.64 | time/fps = 115 | time/time_elapsed = 70 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.633265808224678 | train/policy_gradient_loss = -0.0230875588960771 | train/value_loss = 50.796407270431516 | train/approx_kl = 0.010280366986989975 | train/clip_fraction = 0.1154296875 | train/loss = 19.713285446166992 | train/explained_variance = 0.26903200149536133 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 7]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 44.64 | rollout/ep_len_mean = 44.64 | time/fps = 115 | time/time_elapsed = 70 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.633265808224678 | train/policy_gradient_loss = -0.0230875588960771 | train/value_loss = 50.796407270431516 | train/approx_kl = 0.010280366986989975 | train/clip_fraction = 0.1154296875 | train/loss = 19.713285446166992 | train/explained_variance = 0.26903200149536133 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 6]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 44.09 | rollout/ep_len_mean = 44.09 | time/fps = 114 | time/time_elapsed = 71 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.635085660405457 | train/policy_gradient_loss = -0.019524750619893894 | train/value_loss = 51.484919738769534 | train/approx_kl = 0.009435750544071198 | train/clip_fraction = 0.088720703125 | train/loss = 15.843541145324707 | train/explained_variance = 0.25749021768569946 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 6]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 44.09 | rollout/ep_len_mean = 44.09 | time/fps = 114 | time/time_elapsed = 71 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.635085660405457 | train/policy_gradient_loss = -0.019524750619893894 | train/value_loss = 51.484919738769534 | train/approx_kl = 0.009435750544071198 | train/clip_fraction = 0.088720703125 | train/loss = 15.843541145324707 | train/explained_variance = 0.25749021768569946 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 9]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 45.69 | rollout/ep_len_mean = 45.69 | time/fps = 117 | time/time_elapsed = 69 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6443713787943125 | train/policy_gradient_loss = -0.012846545978391077 | train/value_loss = 55.9754634976387 | train/approx_kl = 0.008374381810426712 | train/clip_fraction = 0.0560546875 | train/loss = 18.558977127075195 | train/explained_variance = 0.21047407388687134 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 9]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 45.69 | rollout/ep_len_mean = 45.69 | time/fps = 117 | time/time_elapsed = 69 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6443713787943125 | train/policy_gradient_loss = -0.012846545978391077 | train/value_loss = 55.9754634976387 | train/approx_kl = 0.008374381810426712 | train/clip_fraction = 0.0560546875 | train/loss = 18.558977127075195 | train/explained_variance = 0.21047407388687134 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:46: [Sb3-PPO[worker: 8]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 46.72 | rollout/ep_len_mean = 46.72 | time/fps = 116 | time/time_elapsed = 70 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6392249969765544 | train/policy_gradient_loss = -0.014018029568251222 | train/value_loss = 57.29294466972351 | train/approx_kl = 0.007265768945217133 | train/clip_fraction = 0.05751953125 | train/loss = 30.07738494873047 | train/explained_variance = 0.24170905351638794 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[Sb3-PPO[worker: 8]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 46.72 | rollout/ep_len_mean = 46.72 | time/fps = 116 | time/time_elapsed = 70 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6392249969765544 | train/policy_gradient_loss = -0.014018029568251222 | train/value_loss = 57.29294466972351 | train/approx_kl = 0.007265768945217133 | train/clip_fraction = 0.05751953125 | train/loss = 30.07738494873047 | train/explained_variance = 0.24170905351638794 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 14:47: ... trained! [0m
    INFO:rlberry_logger:... trained!
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:47: Saved ExperimentManager(Sb3-PPO) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(Sb3-PPO) using pickle.
    [38;21m[INFO] 14:47: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/Sb3-PPO_2024-04-03_14-38-22_219abedb/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/Sb3-PPO_2024-04-03_14-38-22_219abedb/manager_obj.pickle'


## Plot training and evaluation.

After training agents, a researcher can analyse two things. The training curves (usually smoothed and averaged over multiple seeds) and the evaluations (draw a trained agent at random from the fitted ones, and average its performance over some episode of the environment).


```python
from rlberry.manager import plot_writer_data

data = plot_writer_data([first_agent, second_agent], "rollout/ep_rew_mean")
```

    /usr/local/lib/python3.10/dist-packages/rlberry/manager/plotting.py:165: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data.loc[:, "n_simu"] = data["n_simu"].astype(int)
    /usr/local/lib/python3.10/dist-packages/rlberry/manager/plotting.py:165: DeprecationWarning: In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array. To retain the old behavior, use either `df[df.columns[i]] = newvals` or, if columns are non-unique, `df.isetitem(i, newvals)`
      data.loc[:, "n_simu"] = data["n_simu"].astype(int)




![png](doc1.png)




```python
from rlberry.manager import evaluate_agents

evaluate_agents([first_agent, second_agent])
```

    /usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)
    [38;21m[INFO] 14:47: Evaluating Sb3-A2C... [0m
    INFO:rlberry_logger:Evaluating Sb3-A2C...
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 14:47: Evaluating Sb3-PPO... [0m
    INFO:rlberry_logger:Evaluating Sb3-PPO...
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished





![png](doc2.png)







  <div id="df-d236c75f-11cc-405d-aea2-275c1e62f498" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sb3-A2C</th>
      <th>Sb3-PPO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>297.0</td>
      <td>500.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>408.0</td>
      <td>173.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>500.0</td>
      <td>500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>500.0</td>
      <td>320.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>186.0</td>
      <td>286.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d236c75f-11cc-405d-aea2-275c1e62f498')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

  <script>
    const buttonEl =
      document.querySelector('#df-d236c75f-11cc-405d-aea2-275c1e62f498 button.colab-df-convert');
    buttonEl.style.display =
      google.colab.kernel.accessAllowed ? 'block' : 'none';

    async function convertToInteractive(key) {
      const element = document.querySelector('#df-d236c75f-11cc-405d-aea2-275c1e62f498');
      const dataTable =
        await google.colab.kernel.invokeFunction('convertToInteractive',
                                                  [key], {});
      if (!dataTable) return;

      const docLinkHtml = 'Like what you see? Visit the ' +
        '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
        + ' to learn more about interactive tables.';
      element.innerHTML = '';
      dataTable['output_type'] = 'display_data';
      await google.colab.output.renderOutput(dataTable, element);
      const docLink = document.createElement('div');
      docLink.innerHTML = docLinkHtml;
      element.appendChild(docLink);
    }
  </script>
  </div>


<div id="df-de67d85c-0716-4ca4-b9fd-3d9fd0627e6f">
  <button class="colab-df-quickchart" onclick="quickchart('df-de67d85c-0716-4ca4-b9fd-3d9fd0627e6f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-de67d85c-0716-4ca4-b9fd-3d9fd0627e6f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




## Statistically sound comparison of agents

A posteriori, it is also important to not only  visualise performances but to actually compare them through test statistics.


```python
from rlberry.manager import compare_agents

compare_agents(
    [first_agent, second_agent],
    method="tukey_hsd",
    eval_function=None,
    n_simulations=50,
    alpha=0.05,
    B=10000,
    seed=None,
)
```

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished







  <div id="df-f3437c30-0350-4b04-8159-eea00a2d66ac" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Agent1 vs Agent2</th>
      <th>mean Agent1</th>
      <th>mean Agent2</th>
      <th>mean diff</th>
      <th>std diff</th>
      <th>decisions</th>
      <th>p-val</th>
      <th>significance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sb3-A2C vs Sb3-PPO</td>
      <td>295.525</td>
      <td>368.212</td>
      <td>-72.687</td>
      <td>186.19375</td>
      <td>accept</td>
      <td>0.171699</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f3437c30-0350-4b04-8159-eea00a2d66ac')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

  <script>
    const buttonEl =
      document.querySelector('#df-f3437c30-0350-4b04-8159-eea00a2d66ac button.colab-df-convert');
    buttonEl.style.display =
      google.colab.kernel.accessAllowed ? 'block' : 'none';

    async function convertToInteractive(key) {
      const element = document.querySelector('#df-f3437c30-0350-4b04-8159-eea00a2d66ac');
      const dataTable =
        await google.colab.kernel.invokeFunction('convertToInteractive',
                                                  [key], {});
      if (!dataTable) return;

      const docLinkHtml = 'Like what you see? Visit the ' +
        '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
        + ' to learn more about interactive tables.';
      element.innerHTML = '';
      dataTable['output_type'] = 'display_data';
      await google.colab.output.renderOutput(dataTable, element);
      const docLink = document.createElement('div');
      docLink.innerHTML = docLinkHtml;
      element.appendChild(docLink);
    }
  </script>
  </div>

  </div>
  </div>




Here the comparison algorithm accepts that the means of agents evaluations are different from a statistical point of view, i.e, one can conclude the PPO is better than A2C in average on CartPole-v1.

### Adaptive number of seeds during training with AdaStop

Finally, rather than fitting A2C and PPO 10 times each, it is possible to fit each agent as little as possible but still get a statistically valid comparison. In order to do so, instead of doing the training of the agent directly with n_fit=K, the number of indenpendent runs n_fit will be chosen adaptively with ```AdastopComparator```.


```python
from rlberry.manager import AdastopComparator

env_ctor, env_kwargs = gym_make, dict(id="CartPole-v1")

managers = [
    {
        "agent_class": StableBaselinesAgent,
        "train_env": (env_ctor, env_kwargs),
        "fit_budget": 1e4,
        "agent_name": "Sb3-A2C",
        "init_kwargs": {"algo_cls": A2C},
    },
    {
        "agent_class": StableBaselinesAgent,
        "train_env": (env_ctor, env_kwargs),
        "agent_name": "Sb3-PPO",
        "fit_budget": 1e4,
        "init_kwargs": {"algo_cls": PPO},
    },
]

comparator = AdastopComparator()
comparator.compare(managers)
print(comparator.managers_paths)
```

    [38;21m[INFO] 14:57: Running ExperimentManager fit() for A2C with n_fit = 5 and max_workers = None. [0m
    INFO:rlberry_logger:Running ExperimentManager fit() for A2C with n_fit = 5 and max_workers = None.
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:57:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        2           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        2           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:57:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        0           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        0           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:57:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        3           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        3           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:57:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        1           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        1           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:57:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        4           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        4           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:57: [A2C[worker: 2]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.028305411338806152 | train/entropy_loss = -0.5929406881332397 | train/policy_loss = 1.789804220199585 | train/value_loss = 9.144118309020996 | time/iterations = 100 | rollout/ep_rew_mean = 35.5 | rollout/ep_len_mean = 35.5 | time/fps = 82 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.028305411338806152 | train/entropy_loss = -0.5929406881332397 | train/policy_loss = 1.789804220199585 | train/value_loss = 9.144118309020996 | time/iterations = 100 | rollout/ep_rew_mean = 35.5 | rollout/ep_len_mean = 35.5 | time/fps = 82 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:57: [A2C[worker: 0]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.010118961334228516 | train/entropy_loss = -0.6710174679756165 | train/policy_loss = 1.6453092098236084 | train/value_loss = 7.85705041885376 | time/iterations = 100 | rollout/ep_rew_mean = 41.666666666666664 | rollout/ep_len_mean = 41.666666666666664 | time/fps = 79 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.010118961334228516 | train/entropy_loss = -0.6710174679756165 | train/policy_loss = 1.6453092098236084 | train/value_loss = 7.85705041885376 | time/iterations = 100 | rollout/ep_rew_mean = 41.666666666666664 | rollout/ep_len_mean = 41.666666666666664 | time/fps = 79 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:57: [A2C[worker: 3]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.07400065660476685 | train/entropy_loss = -0.5532559752464294 | train/policy_loss = 0.27084100246429443 | train/value_loss = 2.562013626098633 | time/iterations = 100 | rollout/ep_rew_mean = 28.41176470588235 | rollout/ep_len_mean = 28.41176470588235 | time/fps = 79 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.07400065660476685 | train/entropy_loss = -0.5532559752464294 | train/policy_loss = 0.27084100246429443 | train/value_loss = 2.562013626098633 | time/iterations = 100 | rollout/ep_rew_mean = 28.41176470588235 | rollout/ep_len_mean = 28.41176470588235 | time/fps = 79 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:57: [A2C[worker: 1]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.08095705509185791 | train/entropy_loss = -0.6755411028862 | train/policy_loss = 1.6535546779632568 | train/value_loss = 9.280721664428711 | time/iterations = 100 | rollout/ep_rew_mean = 36.61538461538461 | rollout/ep_len_mean = 36.61538461538461 | time/fps = 78 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.08095705509185791 | train/entropy_loss = -0.6755411028862 | train/policy_loss = 1.6535546779632568 | train/value_loss = 9.280721664428711 | time/iterations = 100 | rollout/ep_rew_mean = 36.61538461538461 | rollout/ep_len_mean = 36.61538461538461 | time/fps = 78 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:57: [A2C[worker: 4]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.4930112361907959 | train/entropy_loss = -0.6930809020996094 | train/policy_loss = 1.3694995641708374 | train/value_loss = 4.680158615112305 | time/iterations = 100 | rollout/ep_rew_mean = 19.84 | rollout/ep_len_mean = 19.84 | time/fps = 77 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.4930112361907959 | train/entropy_loss = -0.6930809020996094 | train/policy_loss = 1.3694995641708374 | train/value_loss = 4.680158615112305 | time/iterations = 100 | rollout/ep_rew_mean = 19.84 | rollout/ep_len_mean = 19.84 | time/fps = 77 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    [38;21m[INFO] 14:57: [A2C[worker: 2]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.01728665828704834 | train/entropy_loss = -0.6788516044616699 | train/policy_loss = 1.5419367551803589 | train/value_loss = 7.4378557205200195 | time/iterations = 200 | rollout/ep_rew_mean = 39.75 | rollout/ep_len_mean = 39.75 | time/fps = 82 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.01728665828704834 | train/entropy_loss = -0.6788516044616699 | train/policy_loss = 1.5419367551803589 | train/value_loss = 7.4378557205200195 | time/iterations = 200 | rollout/ep_rew_mean = 39.75 | rollout/ep_len_mean = 39.75 | time/fps = 82 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:57: [A2C[worker: 0]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.09108155965805054 | train/entropy_loss = -0.5774691104888916 | train/policy_loss = 1.3573901653289795 | train/value_loss = 7.399888038635254 | time/iterations = 200 | rollout/ep_rew_mean = 47.523809523809526 | rollout/ep_len_mean = 47.523809523809526 | time/fps = 80 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.09108155965805054 | train/entropy_loss = -0.5774691104888916 | train/policy_loss = 1.3573901653289795 | train/value_loss = 7.399888038635254 | time/iterations = 200 | rollout/ep_rew_mean = 47.523809523809526 | rollout/ep_len_mean = 47.523809523809526 | time/fps = 80 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:57: [A2C[worker: 3]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.13870251178741455 | train/entropy_loss = -0.497709184885025 | train/policy_loss = 1.0789985656738281 | train/value_loss = 8.075952529907227 | time/iterations = 200 | rollout/ep_rew_mean = 26.236842105263158 | rollout/ep_len_mean = 26.236842105263158 | time/fps = 79 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.13870251178741455 | train/entropy_loss = -0.497709184885025 | train/policy_loss = 1.0789985656738281 | train/value_loss = 8.075952529907227 | time/iterations = 200 | rollout/ep_rew_mean = 26.236842105263158 | rollout/ep_len_mean = 26.236842105263158 | time/fps = 79 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:57: [A2C[worker: 1]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.056650638580322266 | train/entropy_loss = -0.4899246096611023 | train/policy_loss = 2.0306453704833984 | train/value_loss = 7.291441917419434 | time/iterations = 200 | rollout/ep_rew_mean = 46.23809523809524 | rollout/ep_len_mean = 46.23809523809524 | time/fps = 78 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.056650638580322266 | train/entropy_loss = -0.4899246096611023 | train/policy_loss = 2.0306453704833984 | train/value_loss = 7.291441917419434 | time/iterations = 200 | rollout/ep_rew_mean = 46.23809523809524 | rollout/ep_len_mean = 46.23809523809524 | time/fps = 78 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:57: [A2C[worker: 4]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.008759856224060059 | train/entropy_loss = -0.6930942535400391 | train/policy_loss = 1.6664243936538696 | train/value_loss = 7.253000736236572 | time/iterations = 200 | rollout/ep_rew_mean = 22.976744186046513 | rollout/ep_len_mean = 22.976744186046513 | time/fps = 78 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.008759856224060059 | train/entropy_loss = -0.6930942535400391 | train/policy_loss = 1.6664243936538696 | train/value_loss = 7.253000736236572 | time/iterations = 200 | rollout/ep_rew_mean = 22.976744186046513 | rollout/ep_len_mean = 22.976744186046513 | time/fps = 78 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 14:57: [A2C[worker: 0]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.37607860565185547 | train/entropy_loss = -0.5371400117874146 | train/policy_loss = 1.692704439163208 | train/value_loss = 7.975527763366699 | time/iterations = 300 | rollout/ep_rew_mean = 43.11764705882353 | rollout/ep_len_mean = 43.11764705882353 | time/fps = 80 | time/time_elapsed = 18 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.37607860565185547 | train/entropy_loss = -0.5371400117874146 | train/policy_loss = 1.692704439163208 | train/value_loss = 7.975527763366699 | time/iterations = 300 | rollout/ep_rew_mean = 43.11764705882353 | rollout/ep_len_mean = 43.11764705882353 | time/fps = 80 | time/time_elapsed = 18 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:57: [A2C[worker: 2]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.031310081481933594 | train/entropy_loss = -0.615744948387146 | train/policy_loss = 1.9359557628631592 | train/value_loss = 6.445841312408447 | time/iterations = 300 | rollout/ep_rew_mean = 45.45454545454545 | rollout/ep_len_mean = 45.45454545454545 | time/fps = 80 | time/time_elapsed = 18 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.031310081481933594 | train/entropy_loss = -0.615744948387146 | train/policy_loss = 1.9359557628631592 | train/value_loss = 6.445841312408447 | time/iterations = 300 | rollout/ep_rew_mean = 45.45454545454545 | rollout/ep_len_mean = 45.45454545454545 | time/fps = 80 | time/time_elapsed = 18 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:57: [A2C[worker: 3]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.2188708782196045 | train/entropy_loss = -0.48533138632774353 | train/policy_loss = 1.6806062459945679 | train/value_loss = 7.451612949371338 | time/iterations = 300 | rollout/ep_rew_mean = 25.67241379310345 | rollout/ep_len_mean = 25.67241379310345 | time/fps = 80 | time/time_elapsed = 18 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.2188708782196045 | train/entropy_loss = -0.48533138632774353 | train/policy_loss = 1.6806062459945679 | train/value_loss = 7.451612949371338 | time/iterations = 300 | rollout/ep_rew_mean = 25.67241379310345 | rollout/ep_len_mean = 25.67241379310345 | time/fps = 80 | time/time_elapsed = 18 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:57: [A2C[worker: 1]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.029486358165740967 | train/entropy_loss = -0.5485534071922302 | train/policy_loss = 1.5417178869247437 | train/value_loss = 5.942726135253906 | time/iterations = 300 | rollout/ep_rew_mean = 47.70967741935484 | rollout/ep_len_mean = 47.70967741935484 | time/fps = 79 | time/time_elapsed = 18 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.029486358165740967 | train/entropy_loss = -0.5485534071922302 | train/policy_loss = 1.5417178869247437 | train/value_loss = 5.942726135253906 | time/iterations = 300 | rollout/ep_rew_mean = 47.70967741935484 | rollout/ep_len_mean = 47.70967741935484 | time/fps = 79 | time/time_elapsed = 18 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:57: [A2C[worker: 4]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.012317836284637451 | train/entropy_loss = -0.6823832988739014 | train/policy_loss = 1.7301744222640991 | train/value_loss = 6.8281450271606445 | time/iterations = 300 | rollout/ep_rew_mean = 24.557377049180328 | rollout/ep_len_mean = 24.557377049180328 | time/fps = 79 | time/time_elapsed = 18 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.012317836284637451 | train/entropy_loss = -0.6823832988739014 | train/policy_loss = 1.7301744222640991 | train/value_loss = 6.8281450271606445 | time/iterations = 300 | rollout/ep_rew_mean = 24.557377049180328 | rollout/ep_len_mean = 24.557377049180328 | time/fps = 79 | time/time_elapsed = 18 | time/total_timesteps = 1500 |
    [38;21m[INFO] 14:58: [A2C[worker: 3]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -1.058413028717041 | train/entropy_loss = -0.30318859219551086 | train/policy_loss = 0.8710079193115234 | train/value_loss = 11.344284057617188 | time/iterations = 400 | rollout/ep_rew_mean = 24.7375 | rollout/ep_len_mean = 24.7375 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -1.058413028717041 | train/entropy_loss = -0.30318859219551086 | train/policy_loss = 0.8710079193115234 | train/value_loss = 11.344284057617188 | time/iterations = 400 | rollout/ep_rew_mean = 24.7375 | rollout/ep_len_mean = 24.7375 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:58: [A2C[worker: 2]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.007683753967285156 | train/entropy_loss = -0.6121469736099243 | train/policy_loss = 1.7847130298614502 | train/value_loss = 5.6320109367370605 | time/iterations = 400 | rollout/ep_rew_mean = 49.0 | rollout/ep_len_mean = 49.0 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.007683753967285156 | train/entropy_loss = -0.6121469736099243 | train/policy_loss = 1.7847130298614502 | train/value_loss = 5.6320109367370605 | time/iterations = 400 | rollout/ep_rew_mean = 49.0 | rollout/ep_len_mean = 49.0 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:58: [A2C[worker: 0]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.07657104730606079 | train/entropy_loss = -0.6082916259765625 | train/policy_loss = 1.2245433330535889 | train/value_loss = 4.849370956420898 | time/iterations = 400 | rollout/ep_rew_mean = 42.56521739130435 | rollout/ep_len_mean = 42.56521739130435 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.07657104730606079 | train/entropy_loss = -0.6082916259765625 | train/policy_loss = 1.2245433330535889 | train/value_loss = 4.849370956420898 | time/iterations = 400 | rollout/ep_rew_mean = 42.56521739130435 | rollout/ep_len_mean = 42.56521739130435 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:58: [A2C[worker: 4]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.0023389458656311035 | train/entropy_loss = -0.6840308308601379 | train/policy_loss = 1.4429903030395508 | train/value_loss = 6.0909037590026855 | time/iterations = 400 | rollout/ep_rew_mean = 27.464788732394368 | rollout/ep_len_mean = 27.464788732394368 | time/fps = 79 | time/time_elapsed = 25 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.0023389458656311035 | train/entropy_loss = -0.6840308308601379 | train/policy_loss = 1.4429903030395508 | train/value_loss = 6.0909037590026855 | time/iterations = 400 | rollout/ep_rew_mean = 27.464788732394368 | rollout/ep_len_mean = 27.464788732394368 | time/fps = 79 | time/time_elapsed = 25 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:58: [A2C[worker: 1]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.0035207271575927734 | train/entropy_loss = -0.6493740081787109 | train/policy_loss = 0.975862979888916 | train/value_loss = 5.576460361480713 | time/iterations = 400 | rollout/ep_rew_mean = 46.166666666666664 | rollout/ep_len_mean = 46.166666666666664 | time/fps = 79 | time/time_elapsed = 25 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.0035207271575927734 | train/entropy_loss = -0.6493740081787109 | train/policy_loss = 0.975862979888916 | train/value_loss = 5.576460361480713 | time/iterations = 400 | rollout/ep_rew_mean = 46.166666666666664 | rollout/ep_len_mean = 46.166666666666664 | time/fps = 79 | time/time_elapsed = 25 | time/total_timesteps = 2000 |
    [38;21m[INFO] 14:58: [A2C[worker: 3]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.014823198318481445 | train/entropy_loss = -0.5995071530342102 | train/policy_loss = -9.691522598266602 | train/value_loss = 383.41552734375 | time/iterations = 500 | rollout/ep_rew_mean = 25.19191919191919 | rollout/ep_len_mean = 25.19191919191919 | time/fps = 81 | time/time_elapsed = 30 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.014823198318481445 | train/entropy_loss = -0.5995071530342102 | train/policy_loss = -9.691522598266602 | train/value_loss = 383.41552734375 | time/iterations = 500 | rollout/ep_rew_mean = 25.19191919191919 | rollout/ep_len_mean = 25.19191919191919 | time/fps = 81 | time/time_elapsed = 30 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:58: [A2C[worker: 0]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.11124038696289062 | train/entropy_loss = -0.530320405960083 | train/policy_loss = 1.9482929706573486 | train/value_loss = 5.676708221435547 | time/iterations = 500 | rollout/ep_rew_mean = 45.056603773584904 | rollout/ep_len_mean = 45.056603773584904 | time/fps = 80 | time/time_elapsed = 31 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.11124038696289062 | train/entropy_loss = -0.530320405960083 | train/policy_loss = 1.9482929706573486 | train/value_loss = 5.676708221435547 | time/iterations = 500 | rollout/ep_rew_mean = 45.056603773584904 | rollout/ep_len_mean = 45.056603773584904 | time/fps = 80 | time/time_elapsed = 31 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:58: [A2C[worker: 4]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.0054686665534973145 | train/entropy_loss = -0.5380524396896362 | train/policy_loss = -21.372669219970703 | train/value_loss = 559.7186889648438 | time/iterations = 500 | rollout/ep_rew_mean = 31.175 | rollout/ep_len_mean = 31.175 | time/fps = 80 | time/time_elapsed = 31 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.0054686665534973145 | train/entropy_loss = -0.5380524396896362 | train/policy_loss = -21.372669219970703 | train/value_loss = 559.7186889648438 | time/iterations = 500 | rollout/ep_rew_mean = 31.175 | rollout/ep_len_mean = 31.175 | time/fps = 80 | time/time_elapsed = 31 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:58: [A2C[worker: 2]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.00229799747467041 | train/entropy_loss = -0.6244776844978333 | train/policy_loss = 1.2794867753982544 | train/value_loss = 4.9674973487854 | time/iterations = 500 | rollout/ep_rew_mean = 52.80434782608695 | rollout/ep_len_mean = 52.80434782608695 | time/fps = 80 | time/time_elapsed = 31 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.00229799747467041 | train/entropy_loss = -0.6244776844978333 | train/policy_loss = 1.2794867753982544 | train/value_loss = 4.9674973487854 | time/iterations = 500 | rollout/ep_rew_mean = 52.80434782608695 | rollout/ep_len_mean = 52.80434782608695 | time/fps = 80 | time/time_elapsed = 31 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:58: [A2C[worker: 1]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.00014781951904296875 | train/entropy_loss = -0.5360603332519531 | train/policy_loss = 0.7718409299850464 | train/value_loss = 4.969958782196045 | time/iterations = 500 | rollout/ep_rew_mean = 50.3469387755102 | rollout/ep_len_mean = 50.3469387755102 | time/fps = 79 | time/time_elapsed = 31 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.00014781951904296875 | train/entropy_loss = -0.5360603332519531 | train/policy_loss = 0.7718409299850464 | train/value_loss = 4.969958782196045 | time/iterations = 500 | rollout/ep_rew_mean = 50.3469387755102 | rollout/ep_len_mean = 50.3469387755102 | time/fps = 79 | time/time_elapsed = 31 | time/total_timesteps = 2500 |
    [38;21m[INFO] 14:58: [A2C[worker: 3]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.004991650581359863 | train/entropy_loss = -0.6047713160514832 | train/policy_loss = 0.9499195218086243 | train/value_loss = 5.69765567779541 | time/iterations = 600 | rollout/ep_rew_mean = 25.75 | rollout/ep_len_mean = 25.75 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.004991650581359863 | train/entropy_loss = -0.6047713160514832 | train/policy_loss = 0.9499195218086243 | train/value_loss = 5.69765567779541 | time/iterations = 600 | rollout/ep_rew_mean = 25.75 | rollout/ep_len_mean = 25.75 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:58: [A2C[worker: 4]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0014711618423461914 | train/entropy_loss = -0.6609852910041809 | train/policy_loss = 1.22150719165802 | train/value_loss = 4.820957660675049 | time/iterations = 600 | rollout/ep_rew_mean = 33.07865168539326 | rollout/ep_len_mean = 33.07865168539326 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0014711618423461914 | train/entropy_loss = -0.6609852910041809 | train/policy_loss = 1.22150719165802 | train/value_loss = 4.820957660675049 | time/iterations = 600 | rollout/ep_rew_mean = 33.07865168539326 | rollout/ep_len_mean = 33.07865168539326 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:58: [A2C[worker: 0]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.010740995407104492 | train/entropy_loss = -0.5324840545654297 | train/policy_loss = 1.0560725927352905 | train/value_loss = 4.689620018005371 | time/iterations = 600 | rollout/ep_rew_mean = 49.2 | rollout/ep_len_mean = 49.2 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.010740995407104492 | train/entropy_loss = -0.5324840545654297 | train/policy_loss = 1.0560725927352905 | train/value_loss = 4.689620018005371 | time/iterations = 600 | rollout/ep_rew_mean = 49.2 | rollout/ep_len_mean = 49.2 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:58: [A2C[worker: 2]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0008081197738647461 | train/entropy_loss = -0.6198944449424744 | train/policy_loss = -6.1380414962768555 | train/value_loss = 245.4508514404297 | time/iterations = 600 | rollout/ep_rew_mean = 56.43396226415094 | rollout/ep_len_mean = 56.43396226415094 | time/fps = 79 | time/time_elapsed = 37 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0008081197738647461 | train/entropy_loss = -0.6198944449424744 | train/policy_loss = -6.1380414962768555 | train/value_loss = 245.4508514404297 | time/iterations = 600 | rollout/ep_rew_mean = 56.43396226415094 | rollout/ep_len_mean = 56.43396226415094 | time/fps = 79 | time/time_elapsed = 37 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:58: [A2C[worker: 1]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.0011553764343261719 | train/entropy_loss = -0.4471665918827057 | train/policy_loss = 0.5887033939361572 | train/value_loss = 4.394412994384766 | time/iterations = 600 | rollout/ep_rew_mean = 52.5 | rollout/ep_len_mean = 52.5 | time/fps = 79 | time/time_elapsed = 37 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.0011553764343261719 | train/entropy_loss = -0.4471665918827057 | train/policy_loss = 0.5887033939361572 | train/value_loss = 4.394412994384766 | time/iterations = 600 | rollout/ep_rew_mean = 52.5 | rollout/ep_len_mean = 52.5 | time/fps = 79 | time/time_elapsed = 37 | time/total_timesteps = 3000 |
    [38;21m[INFO] 14:58: [A2C[worker: 3]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.004534006118774414 | train/entropy_loss = -0.5856071710586548 | train/policy_loss = 0.89299076795578 | train/value_loss = 5.199253559112549 | time/iterations = 700 | rollout/ep_rew_mean = 25.66 | rollout/ep_len_mean = 25.66 | time/fps = 81 | time/time_elapsed = 43 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.004534006118774414 | train/entropy_loss = -0.5856071710586548 | train/policy_loss = 0.89299076795578 | train/value_loss = 5.199253559112549 | time/iterations = 700 | rollout/ep_rew_mean = 25.66 | rollout/ep_len_mean = 25.66 | time/fps = 81 | time/time_elapsed = 43 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:58: [A2C[worker: 4]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.00011897087097167969 | train/entropy_loss = -0.6292620897293091 | train/policy_loss = 0.8634392619132996 | train/value_loss = 4.284634590148926 | time/iterations = 700 | rollout/ep_rew_mean = 35.90721649484536 | rollout/ep_len_mean = 35.90721649484536 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.00011897087097167969 | train/entropy_loss = -0.6292620897293091 | train/policy_loss = 0.8634392619132996 | train/value_loss = 4.284634590148926 | time/iterations = 700 | rollout/ep_rew_mean = 35.90721649484536 | rollout/ep_len_mean = 35.90721649484536 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:58: [A2C[worker: 0]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.008479416370391846 | train/entropy_loss = -0.6322427988052368 | train/policy_loss = 0.9408775568008423 | train/value_loss = 4.19278621673584 | time/iterations = 700 | rollout/ep_rew_mean = 50.30882352941177 | rollout/ep_len_mean = 50.30882352941177 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.008479416370391846 | train/entropy_loss = -0.6322427988052368 | train/policy_loss = 0.9408775568008423 | train/value_loss = 4.19278621673584 | time/iterations = 700 | rollout/ep_rew_mean = 50.30882352941177 | rollout/ep_len_mean = 50.30882352941177 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:58: [A2C[worker: 1]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 6.920099258422852e-05 | train/entropy_loss = -0.3688517212867737 | train/policy_loss = 0.48567304015159607 | train/value_loss = 3.874439239501953 | time/iterations = 700 | rollout/ep_rew_mean = 55.38095238095238 | rollout/ep_len_mean = 55.38095238095238 | time/fps = 79 | time/time_elapsed = 43 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 6.920099258422852e-05 | train/entropy_loss = -0.3688517212867737 | train/policy_loss = 0.48567304015159607 | train/value_loss = 3.874439239501953 | time/iterations = 700 | rollout/ep_rew_mean = 55.38095238095238 | rollout/ep_len_mean = 55.38095238095238 | time/fps = 79 | time/time_elapsed = 43 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:58: [A2C[worker: 2]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.00324857234954834 | train/entropy_loss = -0.5817568898200989 | train/policy_loss = 1.0730394124984741 | train/value_loss = 3.838426113128662 | time/iterations = 700 | rollout/ep_rew_mean = 59.719298245614034 | rollout/ep_len_mean = 59.719298245614034 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.00324857234954834 | train/entropy_loss = -0.5817568898200989 | train/policy_loss = 1.0730394124984741 | train/value_loss = 3.838426113128662 | time/iterations = 700 | rollout/ep_rew_mean = 59.719298245614034 | rollout/ep_len_mean = 59.719298245614034 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |
    [38;21m[INFO] 14:58: [A2C[worker: 3]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.011298298835754395 | train/entropy_loss = -0.6282132863998413 | train/policy_loss = 0.8669827580451965 | train/value_loss = 4.641777515411377 | time/iterations = 800 | rollout/ep_rew_mean = 27.82 | rollout/ep_len_mean = 27.82 | time/fps = 81 | time/time_elapsed = 49 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.011298298835754395 | train/entropy_loss = -0.6282132863998413 | train/policy_loss = 0.8669827580451965 | train/value_loss = 4.641777515411377 | time/iterations = 800 | rollout/ep_rew_mean = 27.82 | rollout/ep_len_mean = 27.82 | time/fps = 81 | time/time_elapsed = 49 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:58: [A2C[worker: 4]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 9.053945541381836e-05 | train/entropy_loss = -0.5238251686096191 | train/policy_loss = 1.0659483671188354 | train/value_loss = 3.7420761585235596 | time/iterations = 800 | rollout/ep_rew_mean = 38.25 | rollout/ep_len_mean = 38.25 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 9.053945541381836e-05 | train/entropy_loss = -0.5238251686096191 | train/policy_loss = 1.0659483671188354 | train/value_loss = 3.7420761585235596 | time/iterations = 800 | rollout/ep_rew_mean = 38.25 | rollout/ep_len_mean = 38.25 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:58: [A2C[worker: 1]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.00013428926467895508 | train/entropy_loss = -0.46073395013809204 | train/policy_loss = 0.7376588582992554 | train/value_loss = 3.3946220874786377 | time/iterations = 800 | rollout/ep_rew_mean = 55.7887323943662 | rollout/ep_len_mean = 55.7887323943662 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.00013428926467895508 | train/entropy_loss = -0.46073395013809204 | train/policy_loss = 0.7376588582992554 | train/value_loss = 3.3946220874786377 | time/iterations = 800 | rollout/ep_rew_mean = 55.7887323943662 | rollout/ep_len_mean = 55.7887323943662 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:58: [A2C[worker: 0]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0031080245971679688 | train/entropy_loss = -0.6596562266349792 | train/policy_loss = 0.9285008311271667 | train/value_loss = 3.6597683429718018 | time/iterations = 800 | rollout/ep_rew_mean = 53.54054054054054 | rollout/ep_len_mean = 53.54054054054054 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0031080245971679688 | train/entropy_loss = -0.6596562266349792 | train/policy_loss = 0.9285008311271667 | train/value_loss = 3.6597683429718018 | time/iterations = 800 | rollout/ep_rew_mean = 53.54054054054054 | rollout/ep_len_mean = 53.54054054054054 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:58: [A2C[worker: 2]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.0033981800079345703 | train/entropy_loss = -0.6527501344680786 | train/policy_loss = 0.7537058591842651 | train/value_loss = 3.318408966064453 | time/iterations = 800 | rollout/ep_rew_mean = 63.67213114754098 | rollout/ep_len_mean = 63.67213114754098 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.0033981800079345703 | train/entropy_loss = -0.6527501344680786 | train/policy_loss = 0.7537058591842651 | train/value_loss = 3.318408966064453 | time/iterations = 800 | rollout/ep_rew_mean = 63.67213114754098 | rollout/ep_len_mean = 63.67213114754098 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |
    [38;21m[INFO] 14:58: [A2C[worker: 3]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0044730305671691895 | train/entropy_loss = -0.5134053230285645 | train/policy_loss = 0.8648560643196106 | train/value_loss = 4.184412956237793 | time/iterations = 900 | rollout/ep_rew_mean = 30.38 | rollout/ep_len_mean = 30.38 | time/fps = 81 | time/time_elapsed = 55 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0044730305671691895 | train/entropy_loss = -0.5134053230285645 | train/policy_loss = 0.8648560643196106 | train/value_loss = 4.184412956237793 | time/iterations = 900 | rollout/ep_rew_mean = 30.38 | rollout/ep_len_mean = 30.38 | time/fps = 81 | time/time_elapsed = 55 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:58: [A2C[worker: 4]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0003235340118408203 | train/entropy_loss = -0.6430960297584534 | train/policy_loss = 0.925168514251709 | train/value_loss = 3.252087116241455 | time/iterations = 900 | rollout/ep_rew_mean = 43.07 | rollout/ep_len_mean = 43.07 | time/fps = 80 | time/time_elapsed = 55 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0003235340118408203 | train/entropy_loss = -0.6430960297584534 | train/policy_loss = 0.925168514251709 | train/value_loss = 3.252087116241455 | time/iterations = 900 | rollout/ep_rew_mean = 43.07 | rollout/ep_len_mean = 43.07 | time/fps = 80 | time/time_elapsed = 55 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:58: [A2C[worker: 0]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.004901289939880371 | train/entropy_loss = -0.38004499673843384 | train/policy_loss = 1.321666955947876 | train/value_loss = 3.1557953357696533 | time/iterations = 900 | rollout/ep_rew_mean = 56.44303797468354 | rollout/ep_len_mean = 56.44303797468354 | time/fps = 80 | time/time_elapsed = 56 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.004901289939880371 | train/entropy_loss = -0.38004499673843384 | train/policy_loss = 1.321666955947876 | train/value_loss = 3.1557953357696533 | time/iterations = 900 | rollout/ep_rew_mean = 56.44303797468354 | rollout/ep_len_mean = 56.44303797468354 | time/fps = 80 | time/time_elapsed = 56 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:58: [A2C[worker: 1]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0024073123931884766 | train/entropy_loss = -0.23515991866588593 | train/policy_loss = 1.8878180980682373 | train/value_loss = 2.9200401306152344 | time/iterations = 900 | rollout/ep_rew_mean = 58.828947368421055 | rollout/ep_len_mean = 58.828947368421055 | time/fps = 80 | time/time_elapsed = 56 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0024073123931884766 | train/entropy_loss = -0.23515991866588593 | train/policy_loss = 1.8878180980682373 | train/value_loss = 2.9200401306152344 | time/iterations = 900 | rollout/ep_rew_mean = 58.828947368421055 | rollout/ep_len_mean = 58.828947368421055 | time/fps = 80 | time/time_elapsed = 56 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:58: [A2C[worker: 2]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.00017267465591430664 | train/entropy_loss = -0.6740339994430542 | train/policy_loss = -9.478140830993652 | train/value_loss = 874.0905151367188 | time/iterations = 900 | rollout/ep_rew_mean = 66.05882352941177 | rollout/ep_len_mean = 66.05882352941177 | time/fps = 80 | time/time_elapsed = 56 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.00017267465591430664 | train/entropy_loss = -0.6740339994430542 | train/policy_loss = -9.478140830993652 | train/value_loss = 874.0905151367188 | time/iterations = 900 | rollout/ep_rew_mean = 66.05882352941177 | rollout/ep_len_mean = 66.05882352941177 | time/fps = 80 | time/time_elapsed = 56 | time/total_timesteps = 4500 |
    [38;21m[INFO] 14:58: [A2C[worker: 3]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.00027680397033691406 | train/entropy_loss = -0.5564754605293274 | train/policy_loss = 1.0853664875030518 | train/value_loss = 3.7666122913360596 | time/iterations = 1000 | rollout/ep_rew_mean = 33.42 | rollout/ep_len_mean = 33.42 | time/fps = 80 | time/time_elapsed = 61 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.00027680397033691406 | train/entropy_loss = -0.5564754605293274 | train/policy_loss = 1.0853664875030518 | train/value_loss = 3.7666122913360596 | time/iterations = 1000 | rollout/ep_rew_mean = 33.42 | rollout/ep_len_mean = 33.42 | time/fps = 80 | time/time_elapsed = 61 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:58: [A2C[worker: 4]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 7.086992263793945e-05 | train/entropy_loss = -0.6654736399650574 | train/policy_loss = 0.8892915844917297 | train/value_loss = 2.81494402885437 | time/iterations = 1000 | rollout/ep_rew_mean = 46.94 | rollout/ep_len_mean = 46.94 | time/fps = 80 | time/time_elapsed = 61 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 7.086992263793945e-05 | train/entropy_loss = -0.6654736399650574 | train/policy_loss = 0.8892915844917297 | train/value_loss = 2.81494402885437 | time/iterations = 1000 | rollout/ep_rew_mean = 46.94 | rollout/ep_len_mean = 46.94 | time/fps = 80 | time/time_elapsed = 61 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:58: [A2C[worker: 0]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.001016855239868164 | train/entropy_loss = -0.5300703048706055 | train/policy_loss = 0.7983027696609497 | train/value_loss = 2.7042152881622314 | time/iterations = 1000 | rollout/ep_rew_mean = 58.01176470588236 | rollout/ep_len_mean = 58.01176470588236 | time/fps = 80 | time/time_elapsed = 62 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.001016855239868164 | train/entropy_loss = -0.5300703048706055 | train/policy_loss = 0.7983027696609497 | train/value_loss = 2.7042152881622314 | time/iterations = 1000 | rollout/ep_rew_mean = 58.01176470588236 | rollout/ep_len_mean = 58.01176470588236 | time/fps = 80 | time/time_elapsed = 62 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:58: [A2C[worker: 2]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.000759124755859375 | train/entropy_loss = -0.45209187269210815 | train/policy_loss = -24.4777774810791 | train/value_loss = 2429.231689453125 | time/iterations = 1000 | rollout/ep_rew_mean = 67.5 | rollout/ep_len_mean = 67.5 | time/fps = 80 | time/time_elapsed = 62 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.000759124755859375 | train/entropy_loss = -0.45209187269210815 | train/policy_loss = -24.4777774810791 | train/value_loss = 2429.231689453125 | time/iterations = 1000 | rollout/ep_rew_mean = 67.5 | rollout/ep_len_mean = 67.5 | time/fps = 80 | time/time_elapsed = 62 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:58: [A2C[worker: 1]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.0014405250549316406 | train/entropy_loss = -0.47017431259155273 | train/policy_loss = 0.43397268652915955 | train/value_loss = 2.4856960773468018 | time/iterations = 1000 | rollout/ep_rew_mean = 59.96341463414634 | rollout/ep_len_mean = 59.96341463414634 | time/fps = 80 | time/time_elapsed = 62 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.0014405250549316406 | train/entropy_loss = -0.47017431259155273 | train/policy_loss = 0.43397268652915955 | train/value_loss = 2.4856960773468018 | time/iterations = 1000 | rollout/ep_rew_mean = 59.96341463414634 | rollout/ep_len_mean = 59.96341463414634 | time/fps = 80 | time/time_elapsed = 62 | time/total_timesteps = 5000 |
    [38;21m[INFO] 14:58: [A2C[worker: 4]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0008807182312011719 | train/entropy_loss = -0.6374331116676331 | train/policy_loss = 0.8517583608627319 | train/value_loss = 2.393681764602661 | time/iterations = 1100 | rollout/ep_rew_mean = 50.05 | rollout/ep_len_mean = 50.05 | time/fps = 80 | time/time_elapsed = 68 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0008807182312011719 | train/entropy_loss = -0.6374331116676331 | train/policy_loss = 0.8517583608627319 | train/value_loss = 2.393681764602661 | time/iterations = 1100 | rollout/ep_rew_mean = 50.05 | rollout/ep_len_mean = 50.05 | time/fps = 80 | time/time_elapsed = 68 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:58: [A2C[worker: 3]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0010949969291687012 | train/entropy_loss = -0.569749116897583 | train/policy_loss = 0.6180968284606934 | train/value_loss = 3.141446828842163 | time/iterations = 1100 | rollout/ep_rew_mean = 37.23 | rollout/ep_len_mean = 37.23 | time/fps = 80 | time/time_elapsed = 68 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0010949969291687012 | train/entropy_loss = -0.569749116897583 | train/policy_loss = 0.6180968284606934 | train/value_loss = 3.141446828842163 | time/iterations = 1100 | rollout/ep_rew_mean = 37.23 | rollout/ep_len_mean = 37.23 | time/fps = 80 | time/time_elapsed = 68 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:58: [A2C[worker: 0]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.00038558244705200195 | train/entropy_loss = -0.5621659159660339 | train/policy_loss = 0.6359560489654541 | train/value_loss = 2.2663376331329346 | time/iterations = 1100 | rollout/ep_rew_mean = 59.97752808988764 | rollout/ep_len_mean = 59.97752808988764 | time/fps = 79 | time/time_elapsed = 68 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.00038558244705200195 | train/entropy_loss = -0.5621659159660339 | train/policy_loss = 0.6359560489654541 | train/value_loss = 2.2663376331329346 | time/iterations = 1100 | rollout/ep_rew_mean = 59.97752808988764 | rollout/ep_len_mean = 59.97752808988764 | time/fps = 79 | time/time_elapsed = 68 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:58: [A2C[worker: 2]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0005648136138916016 | train/entropy_loss = -0.5196690559387207 | train/policy_loss = 1.099816083908081 | train/value_loss = 2.043062448501587 | time/iterations = 1100 | rollout/ep_rew_mean = 68.0253164556962 | rollout/ep_len_mean = 68.0253164556962 | time/fps = 79 | time/time_elapsed = 68 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0005648136138916016 | train/entropy_loss = -0.5196690559387207 | train/policy_loss = 1.099816083908081 | train/value_loss = 2.043062448501587 | time/iterations = 1100 | rollout/ep_rew_mean = 68.0253164556962 | rollout/ep_len_mean = 68.0253164556962 | time/fps = 79 | time/time_elapsed = 68 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:58: [A2C[worker: 1]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -4.696846008300781e-05 | train/entropy_loss = -0.4980483055114746 | train/policy_loss = 0.40066057443618774 | train/value_loss = 2.0641415119171143 | time/iterations = 1100 | rollout/ep_rew_mean = 64.16470588235295 | rollout/ep_len_mean = 64.16470588235295 | time/fps = 79 | time/time_elapsed = 69 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -4.696846008300781e-05 | train/entropy_loss = -0.4980483055114746 | train/policy_loss = 0.40066057443618774 | train/value_loss = 2.0641415119171143 | time/iterations = 1100 | rollout/ep_rew_mean = 64.16470588235295 | rollout/ep_len_mean = 64.16470588235295 | time/fps = 79 | time/time_elapsed = 69 | time/total_timesteps = 5500 |
    [38;21m[INFO] 14:58: [A2C[worker: 3]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.001431584358215332 | train/entropy_loss = -0.565502941608429 | train/policy_loss = 0.6797980070114136 | train/value_loss = 2.6637799739837646 | time/iterations = 1200 | rollout/ep_rew_mean = 41.45 | rollout/ep_len_mean = 41.45 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.001431584358215332 | train/entropy_loss = -0.565502941608429 | train/policy_loss = 0.6797980070114136 | train/value_loss = 2.6637799739837646 | time/iterations = 1200 | rollout/ep_rew_mean = 41.45 | rollout/ep_len_mean = 41.45 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:58: [A2C[worker: 4]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.0005317926406860352 | train/entropy_loss = -0.5315011739730835 | train/policy_loss = 0.8301931619644165 | train/value_loss = 1.9870036840438843 | time/iterations = 1200 | rollout/ep_rew_mean = 53.67 | rollout/ep_len_mean = 53.67 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.0005317926406860352 | train/entropy_loss = -0.5315011739730835 | train/policy_loss = 0.8301931619644165 | train/value_loss = 1.9870036840438843 | time/iterations = 1200 | rollout/ep_rew_mean = 53.67 | rollout/ep_len_mean = 53.67 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:58: [A2C[worker: 0]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -8.881092071533203e-05 | train/entropy_loss = -0.4968135356903076 | train/policy_loss = 0.8614799380302429 | train/value_loss = 1.8727582693099976 | time/iterations = 1200 | rollout/ep_rew_mean = 63.56989247311828 | rollout/ep_len_mean = 63.56989247311828 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -8.881092071533203e-05 | train/entropy_loss = -0.4968135356903076 | train/policy_loss = 0.8614799380302429 | train/value_loss = 1.8727582693099976 | time/iterations = 1200 | rollout/ep_rew_mean = 63.56989247311828 | rollout/ep_len_mean = 63.56989247311828 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:58: [A2C[worker: 2]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 9.560585021972656e-05 | train/entropy_loss = -0.6548899412155151 | train/policy_loss = 0.5850970149040222 | train/value_loss = 1.655248999595642 | time/iterations = 1200 | rollout/ep_rew_mean = 70.69135802469135 | rollout/ep_len_mean = 70.69135802469135 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 9.560585021972656e-05 | train/entropy_loss = -0.6548899412155151 | train/policy_loss = 0.5850970149040222 | train/value_loss = 1.655248999595642 | time/iterations = 1200 | rollout/ep_rew_mean = 70.69135802469135 | rollout/ep_len_mean = 70.69135802469135 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:58: [A2C[worker: 1]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 3.3020973205566406e-05 | train/entropy_loss = -0.4886438250541687 | train/policy_loss = 0.3137657046318054 | train/value_loss = 1.6837728023529053 | time/iterations = 1200 | rollout/ep_rew_mean = 67.47727272727273 | rollout/ep_len_mean = 67.47727272727273 | time/fps = 79 | time/time_elapsed = 75 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 3.3020973205566406e-05 | train/entropy_loss = -0.4886438250541687 | train/policy_loss = 0.3137657046318054 | train/value_loss = 1.6837728023529053 | time/iterations = 1200 | rollout/ep_rew_mean = 67.47727272727273 | rollout/ep_len_mean = 67.47727272727273 | time/fps = 79 | time/time_elapsed = 75 | time/total_timesteps = 6000 |
    [38;21m[INFO] 14:58: [A2C[worker: 3]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.00012981891632080078 | train/entropy_loss = -0.49956122040748596 | train/policy_loss = 0.5864171981811523 | train/value_loss = 2.241668939590454 | time/iterations = 1300 | rollout/ep_rew_mean = 46.79 | rollout/ep_len_mean = 46.79 | time/fps = 81 | time/time_elapsed = 80 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.00012981891632080078 | train/entropy_loss = -0.49956122040748596 | train/policy_loss = 0.5864171981811523 | train/value_loss = 2.241668939590454 | time/iterations = 1300 | rollout/ep_rew_mean = 46.79 | rollout/ep_len_mean = 46.79 | time/fps = 81 | time/time_elapsed = 80 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:58: [A2C[worker: 4]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.0009091496467590332 | train/entropy_loss = -0.637565016746521 | train/policy_loss = 0.5402558445930481 | train/value_loss = 1.6134631633758545 | time/iterations = 1300 | rollout/ep_rew_mean = 58.67 | rollout/ep_len_mean = 58.67 | time/fps = 81 | time/time_elapsed = 80 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.0009091496467590332 | train/entropy_loss = -0.637565016746521 | train/policy_loss = 0.5402558445930481 | train/value_loss = 1.6134631633758545 | time/iterations = 1300 | rollout/ep_rew_mean = 58.67 | rollout/ep_len_mean = 58.67 | time/fps = 81 | time/time_elapsed = 80 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:58: [A2C[worker: 0]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -2.110004425048828e-05 | train/entropy_loss = -0.5229418873786926 | train/policy_loss = 0.3962555527687073 | train/value_loss = 1.5082709789276123 | time/iterations = 1300 | rollout/ep_rew_mean = 66.47916666666667 | rollout/ep_len_mean = 66.47916666666667 | time/fps = 80 | time/time_elapsed = 80 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -2.110004425048828e-05 | train/entropy_loss = -0.5229418873786926 | train/policy_loss = 0.3962555527687073 | train/value_loss = 1.5082709789276123 | time/iterations = 1300 | rollout/ep_rew_mean = 66.47916666666667 | rollout/ep_len_mean = 66.47916666666667 | time/fps = 80 | time/time_elapsed = 80 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:58: [A2C[worker: 2]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.0006992816925048828 | train/entropy_loss = -0.2886084318161011 | train/policy_loss = 1.3875987529754639 | train/value_loss = 1.3070732355117798 | time/iterations = 1300 | rollout/ep_rew_mean = 77.35714285714286 | rollout/ep_len_mean = 77.35714285714286 | time/fps = 80 | time/time_elapsed = 80 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.0006992816925048828 | train/entropy_loss = -0.2886084318161011 | train/policy_loss = 1.3875987529754639 | train/value_loss = 1.3070732355117798 | time/iterations = 1300 | rollout/ep_rew_mean = 77.35714285714286 | rollout/ep_len_mean = 77.35714285714286 | time/fps = 80 | time/time_elapsed = 80 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:59: [A2C[worker: 1]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.00015050172805786133 | train/entropy_loss = -0.39929869771003723 | train/policy_loss = 0.5110852122306824 | train/value_loss = 1.3407256603240967 | time/iterations = 1300 | rollout/ep_rew_mean = 70.47252747252747 | rollout/ep_len_mean = 70.47252747252747 | time/fps = 80 | time/time_elapsed = 81 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.00015050172805786133 | train/entropy_loss = -0.39929869771003723 | train/policy_loss = 0.5110852122306824 | train/value_loss = 1.3407256603240967 | time/iterations = 1300 | rollout/ep_rew_mean = 70.47252747252747 | rollout/ep_len_mean = 70.47252747252747 | time/fps = 80 | time/time_elapsed = 81 | time/total_timesteps = 6500 |
    [38;21m[INFO] 14:59: [A2C[worker: 3]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.0005506277084350586 | train/entropy_loss = -0.5993375182151794 | train/policy_loss = 0.6051535606384277 | train/value_loss = 1.84113347530365 | time/iterations = 1400 | rollout/ep_rew_mean = 49.87 | rollout/ep_len_mean = 49.87 | time/fps = 81 | time/time_elapsed = 86 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.0005506277084350586 | train/entropy_loss = -0.5993375182151794 | train/policy_loss = 0.6051535606384277 | train/value_loss = 1.84113347530365 | time/iterations = 1400 | rollout/ep_rew_mean = 49.87 | rollout/ep_len_mean = 49.87 | time/fps = 81 | time/time_elapsed = 86 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:59: [A2C[worker: 4]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.0002955198287963867 | train/entropy_loss = -0.5290454626083374 | train/policy_loss = 0.6528939008712769 | train/value_loss = 1.2774860858917236 | time/iterations = 1400 | rollout/ep_rew_mean = 62.87 | rollout/ep_len_mean = 62.87 | time/fps = 81 | time/time_elapsed = 86 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.0002955198287963867 | train/entropy_loss = -0.5290454626083374 | train/policy_loss = 0.6528939008712769 | train/value_loss = 1.2774860858917236 | time/iterations = 1400 | rollout/ep_rew_mean = 62.87 | rollout/ep_len_mean = 62.87 | time/fps = 81 | time/time_elapsed = 86 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:59: [A2C[worker: 0]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.0004247426986694336 | train/entropy_loss = -0.5249357223510742 | train/policy_loss = 0.3760432302951813 | train/value_loss = 1.1898407936096191 | time/iterations = 1400 | rollout/ep_rew_mean = 69.67 | rollout/ep_len_mean = 69.67 | time/fps = 80 | time/time_elapsed = 86 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.0004247426986694336 | train/entropy_loss = -0.5249357223510742 | train/policy_loss = 0.3760432302951813 | train/value_loss = 1.1898407936096191 | time/iterations = 1400 | rollout/ep_rew_mean = 69.67 | rollout/ep_len_mean = 69.67 | time/fps = 80 | time/time_elapsed = 86 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:59: [A2C[worker: 2]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.005190730094909668 | train/entropy_loss = -0.596160888671875 | train/policy_loss = 0.4833240509033203 | train/value_loss = 0.9995946884155273 | time/iterations = 1400 | rollout/ep_rew_mean = 78.58823529411765 | rollout/ep_len_mean = 78.58823529411765 | time/fps = 80 | time/time_elapsed = 86 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.005190730094909668 | train/entropy_loss = -0.596160888671875 | train/policy_loss = 0.4833240509033203 | train/value_loss = 0.9995946884155273 | time/iterations = 1400 | rollout/ep_rew_mean = 78.58823529411765 | rollout/ep_len_mean = 78.58823529411765 | time/fps = 80 | time/time_elapsed = 86 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:59: [A2C[worker: 1]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.00029289722442626953 | train/entropy_loss = -0.3755926191806793 | train/policy_loss = 0.36038723587989807 | train/value_loss = 1.0293537378311157 | time/iterations = 1400 | rollout/ep_rew_mean = 74.15053763440861 | rollout/ep_len_mean = 74.15053763440861 | time/fps = 80 | time/time_elapsed = 87 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.00029289722442626953 | train/entropy_loss = -0.3755926191806793 | train/policy_loss = 0.36038723587989807 | train/value_loss = 1.0293537378311157 | time/iterations = 1400 | rollout/ep_rew_mean = 74.15053763440861 | rollout/ep_len_mean = 74.15053763440861 | time/fps = 80 | time/time_elapsed = 87 | time/total_timesteps = 7000 |
    [38;21m[INFO] 14:59: [A2C[worker: 3]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0012598037719726562 | train/entropy_loss = -0.4173339903354645 | train/policy_loss = 0.4656718671321869 | train/value_loss = 1.4834738969802856 | time/iterations = 1500 | rollout/ep_rew_mean = 54.65 | rollout/ep_len_mean = 54.65 | time/fps = 81 | time/time_elapsed = 92 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0012598037719726562 | train/entropy_loss = -0.4173339903354645 | train/policy_loss = 0.4656718671321869 | train/value_loss = 1.4834738969802856 | time/iterations = 1500 | rollout/ep_rew_mean = 54.65 | rollout/ep_len_mean = 54.65 | time/fps = 81 | time/time_elapsed = 92 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:59: [A2C[worker: 4]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.00031185150146484375 | train/entropy_loss = -0.5005007982254028 | train/policy_loss = 0.7317445874214172 | train/value_loss = 0.9737120866775513 | time/iterations = 1500 | rollout/ep_rew_mean = 65.6 | rollout/ep_len_mean = 65.6 | time/fps = 81 | time/time_elapsed = 92 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.00031185150146484375 | train/entropy_loss = -0.5005007982254028 | train/policy_loss = 0.7317445874214172 | train/value_loss = 0.9737120866775513 | time/iterations = 1500 | rollout/ep_rew_mean = 65.6 | rollout/ep_len_mean = 65.6 | time/fps = 81 | time/time_elapsed = 92 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:59: [A2C[worker: 0]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.000676274299621582 | train/entropy_loss = -0.3929409086704254 | train/policy_loss = 0.233152836561203 | train/value_loss = 0.9026087522506714 | time/iterations = 1500 | rollout/ep_rew_mean = 73.07 | rollout/ep_len_mean = 73.07 | time/fps = 80 | time/time_elapsed = 92 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.000676274299621582 | train/entropy_loss = -0.3929409086704254 | train/policy_loss = 0.233152836561203 | train/value_loss = 0.9026087522506714 | time/iterations = 1500 | rollout/ep_rew_mean = 73.07 | rollout/ep_len_mean = 73.07 | time/fps = 80 | time/time_elapsed = 92 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:59: [A2C[worker: 2]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0004513263702392578 | train/entropy_loss = -0.4784485399723053 | train/policy_loss = 0.2390930950641632 | train/value_loss = 0.7361486554145813 | time/iterations = 1500 | rollout/ep_rew_mean = 83.25287356321839 | rollout/ep_len_mean = 83.25287356321839 | time/fps = 80 | time/time_elapsed = 93 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0004513263702392578 | train/entropy_loss = -0.4784485399723053 | train/policy_loss = 0.2390930950641632 | train/value_loss = 0.7361486554145813 | time/iterations = 1500 | rollout/ep_rew_mean = 83.25287356321839 | rollout/ep_len_mean = 83.25287356321839 | time/fps = 80 | time/time_elapsed = 93 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:59: [A2C[worker: 1]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -3.0159950256347656e-05 | train/entropy_loss = -0.4725797772407532 | train/policy_loss = 0.4035821557044983 | train/value_loss = 0.7609403729438782 | time/iterations = 1500 | rollout/ep_rew_mean = 77.91578947368421 | rollout/ep_len_mean = 77.91578947368421 | time/fps = 80 | time/time_elapsed = 93 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -3.0159950256347656e-05 | train/entropy_loss = -0.4725797772407532 | train/policy_loss = 0.4035821557044983 | train/value_loss = 0.7609403729438782 | time/iterations = 1500 | rollout/ep_rew_mean = 77.91578947368421 | rollout/ep_len_mean = 77.91578947368421 | time/fps = 80 | time/time_elapsed = 93 | time/total_timesteps = 7500 |
    [38;21m[INFO] 14:59: [A2C[worker: 3]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0005797147750854492 | train/entropy_loss = -0.5182167887687683 | train/policy_loss = 0.3960009515285492 | train/value_loss = 1.1624362468719482 | time/iterations = 1600 | rollout/ep_rew_mean = 58.48 | rollout/ep_len_mean = 58.48 | time/fps = 81 | time/time_elapsed = 98 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0005797147750854492 | train/entropy_loss = -0.5182167887687683 | train/policy_loss = 0.3960009515285492 | train/value_loss = 1.1624362468719482 | time/iterations = 1600 | rollout/ep_rew_mean = 58.48 | rollout/ep_len_mean = 58.48 | time/fps = 81 | time/time_elapsed = 98 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:59: [A2C[worker: 4]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.00013172626495361328 | train/entropy_loss = -0.5642613172531128 | train/policy_loss = 0.23594875633716583 | train/value_loss = 0.7230726480484009 | time/iterations = 1600 | rollout/ep_rew_mean = 71.24 | rollout/ep_len_mean = 71.24 | time/fps = 81 | time/time_elapsed = 98 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.00013172626495361328 | train/entropy_loss = -0.5642613172531128 | train/policy_loss = 0.23594875633716583 | train/value_loss = 0.7230726480484009 | time/iterations = 1600 | rollout/ep_rew_mean = 71.24 | rollout/ep_len_mean = 71.24 | time/fps = 81 | time/time_elapsed = 98 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:59: [A2C[worker: 0]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.00013315677642822266 | train/entropy_loss = -0.5476351976394653 | train/policy_loss = 0.35055047273635864 | train/value_loss = 0.6474453210830688 | time/iterations = 1600 | rollout/ep_rew_mean = 74.72 | rollout/ep_len_mean = 74.72 | time/fps = 80 | time/time_elapsed = 98 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.00013315677642822266 | train/entropy_loss = -0.5476351976394653 | train/policy_loss = 0.35055047273635864 | train/value_loss = 0.6474453210830688 | time/iterations = 1600 | rollout/ep_rew_mean = 74.72 | rollout/ep_len_mean = 74.72 | time/fps = 80 | time/time_elapsed = 98 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:59: [A2C[worker: 2]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 7.164478302001953e-05 | train/entropy_loss = -0.5874301195144653 | train/policy_loss = 0.29882076382637024 | train/value_loss = 0.5162824392318726 | time/iterations = 1600 | rollout/ep_rew_mean = 87.67777777777778 | rollout/ep_len_mean = 87.67777777777778 | time/fps = 80 | time/time_elapsed = 99 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 7.164478302001953e-05 | train/entropy_loss = -0.5874301195144653 | train/policy_loss = 0.29882076382637024 | train/value_loss = 0.5162824392318726 | time/iterations = 1600 | rollout/ep_rew_mean = 87.67777777777778 | rollout/ep_len_mean = 87.67777777777778 | time/fps = 80 | time/time_elapsed = 99 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:59: [A2C[worker: 1]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0001875162124633789 | train/entropy_loss = -0.3827807903289795 | train/policy_loss = 0.13148829340934753 | train/value_loss = 0.5311938524246216 | time/iterations = 1600 | rollout/ep_rew_mean = 80.76288659793815 | rollout/ep_len_mean = 80.76288659793815 | time/fps = 80 | time/time_elapsed = 99 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0001875162124633789 | train/entropy_loss = -0.3827807903289795 | train/policy_loss = 0.13148829340934753 | train/value_loss = 0.5311938524246216 | time/iterations = 1600 | rollout/ep_rew_mean = 80.76288659793815 | rollout/ep_len_mean = 80.76288659793815 | time/fps = 80 | time/time_elapsed = 99 | time/total_timesteps = 8000 |
    [38;21m[INFO] 14:59: [A2C[worker: 3]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -0.0001188516616821289 | train/entropy_loss = -0.5335004925727844 | train/policy_loss = 0.25600093603134155 | train/value_loss = 0.8859585523605347 | time/iterations = 1700 | rollout/ep_rew_mean = 63.2 | rollout/ep_len_mean = 63.2 | time/fps = 81 | time/time_elapsed = 104 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -0.0001188516616821289 | train/entropy_loss = -0.5335004925727844 | train/policy_loss = 0.25600093603134155 | train/value_loss = 0.8859585523605347 | time/iterations = 1700 | rollout/ep_rew_mean = 63.2 | rollout/ep_len_mean = 63.2 | time/fps = 81 | time/time_elapsed = 104 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:59: [A2C[worker: 4]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 2.8848648071289062e-05 | train/entropy_loss = -0.6418264508247375 | train/policy_loss = 0.32908764481544495 | train/value_loss = 0.5009179711341858 | time/iterations = 1700 | rollout/ep_rew_mean = 75.33 | rollout/ep_len_mean = 75.33 | time/fps = 81 | time/time_elapsed = 104 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 2.8848648071289062e-05 | train/entropy_loss = -0.6418264508247375 | train/policy_loss = 0.32908764481544495 | train/value_loss = 0.5009179711341858 | time/iterations = 1700 | rollout/ep_rew_mean = 75.33 | rollout/ep_len_mean = 75.33 | time/fps = 81 | time/time_elapsed = 104 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:59: [A2C[worker: 0]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -0.0001709461212158203 | train/entropy_loss = -0.4986186921596527 | train/policy_loss = 0.35891956090927124 | train/value_loss = 0.43840113282203674 | time/iterations = 1700 | rollout/ep_rew_mean = 81.57 | rollout/ep_len_mean = 81.57 | time/fps = 80 | time/time_elapsed = 105 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -0.0001709461212158203 | train/entropy_loss = -0.4986186921596527 | train/policy_loss = 0.35891956090927124 | train/value_loss = 0.43840113282203674 | time/iterations = 1700 | rollout/ep_rew_mean = 81.57 | rollout/ep_len_mean = 81.57 | time/fps = 80 | time/time_elapsed = 105 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:59: [A2C[worker: 2]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 0.0003292560577392578 | train/entropy_loss = -0.47510701417922974 | train/policy_loss = 0.2571336030960083 | train/value_loss = 0.331814706325531 | time/iterations = 1700 | rollout/ep_rew_mean = 89.69565217391305 | rollout/ep_len_mean = 89.69565217391305 | time/fps = 80 | time/time_elapsed = 105 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 0.0003292560577392578 | train/entropy_loss = -0.47510701417922974 | train/policy_loss = 0.2571336030960083 | train/value_loss = 0.331814706325531 | time/iterations = 1700 | rollout/ep_rew_mean = 89.69565217391305 | rollout/ep_len_mean = 89.69565217391305 | time/fps = 80 | time/time_elapsed = 105 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:59: [A2C[worker: 1]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 5.960464477539063e-08 | train/entropy_loss = -0.3092193603515625 | train/policy_loss = 0.15335866808891296 | train/value_loss = 0.3478584885597229 | time/iterations = 1700 | rollout/ep_rew_mean = 84.89 | rollout/ep_len_mean = 84.89 | time/fps = 79 | time/time_elapsed = 106 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 5.960464477539063e-08 | train/entropy_loss = -0.3092193603515625 | train/policy_loss = 0.15335866808891296 | train/value_loss = 0.3478584885597229 | time/iterations = 1700 | rollout/ep_rew_mean = 84.89 | rollout/ep_len_mean = 84.89 | time/fps = 79 | time/time_elapsed = 106 | time/total_timesteps = 8500 |
    [38;21m[INFO] 14:59: [A2C[worker: 3]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -0.00036716461181640625 | train/entropy_loss = -0.39411264657974243 | train/policy_loss = 0.3742735981941223 | train/value_loss = 0.6390293836593628 | time/iterations = 1800 | rollout/ep_rew_mean = 66.35 | rollout/ep_len_mean = 66.35 | time/fps = 80 | time/time_elapsed = 111 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -0.00036716461181640625 | train/entropy_loss = -0.39411264657974243 | train/policy_loss = 0.3742735981941223 | train/value_loss = 0.6390293836593628 | time/iterations = 1800 | rollout/ep_rew_mean = 66.35 | rollout/ep_len_mean = 66.35 | time/fps = 80 | time/time_elapsed = 111 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:59: [A2C[worker: 4]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -1.823902130126953e-05 | train/entropy_loss = -0.5751914381980896 | train/policy_loss = 0.30691492557525635 | train/value_loss = 0.32241562008857727 | time/iterations = 1800 | rollout/ep_rew_mean = 79.03 | rollout/ep_len_mean = 79.03 | time/fps = 80 | time/time_elapsed = 111 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -1.823902130126953e-05 | train/entropy_loss = -0.5751914381980896 | train/policy_loss = 0.30691492557525635 | train/value_loss = 0.32241562008857727 | time/iterations = 1800 | rollout/ep_rew_mean = 79.03 | rollout/ep_len_mean = 79.03 | time/fps = 80 | time/time_elapsed = 111 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:59: [A2C[worker: 0]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -0.00032448768615722656 | train/entropy_loss = -0.37135934829711914 | train/policy_loss = 0.311820924282074 | train/value_loss = 0.26541775465011597 | time/iterations = 1800 | rollout/ep_rew_mean = 83.74 | rollout/ep_len_mean = 83.74 | time/fps = 79 | time/time_elapsed = 112 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -0.00032448768615722656 | train/entropy_loss = -0.37135934829711914 | train/policy_loss = 0.311820924282074 | train/value_loss = 0.26541775465011597 | time/iterations = 1800 | rollout/ep_rew_mean = 83.74 | rollout/ep_len_mean = 83.74 | time/fps = 79 | time/time_elapsed = 112 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:59: [A2C[worker: 2]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 1.3113021850585938e-06 | train/entropy_loss = -0.43398839235305786 | train/policy_loss = 0.21055611968040466 | train/value_loss = 0.19081199169158936 | time/iterations = 1800 | rollout/ep_rew_mean = 94.51578947368421 | rollout/ep_len_mean = 94.51578947368421 | time/fps = 79 | time/time_elapsed = 112 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 1.3113021850585938e-06 | train/entropy_loss = -0.43398839235305786 | train/policy_loss = 0.21055611968040466 | train/value_loss = 0.19081199169158936 | time/iterations = 1800 | rollout/ep_rew_mean = 94.51578947368421 | rollout/ep_len_mean = 94.51578947368421 | time/fps = 79 | time/time_elapsed = 112 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:59: [A2C[worker: 1]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 1.1920928955078125e-07 | train/entropy_loss = -0.27584192156791687 | train/policy_loss = 0.11686202138662338 | train/value_loss = 0.20508143305778503 | time/iterations = 1800 | rollout/ep_rew_mean = 88.72 | rollout/ep_len_mean = 88.72 | time/fps = 78 | time/time_elapsed = 114 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 1.1920928955078125e-07 | train/entropy_loss = -0.27584192156791687 | train/policy_loss = 0.11686202138662338 | train/value_loss = 0.20508143305778503 | time/iterations = 1800 | rollout/ep_rew_mean = 88.72 | rollout/ep_len_mean = 88.72 | time/fps = 78 | time/time_elapsed = 114 | time/total_timesteps = 9000 |
    [38;21m[INFO] 14:59: [A2C[worker: 3]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -4.76837158203125e-07 | train/entropy_loss = -0.5801633596420288 | train/policy_loss = 0.32568711042404175 | train/value_loss = 0.43057283759117126 | time/iterations = 1900 | rollout/ep_rew_mean = 70.96 | rollout/ep_len_mean = 70.96 | time/fps = 79 | time/time_elapsed = 119 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -4.76837158203125e-07 | train/entropy_loss = -0.5801633596420288 | train/policy_loss = 0.32568711042404175 | train/value_loss = 0.43057283759117126 | time/iterations = 1900 | rollout/ep_rew_mean = 70.96 | rollout/ep_len_mean = 70.96 | time/fps = 79 | time/time_elapsed = 119 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:59: [A2C[worker: 4]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 2.4020671844482422e-05 | train/entropy_loss = -0.5851779580116272 | train/policy_loss = 0.12278906255960464 | train/value_loss = 0.18095146119594574 | time/iterations = 1900 | rollout/ep_rew_mean = 81.22 | rollout/ep_len_mean = 81.22 | time/fps = 79 | time/time_elapsed = 119 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 2.4020671844482422e-05 | train/entropy_loss = -0.5851779580116272 | train/policy_loss = 0.12278906255960464 | train/value_loss = 0.18095146119594574 | time/iterations = 1900 | rollout/ep_rew_mean = 81.22 | rollout/ep_len_mean = 81.22 | time/fps = 79 | time/time_elapsed = 119 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:59: [A2C[worker: 0]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -8.14199447631836e-05 | train/entropy_loss = -0.4903884828090668 | train/policy_loss = 0.10781295597553253 | train/value_loss = 0.13885948061943054 | time/iterations = 1900 | rollout/ep_rew_mean = 89.89 | rollout/ep_len_mean = 89.89 | time/fps = 78 | time/time_elapsed = 120 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -8.14199447631836e-05 | train/entropy_loss = -0.4903884828090668 | train/policy_loss = 0.10781295597553253 | train/value_loss = 0.13885948061943054 | time/iterations = 1900 | rollout/ep_rew_mean = 89.89 | rollout/ep_len_mean = 89.89 | time/fps = 78 | time/time_elapsed = 120 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:59: [A2C[worker: 2]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.00022649765014648438 | train/entropy_loss = -0.3601681590080261 | train/policy_loss = 0.19155782461166382 | train/value_loss = 0.08677476644515991 | time/iterations = 1900 | rollout/ep_rew_mean = 97.27835051546391 | rollout/ep_len_mean = 97.27835051546391 | time/fps = 78 | time/time_elapsed = 120 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.00022649765014648438 | train/entropy_loss = -0.3601681590080261 | train/policy_loss = 0.19155782461166382 | train/value_loss = 0.08677476644515991 | time/iterations = 1900 | rollout/ep_rew_mean = 97.27835051546391 | rollout/ep_len_mean = 97.27835051546391 | time/fps = 78 | time/time_elapsed = 120 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:59: [A2C[worker: 1]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.0 | train/entropy_loss = -0.2401140034198761 | train/policy_loss = 0.248975470662117 | train/value_loss = 0.0978275015950203 | time/iterations = 1900 | rollout/ep_rew_mean = 92.31 | rollout/ep_len_mean = 92.31 | time/fps = 77 | time/time_elapsed = 122 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.0 | train/entropy_loss = -0.2401140034198761 | train/policy_loss = 0.248975470662117 | train/value_loss = 0.0978275015950203 | time/iterations = 1900 | rollout/ep_rew_mean = 92.31 | rollout/ep_len_mean = 92.31 | time/fps = 77 | time/time_elapsed = 122 | time/total_timesteps = 9500 |
    [38;21m[INFO] 14:59: ... trained! [0m
    INFO:rlberry_logger:... trained!
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:59: Saved ExperimentManager(A2C) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(A2C) using pickle.
    [38;21m[INFO] 14:59: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_14-57-32_8830fede/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_14-57-32_8830fede/manager_obj.pickle'
    [38;21m[INFO] 14:59: Running ExperimentManager fit() for PPO with n_fit = 5 and max_workers = None. [0m
    INFO:rlberry_logger:Running ExperimentManager fit() for PPO with n_fit = 5 and max_workers = None.
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:59:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        4           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        4           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:59:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        0           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        0           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:59:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        3           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        3           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:59:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        1           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        1           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 14:59:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        2           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        2           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:00: [PPO[worker: 4]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.881720430107528 | rollout/ep_len_mean = 21.881720430107528 | time/fps = 153 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.881720430107528 | rollout/ep_len_mean = 21.881720430107528 | time/fps = 153 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:00: [PPO[worker: 3]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.50526315789474 | rollout/ep_len_mean = 21.50526315789474 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.50526315789474 | rollout/ep_len_mean = 21.50526315789474 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:00: [PPO[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.102272727272727 | rollout/ep_len_mean = 23.102272727272727 | time/fps = 147 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.102272727272727 | rollout/ep_len_mean = 23.102272727272727 | time/fps = 147 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:00: [PPO[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.61111111111111 | rollout/ep_len_mean = 22.61111111111111 | time/fps = 147 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.61111111111111 | rollout/ep_len_mean = 22.61111111111111 | time/fps = 147 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:00: [PPO[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.023255813953487 | rollout/ep_len_mean = 23.023255813953487 | time/fps = 151 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.023255813953487 | rollout/ep_len_mean = 23.023255813953487 | time/fps = 151 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    [38;21m[INFO] 15:00: [PPO[worker: 4]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.59 | rollout/ep_len_mean = 26.59 | time/fps = 104 | time/time_elapsed = 39 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6863450620323419 | train/policy_gradient_loss = -0.013453802339790854 | train/value_loss = 50.67496230006218 | train/approx_kl = 0.009218430146574974 | train/clip_fraction = 0.097509765625 | train/loss = 6.385239601135254 | train/explained_variance = -0.0032502412796020508 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.59 | rollout/ep_len_mean = 26.59 | time/fps = 104 | time/time_elapsed = 39 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6863450620323419 | train/policy_gradient_loss = -0.013453802339790854 | train/value_loss = 50.67496230006218 | train/approx_kl = 0.009218430146574974 | train/clip_fraction = 0.097509765625 | train/loss = 6.385239601135254 | train/explained_variance = -0.0032502412796020508 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:00: [PPO[worker: 3]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.4 | rollout/ep_len_mean = 26.4 | time/fps = 103 | time/time_elapsed = 39 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6860328065231442 | train/policy_gradient_loss = -0.016126183030428363 | train/value_loss = 50.43991032540798 | train/approx_kl = 0.008711619302630424 | train/clip_fraction = 0.102490234375 | train/loss = 6.996300220489502 | train/explained_variance = 0.0004137754440307617 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.4 | rollout/ep_len_mean = 26.4 | time/fps = 103 | time/time_elapsed = 39 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6860328065231442 | train/policy_gradient_loss = -0.016126183030428363 | train/value_loss = 50.43991032540798 | train/approx_kl = 0.008711619302630424 | train/clip_fraction = 0.102490234375 | train/loss = 6.996300220489502 | train/explained_variance = 0.0004137754440307617 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:00: [PPO[worker: 2]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.94 | rollout/ep_len_mean = 25.94 | time/fps = 103 | time/time_elapsed = 39 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6865187343209982 | train/policy_gradient_loss = -0.013688449657638557 | train/value_loss = 51.497758308053015 | train/approx_kl = 0.007447786629199982 | train/clip_fraction = 0.088232421875 | train/loss = 6.370249271392822 | train/explained_variance = 0.0016186237335205078 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.94 | rollout/ep_len_mean = 25.94 | time/fps = 103 | time/time_elapsed = 39 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6865187343209982 | train/policy_gradient_loss = -0.013688449657638557 | train/value_loss = 51.497758308053015 | train/approx_kl = 0.007447786629199982 | train/clip_fraction = 0.088232421875 | train/loss = 6.370249271392822 | train/explained_variance = 0.0016186237335205078 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:00: [PPO[worker: 1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 24.56 | rollout/ep_len_mean = 24.56 | time/fps = 102 | time/time_elapsed = 39 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6855680016800761 | train/policy_gradient_loss = -0.019546236126916482 | train/value_loss = 55.85037803947925 | train/approx_kl = 0.009038202464580536 | train/clip_fraction = 0.114892578125 | train/loss = 8.413447380065918 | train/explained_variance = -0.005194425582885742 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 24.56 | rollout/ep_len_mean = 24.56 | time/fps = 102 | time/time_elapsed = 39 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6855680016800761 | train/policy_gradient_loss = -0.019546236126916482 | train/value_loss = 55.85037803947925 | train/approx_kl = 0.009038202464580536 | train/clip_fraction = 0.114892578125 | train/loss = 8.413447380065918 | train/explained_variance = -0.005194425582885742 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:00: [PPO[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 28.5 | rollout/ep_len_mean = 28.5 | time/fps = 102 | time/time_elapsed = 40 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6855831030756235 | train/policy_gradient_loss = -0.020951770500687417 | train/value_loss = 58.539475354552266 | train/approx_kl = 0.009215479716658592 | train/clip_fraction = 0.126611328125 | train/loss = 9.782316207885742 | train/explained_variance = 0.0002021193504333496 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 28.5 | rollout/ep_len_mean = 28.5 | time/fps = 102 | time/time_elapsed = 40 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6855831030756235 | train/policy_gradient_loss = -0.020951770500687417 | train/value_loss = 58.539475354552266 | train/approx_kl = 0.009215479716658592 | train/clip_fraction = 0.126611328125 | train/loss = 9.782316207885742 | train/explained_variance = 0.0002021193504333496 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: [PPO[worker: 3]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 36.28 | rollout/ep_len_mean = 36.28 | time/fps = 97 | time/time_elapsed = 62 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6700566191226244 | train/policy_gradient_loss = -0.01657824197609443 | train/value_loss = 39.26095670461655 | train/approx_kl = 0.009997228160500526 | train/clip_fraction = 0.06826171875 | train/loss = 12.370352745056152 | train/explained_variance = 0.10059136152267456 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 36.28 | rollout/ep_len_mean = 36.28 | time/fps = 97 | time/time_elapsed = 62 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6700566191226244 | train/policy_gradient_loss = -0.01657824197609443 | train/value_loss = 39.26095670461655 | train/approx_kl = 0.009997228160500526 | train/clip_fraction = 0.06826171875 | train/loss = 12.370352745056152 | train/explained_variance = 0.10059136152267456 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: [PPO[worker: 4]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.78 | rollout/ep_len_mean = 33.78 | time/fps = 98 | time/time_elapsed = 62 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.666048844717443 | train/policy_gradient_loss = -0.014253238176752347 | train/value_loss = 37.09622389674187 | train/approx_kl = 0.008612859062850475 | train/clip_fraction = 0.049853515625 | train/loss = 12.620468139648438 | train/explained_variance = 0.08953225612640381 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.78 | rollout/ep_len_mean = 33.78 | time/fps = 98 | time/time_elapsed = 62 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.666048844717443 | train/policy_gradient_loss = -0.014253238176752347 | train/value_loss = 37.09622389674187 | train/approx_kl = 0.008612859062850475 | train/clip_fraction = 0.049853515625 | train/loss = 12.620468139648438 | train/explained_variance = 0.08953225612640381 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: [PPO[worker: 2]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 31.38 | rollout/ep_len_mean = 31.38 | time/fps = 96 | time/time_elapsed = 63 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6641447197645902 | train/policy_gradient_loss = -0.016446807964530307 | train/value_loss = 34.83806987404823 | train/approx_kl = 0.009349122643470764 | train/clip_fraction = 0.066015625 | train/loss = 13.647862434387207 | train/explained_variance = 0.1013384461402893 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 31.38 | rollout/ep_len_mean = 31.38 | time/fps = 96 | time/time_elapsed = 63 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6641447197645902 | train/policy_gradient_loss = -0.016446807964530307 | train/value_loss = 34.83806987404823 | train/approx_kl = 0.009349122643470764 | train/clip_fraction = 0.066015625 | train/loss = 13.647862434387207 | train/explained_variance = 0.1013384461402893 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: [PPO[worker: 1]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.24 | rollout/ep_len_mean = 33.24 | time/fps = 96 | time/time_elapsed = 63 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.667991516366601 | train/policy_gradient_loss = -0.019207102236396167 | train/value_loss = 37.87286460995674 | train/approx_kl = 0.009698237292468548 | train/clip_fraction = 0.072607421875 | train/loss = 14.975642204284668 | train/explained_variance = 0.06899595260620117 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.24 | rollout/ep_len_mean = 33.24 | time/fps = 96 | time/time_elapsed = 63 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.667991516366601 | train/policy_gradient_loss = -0.019207102236396167 | train/value_loss = 37.87286460995674 | train/approx_kl = 0.009698237292468548 | train/clip_fraction = 0.072607421875 | train/loss = 14.975642204284668 | train/explained_variance = 0.06899595260620117 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: [PPO[worker: 0]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 37.22 | rollout/ep_len_mean = 37.22 | time/fps = 95 | time/time_elapsed = 64 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6667703006416559 | train/policy_gradient_loss = -0.01425587632402312 | train/value_loss = 38.538201689720154 | train/approx_kl = 0.008471712470054626 | train/clip_fraction = 0.0494140625 | train/loss = 13.141069412231445 | train/explained_variance = 0.08203208446502686 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 37.22 | rollout/ep_len_mean = 37.22 | time/fps = 95 | time/time_elapsed = 64 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6667703006416559 | train/policy_gradient_loss = -0.01425587632402312 | train/value_loss = 38.538201689720154 | train/approx_kl = 0.008471712470054626 | train/clip_fraction = 0.0494140625 | train/loss = 13.141069412231445 | train/explained_variance = 0.08203208446502686 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: [PPO[worker: 3]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.07 | rollout/ep_len_mean = 47.07 | time/fps = 95 | time/time_elapsed = 86 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6316981950774789 | train/policy_gradient_loss = -0.02072375753778033 | train/value_loss = 52.43568618297577 | train/approx_kl = 0.009395284578204155 | train/clip_fraction = 0.093212890625 | train/loss = 19.48152732849121 | train/explained_variance = 0.2199450135231018 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.07 | rollout/ep_len_mean = 47.07 | time/fps = 95 | time/time_elapsed = 86 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6316981950774789 | train/policy_gradient_loss = -0.02072375753778033 | train/value_loss = 52.43568618297577 | train/approx_kl = 0.009395284578204155 | train/clip_fraction = 0.093212890625 | train/loss = 19.48152732849121 | train/explained_variance = 0.2199450135231018 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: [PPO[worker: 4]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.36 | rollout/ep_len_mean = 47.36 | time/fps = 95 | time/time_elapsed = 86 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6303293297067285 | train/policy_gradient_loss = -0.0192519291798817 | train/value_loss = 53.276741510629655 | train/approx_kl = 0.009116223081946373 | train/clip_fraction = 0.09228515625 | train/loss = 26.364980697631836 | train/explained_variance = 0.2059217095375061 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.36 | rollout/ep_len_mean = 47.36 | time/fps = 95 | time/time_elapsed = 86 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6303293297067285 | train/policy_gradient_loss = -0.0192519291798817 | train/value_loss = 53.276741510629655 | train/approx_kl = 0.009116223081946373 | train/clip_fraction = 0.09228515625 | train/loss = 26.364980697631836 | train/explained_variance = 0.2059217095375061 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: [PPO[worker: 2]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 40.84 | rollout/ep_len_mean = 40.84 | time/fps = 94 | time/time_elapsed = 86 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6328764887526631 | train/policy_gradient_loss = -0.023230642164708114 | train/value_loss = 43.83278710246086 | train/approx_kl = 0.009393437765538692 | train/clip_fraction = 0.10986328125 | train/loss = 18.86742401123047 | train/explained_variance = 0.2846730351448059 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 40.84 | rollout/ep_len_mean = 40.84 | time/fps = 94 | time/time_elapsed = 86 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6328764887526631 | train/policy_gradient_loss = -0.023230642164708114 | train/value_loss = 43.83278710246086 | train/approx_kl = 0.009393437765538692 | train/clip_fraction = 0.10986328125 | train/loss = 18.86742401123047 | train/explained_variance = 0.2846730351448059 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: [PPO[worker: 1]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 46.21 | rollout/ep_len_mean = 46.21 | time/fps = 94 | time/time_elapsed = 86 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6402351712808013 | train/policy_gradient_loss = -0.018197900809173006 | train/value_loss = 56.2570192694664 | train/approx_kl = 0.009540338069200516 | train/clip_fraction = 0.095751953125 | train/loss = 21.976451873779297 | train/explained_variance = 0.23956573009490967 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 46.21 | rollout/ep_len_mean = 46.21 | time/fps = 94 | time/time_elapsed = 86 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6402351712808013 | train/policy_gradient_loss = -0.018197900809173006 | train/value_loss = 56.2570192694664 | train/approx_kl = 0.009540338069200516 | train/clip_fraction = 0.095751953125 | train/loss = 21.976451873779297 | train/explained_variance = 0.23956573009490967 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: [PPO[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 48.48 | rollout/ep_len_mean = 48.48 | time/fps = 93 | time/time_elapsed = 87 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.641019894182682 | train/policy_gradient_loss = -0.018460670115746324 | train/value_loss = 56.663608932495116 | train/approx_kl = 0.010359940119087696 | train/clip_fraction = 0.097509765625 | train/loss = 22.871599197387695 | train/explained_variance = 0.23288685083389282 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 48.48 | rollout/ep_len_mean = 48.48 | time/fps = 93 | time/time_elapsed = 87 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.641019894182682 | train/policy_gradient_loss = -0.018460670115746324 | train/value_loss = 56.663608932495116 | train/approx_kl = 0.010359940119087696 | train/clip_fraction = 0.097509765625 | train/loss = 22.871599197387695 | train/explained_variance = 0.23288685083389282 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:01: ... trained! [0m
    INFO:rlberry_logger:... trained!
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:01: Saved ExperimentManager(PPO) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(PPO) using pickle.
    [38;21m[INFO] 15:01: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_14-57-32_5b16625f/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_14-57-32_5b16625f/manager_obj.pickle'
    [38;21m[INFO] 15:01: Saved ExperimentManager(A2C) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(A2C) using pickle.
    [38;21m[INFO] 15:01: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_14-57-32_8830fede/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_14-57-32_8830fede/manager_obj.pickle'
    [38;21m[INFO] 15:01: Saved ExperimentManager(PPO) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(PPO) using pickle.
    [38;21m[INFO] 15:01: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_14-57-32_5b16625f/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_14-57-32_5b16625f/manager_obj.pickle'
    [38;21m[INFO] 15:01: Evaluating agent 0 [0m
    INFO:rlberry_logger:Evaluating agent 0
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:01: Evaluating agent 1 [0m
    INFO:rlberry_logger:Evaluating agent 1
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:01: Evaluating agent 2 [0m
    INFO:rlberry_logger:Evaluating agent 2
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:02: Evaluating agent 3 [0m
    INFO:rlberry_logger:Evaluating agent 3
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:02: Evaluating agent 4 [0m
    INFO:rlberry_logger:Evaluating agent 4
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:02: Evaluating agent 0 [0m
    INFO:rlberry_logger:Evaluating agent 0
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:02: Evaluating agent 1 [0m
    INFO:rlberry_logger:Evaluating agent 1
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:02: Evaluating agent 2 [0m
    INFO:rlberry_logger:Evaluating agent 2
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:02: Evaluating agent 3 [0m
    INFO:rlberry_logger:Evaluating agent 3
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:03: Evaluating agent 4 [0m
    INFO:rlberry_logger:Evaluating agent 4
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:03: Running ExperimentManager fit() for A2C with n_fit = 5 and max_workers = None. [0m
    INFO:rlberry_logger:Running ExperimentManager fit() for A2C with n_fit = 5 and max_workers = None.


    Step 0


    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:03:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        4           0.001               500 [0m
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        4           0.001               500
    [38;21m[INFO] 15:03:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        1           0.001               500 [0m
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        1           0.001               500
    [38;21m[INFO] 15:03:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        0           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        0           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:03:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        2           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        2           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:03:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        3           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        3           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:03: [A2C[worker: 4]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.05766713619232178 | train/entropy_loss = -0.6919767260551453 | train/policy_loss = 1.891247034072876 | train/value_loss = 8.477182388305664 | time/iterations = 100 | rollout/ep_rew_mean = 19.23076923076923 | rollout/ep_len_mean = 19.23076923076923 | time/fps = 77 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.05766713619232178 | train/entropy_loss = -0.6919767260551453 | train/policy_loss = 1.891247034072876 | train/value_loss = 8.477182388305664 | time/iterations = 100 | rollout/ep_rew_mean = 19.23076923076923 | rollout/ep_len_mean = 19.23076923076923 | time/fps = 77 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:03: [A2C[worker: 1]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.2983982563018799 | train/entropy_loss = -0.6063982248306274 | train/policy_loss = -3.502866268157959 | train/value_loss = 58.141090393066406 | time/iterations = 100 | rollout/ep_rew_mean = 45.0 | rollout/ep_len_mean = 45.0 | time/fps = 76 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.2983982563018799 | train/entropy_loss = -0.6063982248306274 | train/policy_loss = -3.502866268157959 | train/value_loss = 58.141090393066406 | time/iterations = 100 | rollout/ep_rew_mean = 45.0 | rollout/ep_len_mean = 45.0 | time/fps = 76 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:03: [A2C[worker: 2]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.18617355823516846 | train/entropy_loss = -0.6523889899253845 | train/policy_loss = 0.9841373562812805 | train/value_loss = 5.392605304718018 | time/iterations = 100 | rollout/ep_rew_mean = 33.2 | rollout/ep_len_mean = 33.2 | time/fps = 73 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.18617355823516846 | train/entropy_loss = -0.6523889899253845 | train/policy_loss = 0.9841373562812805 | train/value_loss = 5.392605304718018 | time/iterations = 100 | rollout/ep_rew_mean = 33.2 | rollout/ep_len_mean = 33.2 | time/fps = 73 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:03: [A2C[worker: 0]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.06284141540527344 | train/entropy_loss = -0.6693260669708252 | train/policy_loss = 1.993377685546875 | train/value_loss = 9.944544792175293 | time/iterations = 100 | rollout/ep_rew_mean = 35.0 | rollout/ep_len_mean = 35.0 | time/fps = 74 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.06284141540527344 | train/entropy_loss = -0.6693260669708252 | train/policy_loss = 1.993377685546875 | train/value_loss = 9.944544792175293 | time/iterations = 100 | rollout/ep_rew_mean = 35.0 | rollout/ep_len_mean = 35.0 | time/fps = 74 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:03: [A2C[worker: 3]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.2367088794708252 | train/entropy_loss = -0.6476349830627441 | train/policy_loss = 2.038792371749878 | train/value_loss = 9.22716236114502 | time/iterations = 100 | rollout/ep_rew_mean = 24.25 | rollout/ep_len_mean = 24.25 | time/fps = 73 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.2367088794708252 | train/entropy_loss = -0.6476349830627441 | train/policy_loss = 2.038792371749878 | train/value_loss = 9.22716236114502 | time/iterations = 100 | rollout/ep_rew_mean = 24.25 | rollout/ep_len_mean = 24.25 | time/fps = 73 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    [38;21m[INFO] 15:03: [A2C[worker: 4]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.09927737712860107 | train/entropy_loss = -0.6893783807754517 | train/policy_loss = 1.839686632156372 | train/value_loss = 8.209020614624023 | time/iterations = 200 | rollout/ep_rew_mean = 21.32608695652174 | rollout/ep_len_mean = 21.32608695652174 | time/fps = 78 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.09927737712860107 | train/entropy_loss = -0.6893783807754517 | train/policy_loss = 1.839686632156372 | train/value_loss = 8.209020614624023 | time/iterations = 200 | rollout/ep_rew_mean = 21.32608695652174 | rollout/ep_len_mean = 21.32608695652174 | time/fps = 78 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:03: [A2C[worker: 1]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.10602784156799316 | train/entropy_loss = -0.6317161321640015 | train/policy_loss = 1.8301231861114502 | train/value_loss = 7.9128217697143555 | time/iterations = 200 | rollout/ep_rew_mean = 46.0 | rollout/ep_len_mean = 46.0 | time/fps = 76 | time/time_elapsed = 13 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.10602784156799316 | train/entropy_loss = -0.6317161321640015 | train/policy_loss = 1.8301231861114502 | train/value_loss = 7.9128217697143555 | time/iterations = 200 | rollout/ep_rew_mean = 46.0 | rollout/ep_len_mean = 46.0 | time/fps = 76 | time/time_elapsed = 13 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:03: [A2C[worker: 2]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.07836616039276123 | train/entropy_loss = -0.5153092741966248 | train/policy_loss = 1.625454306602478 | train/value_loss = 5.707804203033447 | time/iterations = 200 | rollout/ep_rew_mean = 30.875 | rollout/ep_len_mean = 30.875 | time/fps = 75 | time/time_elapsed = 13 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.07836616039276123 | train/entropy_loss = -0.5153092741966248 | train/policy_loss = 1.625454306602478 | train/value_loss = 5.707804203033447 | time/iterations = 200 | rollout/ep_rew_mean = 30.875 | rollout/ep_len_mean = 30.875 | time/fps = 75 | time/time_elapsed = 13 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:03: [A2C[worker: 0]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.17917490005493164 | train/entropy_loss = -0.6587372422218323 | train/policy_loss = 1.3536200523376465 | train/value_loss = 7.69503927230835 | time/iterations = 200 | rollout/ep_rew_mean = 36.629629629629626 | rollout/ep_len_mean = 36.629629629629626 | time/fps = 74 | time/time_elapsed = 13 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.17917490005493164 | train/entropy_loss = -0.6587372422218323 | train/policy_loss = 1.3536200523376465 | train/value_loss = 7.69503927230835 | time/iterations = 200 | rollout/ep_rew_mean = 36.629629629629626 | rollout/ep_len_mean = 36.629629629629626 | time/fps = 74 | time/time_elapsed = 13 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:03: [A2C[worker: 3]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.19603121280670166 | train/entropy_loss = -0.6883419752120972 | train/policy_loss = 1.827109932899475 | train/value_loss = 9.819732666015625 | time/iterations = 200 | rollout/ep_rew_mean = 22.5 | rollout/ep_len_mean = 22.5 | time/fps = 74 | time/time_elapsed = 13 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.19603121280670166 | train/entropy_loss = -0.6883419752120972 | train/policy_loss = 1.827109932899475 | train/value_loss = 9.819732666015625 | time/iterations = 200 | rollout/ep_rew_mean = 22.5 | rollout/ep_len_mean = 22.5 | time/fps = 74 | time/time_elapsed = 13 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:03: [A2C[worker: 4]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.024085521697998047 | train/entropy_loss = -0.6818056106567383 | train/policy_loss = -0.47277942299842834 | train/value_loss = 66.29539489746094 | time/iterations = 300 | rollout/ep_rew_mean = 21.608695652173914 | rollout/ep_len_mean = 21.608695652173914 | time/fps = 61 | time/time_elapsed = 24 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.024085521697998047 | train/entropy_loss = -0.6818056106567383 | train/policy_loss = -0.47277942299842834 | train/value_loss = 66.29539489746094 | time/iterations = 300 | rollout/ep_rew_mean = 21.608695652173914 | rollout/ep_len_mean = 21.608695652173914 | time/fps = 61 | time/time_elapsed = 24 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:03: [A2C[worker: 1]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.026349306106567383 | train/entropy_loss = -0.5538308620452881 | train/policy_loss = -6.996323585510254 | train/value_loss = 264.166748046875 | time/iterations = 300 | rollout/ep_rew_mean = 48.16129032258065 | rollout/ep_len_mean = 48.16129032258065 | time/fps = 60 | time/time_elapsed = 24 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.026349306106567383 | train/entropy_loss = -0.5538308620452881 | train/policy_loss = -6.996323585510254 | train/value_loss = 264.166748046875 | time/iterations = 300 | rollout/ep_rew_mean = 48.16129032258065 | rollout/ep_len_mean = 48.16129032258065 | time/fps = 60 | time/time_elapsed = 24 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:03: [A2C[worker: 2]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.05505228042602539 | train/entropy_loss = -0.5886244773864746 | train/policy_loss = 1.7522426843643188 | train/value_loss = 7.098322868347168 | time/iterations = 300 | rollout/ep_rew_mean = 27.754716981132077 | rollout/ep_len_mean = 27.754716981132077 | time/fps = 59 | time/time_elapsed = 25 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.05505228042602539 | train/entropy_loss = -0.5886244773864746 | train/policy_loss = 1.7522426843643188 | train/value_loss = 7.098322868347168 | time/iterations = 300 | rollout/ep_rew_mean = 27.754716981132077 | rollout/ep_len_mean = 27.754716981132077 | time/fps = 59 | time/time_elapsed = 25 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:03: [A2C[worker: 0]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.0394892692565918 | train/entropy_loss = -0.6379895210266113 | train/policy_loss = 1.279188632965088 | train/value_loss = 6.436897277832031 | time/iterations = 300 | rollout/ep_rew_mean = 39.833333333333336 | rollout/ep_len_mean = 39.833333333333336 | time/fps = 58 | time/time_elapsed = 25 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.0394892692565918 | train/entropy_loss = -0.6379895210266113 | train/policy_loss = 1.279188632965088 | train/value_loss = 6.436897277832031 | time/iterations = 300 | rollout/ep_rew_mean = 39.833333333333336 | rollout/ep_len_mean = 39.833333333333336 | time/fps = 58 | time/time_elapsed = 25 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:03: [A2C[worker: 3]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.006180405616760254 | train/entropy_loss = -0.6868502497673035 | train/policy_loss = 1.601068139076233 | train/value_loss = 7.350211143493652 | time/iterations = 300 | rollout/ep_rew_mean = 21.823529411764707 | rollout/ep_len_mean = 21.823529411764707 | time/fps = 58 | time/time_elapsed = 25 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.006180405616760254 | train/entropy_loss = -0.6868502497673035 | train/policy_loss = 1.601068139076233 | train/value_loss = 7.350211143493652 | time/iterations = 300 | rollout/ep_rew_mean = 21.823529411764707 | rollout/ep_len_mean = 21.823529411764707 | time/fps = 58 | time/time_elapsed = 25 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:03: [A2C[worker: 4]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.029446959495544434 | train/entropy_loss = -0.657664954662323 | train/policy_loss = 1.4489883184432983 | train/value_loss = 6.364837646484375 | time/iterations = 400 | rollout/ep_rew_mean = 23.305882352941175 | rollout/ep_len_mean = 23.305882352941175 | time/fps = 63 | time/time_elapsed = 31 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.029446959495544434 | train/entropy_loss = -0.657664954662323 | train/policy_loss = 1.4489883184432983 | train/value_loss = 6.364837646484375 | time/iterations = 400 | rollout/ep_rew_mean = 23.305882352941175 | rollout/ep_len_mean = 23.305882352941175 | time/fps = 63 | time/time_elapsed = 31 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:03: [A2C[worker: 1]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.012204170227050781 | train/entropy_loss = -0.4974386692047119 | train/policy_loss = 1.8138096332550049 | train/value_loss = 5.6334919929504395 | time/iterations = 400 | rollout/ep_rew_mean = 50.38461538461539 | rollout/ep_len_mean = 50.38461538461539 | time/fps = 63 | time/time_elapsed = 31 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.012204170227050781 | train/entropy_loss = -0.4974386692047119 | train/policy_loss = 1.8138096332550049 | train/value_loss = 5.6334919929504395 | time/iterations = 400 | rollout/ep_rew_mean = 50.38461538461539 | rollout/ep_len_mean = 50.38461538461539 | time/fps = 63 | time/time_elapsed = 31 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:03: [A2C[worker: 0]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.2329103946685791 | train/entropy_loss = -0.6210893392562866 | train/policy_loss = 0.8986666798591614 | train/value_loss = 6.403750419616699 | time/iterations = 400 | rollout/ep_rew_mean = 43.58139534883721 | rollout/ep_len_mean = 43.58139534883721 | time/fps = 62 | time/time_elapsed = 32 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.2329103946685791 | train/entropy_loss = -0.6210893392562866 | train/policy_loss = 0.8986666798591614 | train/value_loss = 6.403750419616699 | time/iterations = 400 | rollout/ep_rew_mean = 43.58139534883721 | rollout/ep_len_mean = 43.58139534883721 | time/fps = 62 | time/time_elapsed = 32 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:03: [A2C[worker: 2]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.12567138671875 | train/entropy_loss = -0.5161430239677429 | train/policy_loss = 1.740846037864685 | train/value_loss = 7.350391387939453 | time/iterations = 400 | rollout/ep_rew_mean = 26.506666666666668 | rollout/ep_len_mean = 26.506666666666668 | time/fps = 62 | time/time_elapsed = 31 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.12567138671875 | train/entropy_loss = -0.5161430239677429 | train/policy_loss = 1.740846037864685 | train/value_loss = 7.350391387939453 | time/iterations = 400 | rollout/ep_rew_mean = 26.506666666666668 | rollout/ep_len_mean = 26.506666666666668 | time/fps = 62 | time/time_elapsed = 31 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:03: [A2C[worker: 3]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.12223219871520996 | train/entropy_loss = -0.6492897868156433 | train/policy_loss = 1.5022979974746704 | train/value_loss = 4.8801798820495605 | time/iterations = 400 | rollout/ep_rew_mean = 24.085365853658537 | rollout/ep_len_mean = 24.085365853658537 | time/fps = 61 | time/time_elapsed = 32 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.12223219871520996 | train/entropy_loss = -0.6492897868156433 | train/policy_loss = 1.5022979974746704 | train/value_loss = 4.8801798820495605 | time/iterations = 400 | rollout/ep_rew_mean = 24.085365853658537 | rollout/ep_len_mean = 24.085365853658537 | time/fps = 61 | time/time_elapsed = 32 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:04: [A2C[worker: 4]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.0004974007606506348 | train/entropy_loss = -0.45792898535728455 | train/policy_loss = 1.664541244506836 | train/value_loss = 5.6013054847717285 | time/iterations = 500 | rollout/ep_rew_mean = 26.54945054945055 | rollout/ep_len_mean = 26.54945054945055 | time/fps = 65 | time/time_elapsed = 38 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.0004974007606506348 | train/entropy_loss = -0.45792898535728455 | train/policy_loss = 1.664541244506836 | train/value_loss = 5.6013054847717285 | time/iterations = 500 | rollout/ep_rew_mean = 26.54945054945055 | rollout/ep_len_mean = 26.54945054945055 | time/fps = 65 | time/time_elapsed = 38 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:04: [A2C[worker: 1]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.010665535926818848 | train/entropy_loss = -0.666008710861206 | train/policy_loss = 1.049622893333435 | train/value_loss = 4.923501491546631 | time/iterations = 500 | rollout/ep_rew_mean = 56.48837209302326 | rollout/ep_len_mean = 56.48837209302326 | time/fps = 63 | time/time_elapsed = 39 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.010665535926818848 | train/entropy_loss = -0.666008710861206 | train/policy_loss = 1.049622893333435 | train/value_loss = 4.923501491546631 | time/iterations = 500 | rollout/ep_rew_mean = 56.48837209302326 | rollout/ep_len_mean = 56.48837209302326 | time/fps = 63 | time/time_elapsed = 39 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:04: [A2C[worker: 0]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.020305752754211426 | train/entropy_loss = -0.611649215221405 | train/policy_loss = 1.5021315813064575 | train/value_loss = 5.1855950355529785 | time/iterations = 500 | rollout/ep_rew_mean = 46.14 | rollout/ep_len_mean = 46.14 | time/fps = 63 | time/time_elapsed = 39 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.020305752754211426 | train/entropy_loss = -0.611649215221405 | train/policy_loss = 1.5021315813064575 | train/value_loss = 5.1855950355529785 | time/iterations = 500 | rollout/ep_rew_mean = 46.14 | rollout/ep_len_mean = 46.14 | time/fps = 63 | time/time_elapsed = 39 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:04: [A2C[worker: 2]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.08393669128417969 | train/entropy_loss = -0.6313813924789429 | train/policy_loss = 1.0294907093048096 | train/value_loss = 6.311558723449707 | time/iterations = 500 | rollout/ep_rew_mean = 25.916666666666668 | rollout/ep_len_mean = 25.916666666666668 | time/fps = 63 | time/time_elapsed = 39 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.08393669128417969 | train/entropy_loss = -0.6313813924789429 | train/policy_loss = 1.0294907093048096 | train/value_loss = 6.311558723449707 | time/iterations = 500 | rollout/ep_rew_mean = 25.916666666666668 | rollout/ep_len_mean = 25.916666666666668 | time/fps = 63 | time/time_elapsed = 39 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:04: [A2C[worker: 3]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.03423571586608887 | train/entropy_loss = -0.5844255685806274 | train/policy_loss = 1.5671504735946655 | train/value_loss = 6.079492568969727 | time/iterations = 500 | rollout/ep_rew_mean = 26.64516129032258 | rollout/ep_len_mean = 26.64516129032258 | time/fps = 62 | time/time_elapsed = 40 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.03423571586608887 | train/entropy_loss = -0.5844255685806274 | train/policy_loss = 1.5671504735946655 | train/value_loss = 6.079492568969727 | time/iterations = 500 | rollout/ep_rew_mean = 26.64516129032258 | rollout/ep_len_mean = 26.64516129032258 | time/fps = 62 | time/time_elapsed = 40 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:04: [A2C[worker: 4]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.006161808967590332 | train/entropy_loss = -0.6204568147659302 | train/policy_loss = 1.0654127597808838 | train/value_loss = 4.9905290603637695 | time/iterations = 600 | rollout/ep_rew_mean = 30.536082474226806 | rollout/ep_len_mean = 30.536082474226806 | time/fps = 65 | time/time_elapsed = 45 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.006161808967590332 | train/entropy_loss = -0.6204568147659302 | train/policy_loss = 1.0654127597808838 | train/value_loss = 4.9905290603637695 | time/iterations = 600 | rollout/ep_rew_mean = 30.536082474226806 | rollout/ep_len_mean = 30.536082474226806 | time/fps = 65 | time/time_elapsed = 45 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:04: [A2C[worker: 0]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.025348305702209473 | train/entropy_loss = -0.6067658066749573 | train/policy_loss = 0.7396023869514465 | train/value_loss = 4.569132328033447 | time/iterations = 600 | rollout/ep_rew_mean = 51.44827586206897 | rollout/ep_len_mean = 51.44827586206897 | time/fps = 64 | time/time_elapsed = 46 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.025348305702209473 | train/entropy_loss = -0.6067658066749573 | train/policy_loss = 0.7396023869514465 | train/value_loss = 4.569132328033447 | time/iterations = 600 | rollout/ep_rew_mean = 51.44827586206897 | rollout/ep_len_mean = 51.44827586206897 | time/fps = 64 | time/time_elapsed = 46 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:04: [A2C[worker: 1]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.00202023983001709 | train/entropy_loss = -0.46341484785079956 | train/policy_loss = 1.054260492324829 | train/value_loss = 4.528055667877197 | time/iterations = 600 | rollout/ep_rew_mean = 59.06 | rollout/ep_len_mean = 59.06 | time/fps = 64 | time/time_elapsed = 46 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.00202023983001709 | train/entropy_loss = -0.46341484785079956 | train/policy_loss = 1.054260492324829 | train/value_loss = 4.528055667877197 | time/iterations = 600 | rollout/ep_rew_mean = 59.06 | rollout/ep_len_mean = 59.06 | time/fps = 64 | time/time_elapsed = 46 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:04: [A2C[worker: 2]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.002111077308654785 | train/entropy_loss = -0.5847312808036804 | train/policy_loss = 1.4264709949493408 | train/value_loss = 5.50515079498291 | time/iterations = 600 | rollout/ep_rew_mean = 24.36 | rollout/ep_len_mean = 24.36 | time/fps = 64 | time/time_elapsed = 46 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.002111077308654785 | train/entropy_loss = -0.5847312808036804 | train/policy_loss = 1.4264709949493408 | train/value_loss = 5.50515079498291 | time/iterations = 600 | rollout/ep_rew_mean = 24.36 | rollout/ep_len_mean = 24.36 | time/fps = 64 | time/time_elapsed = 46 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:04: [A2C[worker: 3]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.01974344253540039 | train/entropy_loss = -0.6568433046340942 | train/policy_loss = 1.338356852531433 | train/value_loss = 5.218754768371582 | time/iterations = 600 | rollout/ep_rew_mean = 29.34 | rollout/ep_len_mean = 29.34 | time/fps = 63 | time/time_elapsed = 47 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.01974344253540039 | train/entropy_loss = -0.6568433046340942 | train/policy_loss = 1.338356852531433 | train/value_loss = 5.218754768371582 | time/iterations = 600 | rollout/ep_rew_mean = 29.34 | rollout/ep_len_mean = 29.34 | time/fps = 63 | time/time_elapsed = 47 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:04: [A2C[worker: 4]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.0006470084190368652 | train/entropy_loss = -0.5494086146354675 | train/policy_loss = -0.9461057782173157 | train/value_loss = 244.5751190185547 | time/iterations = 700 | rollout/ep_rew_mean = 34.31 | rollout/ep_len_mean = 34.31 | time/fps = 66 | time/time_elapsed = 52 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.0006470084190368652 | train/entropy_loss = -0.5494086146354675 | train/policy_loss = -0.9461057782173157 | train/value_loss = 244.5751190185547 | time/iterations = 700 | rollout/ep_rew_mean = 34.31 | rollout/ep_len_mean = 34.31 | time/fps = 66 | time/time_elapsed = 52 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:04: [A2C[worker: 0]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.002600252628326416 | train/entropy_loss = -0.6510369181632996 | train/policy_loss = 0.8523088693618774 | train/value_loss = 4.016271591186523 | time/iterations = 700 | rollout/ep_rew_mean = 54.203125 | rollout/ep_len_mean = 54.203125 | time/fps = 65 | time/time_elapsed = 53 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.002600252628326416 | train/entropy_loss = -0.6510369181632996 | train/policy_loss = 0.8523088693618774 | train/value_loss = 4.016271591186523 | time/iterations = 700 | rollout/ep_rew_mean = 54.203125 | rollout/ep_len_mean = 54.203125 | time/fps = 65 | time/time_elapsed = 53 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:04: [A2C[worker: 1]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.013109147548675537 | train/entropy_loss = -0.5005780458450317 | train/policy_loss = 0.8601482510566711 | train/value_loss = 3.864478349685669 | time/iterations = 700 | rollout/ep_rew_mean = 64.20370370370371 | rollout/ep_len_mean = 64.20370370370371 | time/fps = 65 | time/time_elapsed = 53 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.013109147548675537 | train/entropy_loss = -0.5005780458450317 | train/policy_loss = 0.8601482510566711 | train/value_loss = 3.864478349685669 | time/iterations = 700 | rollout/ep_rew_mean = 64.20370370370371 | rollout/ep_len_mean = 64.20370370370371 | time/fps = 65 | time/time_elapsed = 53 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:04: [A2C[worker: 2]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.1576550006866455 | train/entropy_loss = -0.4594573378562927 | train/policy_loss = 0.5053967833518982 | train/value_loss = 5.415733814239502 | time/iterations = 700 | rollout/ep_rew_mean = 24.32 | rollout/ep_len_mean = 24.32 | time/fps = 65 | time/time_elapsed = 53 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.1576550006866455 | train/entropy_loss = -0.4594573378562927 | train/policy_loss = 0.5053967833518982 | train/value_loss = 5.415733814239502 | time/iterations = 700 | rollout/ep_rew_mean = 24.32 | rollout/ep_len_mean = 24.32 | time/fps = 65 | time/time_elapsed = 53 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:04: [A2C[worker: 3]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.007359862327575684 | train/entropy_loss = -0.6558225750923157 | train/policy_loss = 1.2803071737289429 | train/value_loss = 4.652679443359375 | time/iterations = 700 | rollout/ep_rew_mean = 32.61 | rollout/ep_len_mean = 32.61 | time/fps = 64 | time/time_elapsed = 54 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.007359862327575684 | train/entropy_loss = -0.6558225750923157 | train/policy_loss = 1.2803071737289429 | train/value_loss = 4.652679443359375 | time/iterations = 700 | rollout/ep_rew_mean = 32.61 | rollout/ep_len_mean = 32.61 | time/fps = 64 | time/time_elapsed = 54 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:04: [A2C[worker: 4]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -3.445148468017578e-05 | train/entropy_loss = -0.647396445274353 | train/policy_loss = 0.8619315028190613 | train/value_loss = 3.8824729919433594 | time/iterations = 800 | rollout/ep_rew_mean = 37.24 | rollout/ep_len_mean = 37.24 | time/fps = 68 | time/time_elapsed = 58 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -3.445148468017578e-05 | train/entropy_loss = -0.647396445274353 | train/policy_loss = 0.8619315028190613 | train/value_loss = 3.8824729919433594 | time/iterations = 800 | rollout/ep_rew_mean = 37.24 | rollout/ep_len_mean = 37.24 | time/fps = 68 | time/time_elapsed = 58 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:04: [A2C[worker: 0]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0019816160202026367 | train/entropy_loss = -0.2970442771911621 | train/policy_loss = 2.418750524520874 | train/value_loss = 3.50130033493042 | time/iterations = 800 | rollout/ep_rew_mean = 56.28169014084507 | rollout/ep_len_mean = 56.28169014084507 | time/fps = 67 | time/time_elapsed = 59 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0019816160202026367 | train/entropy_loss = -0.2970442771911621 | train/policy_loss = 2.418750524520874 | train/value_loss = 3.50130033493042 | time/iterations = 800 | rollout/ep_rew_mean = 56.28169014084507 | rollout/ep_len_mean = 56.28169014084507 | time/fps = 67 | time/time_elapsed = 59 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:04: [A2C[worker: 1]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.007029175758361816 | train/entropy_loss = -0.5473268628120422 | train/policy_loss = 0.8275980949401855 | train/value_loss = 3.409078598022461 | time/iterations = 800 | rollout/ep_rew_mean = 64.9672131147541 | rollout/ep_len_mean = 64.9672131147541 | time/fps = 67 | time/time_elapsed = 59 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.007029175758361816 | train/entropy_loss = -0.5473268628120422 | train/policy_loss = 0.8275980949401855 | train/value_loss = 3.409078598022461 | time/iterations = 800 | rollout/ep_rew_mean = 64.9672131147541 | rollout/ep_len_mean = 64.9672131147541 | time/fps = 67 | time/time_elapsed = 59 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:04: [A2C[worker: 2]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0031331777572631836 | train/entropy_loss = -0.6422523260116577 | train/policy_loss = 0.8384286165237427 | train/value_loss = 4.481098175048828 | time/iterations = 800 | rollout/ep_rew_mean = 26.45 | rollout/ep_len_mean = 26.45 | time/fps = 67 | time/time_elapsed = 59 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0031331777572631836 | train/entropy_loss = -0.6422523260116577 | train/policy_loss = 0.8384286165237427 | train/value_loss = 4.481098175048828 | time/iterations = 800 | rollout/ep_rew_mean = 26.45 | rollout/ep_len_mean = 26.45 | time/fps = 67 | time/time_elapsed = 59 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:04: [A2C[worker: 3]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.007161259651184082 | train/entropy_loss = -0.6673224568367004 | train/policy_loss = 0.9842885732650757 | train/value_loss = 4.105459213256836 | time/iterations = 800 | rollout/ep_rew_mean = 34.55 | rollout/ep_len_mean = 34.55 | time/fps = 65 | time/time_elapsed = 60 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.007161259651184082 | train/entropy_loss = -0.6673224568367004 | train/policy_loss = 0.9842885732650757 | train/value_loss = 4.105459213256836 | time/iterations = 800 | rollout/ep_rew_mean = 34.55 | rollout/ep_len_mean = 34.55 | time/fps = 65 | time/time_elapsed = 60 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:04: [A2C[worker: 4]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0004991888999938965 | train/entropy_loss = -0.5801981091499329 | train/policy_loss = 0.7801006436347961 | train/value_loss = 3.3824009895324707 | time/iterations = 900 | rollout/ep_rew_mean = 41.56 | rollout/ep_len_mean = 41.56 | time/fps = 69 | time/time_elapsed = 64 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0004991888999938965 | train/entropy_loss = -0.5801981091499329 | train/policy_loss = 0.7801006436347961 | train/value_loss = 3.3824009895324707 | time/iterations = 900 | rollout/ep_rew_mean = 41.56 | rollout/ep_len_mean = 41.56 | time/fps = 69 | time/time_elapsed = 64 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:04: [A2C[worker: 0]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.003604412078857422 | train/entropy_loss = -0.620263934135437 | train/policy_loss = 0.8255172967910767 | train/value_loss = 3.029313802719116 | time/iterations = 900 | rollout/ep_rew_mean = 58.16883116883117 | rollout/ep_len_mean = 58.16883116883117 | time/fps = 68 | time/time_elapsed = 65 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.003604412078857422 | train/entropy_loss = -0.620263934135437 | train/policy_loss = 0.8255172967910767 | train/value_loss = 3.029313802719116 | time/iterations = 900 | rollout/ep_rew_mean = 58.16883116883117 | rollout/ep_len_mean = 58.16883116883117 | time/fps = 68 | time/time_elapsed = 65 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:04: [A2C[worker: 1]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0020982027053833008 | train/entropy_loss = -0.5972659587860107 | train/policy_loss = 0.6739786863327026 | train/value_loss = 2.9078664779663086 | time/iterations = 900 | rollout/ep_rew_mean = 68.95384615384616 | rollout/ep_len_mean = 68.95384615384616 | time/fps = 68 | time/time_elapsed = 65 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0020982027053833008 | train/entropy_loss = -0.5972659587860107 | train/policy_loss = 0.6739786863327026 | train/value_loss = 2.9078664779663086 | time/iterations = 900 | rollout/ep_rew_mean = 68.95384615384616 | rollout/ep_len_mean = 68.95384615384616 | time/fps = 68 | time/time_elapsed = 65 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:04: [A2C[worker: 2]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.06085681915283203 | train/entropy_loss = -0.40550631284713745 | train/policy_loss = 0.6835749745368958 | train/value_loss = 4.284990310668945 | time/iterations = 900 | rollout/ep_rew_mean = 28.12 | rollout/ep_len_mean = 28.12 | time/fps = 68 | time/time_elapsed = 66 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.06085681915283203 | train/entropy_loss = -0.40550631284713745 | train/policy_loss = 0.6835749745368958 | train/value_loss = 4.284990310668945 | time/iterations = 900 | rollout/ep_rew_mean = 28.12 | rollout/ep_len_mean = 28.12 | time/fps = 68 | time/time_elapsed = 66 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:04: [A2C[worker: 3]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.000897526741027832 | train/entropy_loss = -0.6535228490829468 | train/policy_loss = 0.9760711789131165 | train/value_loss = 3.5948593616485596 | time/iterations = 900 | rollout/ep_rew_mean = 39.66 | rollout/ep_len_mean = 39.66 | time/fps = 67 | time/time_elapsed = 66 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.000897526741027832 | train/entropy_loss = -0.6535228490829468 | train/policy_loss = 0.9760711789131165 | train/value_loss = 3.5948593616485596 | time/iterations = 900 | rollout/ep_rew_mean = 39.66 | rollout/ep_len_mean = 39.66 | time/fps = 67 | time/time_elapsed = 66 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:04: [A2C[worker: 4]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 6.341934204101562e-05 | train/entropy_loss = -0.6093338131904602 | train/policy_loss = 0.7835429310798645 | train/value_loss = 2.9375407695770264 | time/iterations = 1000 | rollout/ep_rew_mean = 45.1 | rollout/ep_len_mean = 45.1 | time/fps = 70 | time/time_elapsed = 70 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 6.341934204101562e-05 | train/entropy_loss = -0.6093338131904602 | train/policy_loss = 0.7835429310798645 | train/value_loss = 2.9375407695770264 | time/iterations = 1000 | rollout/ep_rew_mean = 45.1 | rollout/ep_len_mean = 45.1 | time/fps = 70 | time/time_elapsed = 70 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:04: [A2C[worker: 0]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0010474920272827148 | train/entropy_loss = -0.5668686628341675 | train/policy_loss = 0.6538982391357422 | train/value_loss = 2.565509796142578 | time/iterations = 1000 | rollout/ep_rew_mean = 61.15 | rollout/ep_len_mean = 61.15 | time/fps = 70 | time/time_elapsed = 71 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0010474920272827148 | train/entropy_loss = -0.5668686628341675 | train/policy_loss = 0.6538982391357422 | train/value_loss = 2.565509796142578 | time/iterations = 1000 | rollout/ep_rew_mean = 61.15 | rollout/ep_len_mean = 61.15 | time/fps = 70 | time/time_elapsed = 71 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:04: [A2C[worker: 1]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0006040334701538086 | train/entropy_loss = -0.581960916519165 | train/policy_loss = 0.7837586402893066 | train/value_loss = 2.439272403717041 | time/iterations = 1000 | rollout/ep_rew_mean = 71.49253731343283 | rollout/ep_len_mean = 71.49253731343283 | time/fps = 69 | time/time_elapsed = 71 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0006040334701538086 | train/entropy_loss = -0.581960916519165 | train/policy_loss = 0.7837586402893066 | train/value_loss = 2.439272403717041 | time/iterations = 1000 | rollout/ep_rew_mean = 71.49253731343283 | rollout/ep_len_mean = 71.49253731343283 | time/fps = 69 | time/time_elapsed = 71 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:04: [A2C[worker: 2]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0028002262115478516 | train/entropy_loss = -0.5838087797164917 | train/policy_loss = 0.6489311456680298 | train/value_loss = 3.576641082763672 | time/iterations = 1000 | rollout/ep_rew_mean = 30.37 | rollout/ep_len_mean = 30.37 | time/fps = 69 | time/time_elapsed = 71 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0028002262115478516 | train/entropy_loss = -0.5838087797164917 | train/policy_loss = 0.6489311456680298 | train/value_loss = 3.576641082763672 | time/iterations = 1000 | rollout/ep_rew_mean = 30.37 | rollout/ep_len_mean = 30.37 | time/fps = 69 | time/time_elapsed = 71 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:04: [A2C[worker: 3]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.0010719895362854004 | train/entropy_loss = -0.5551344156265259 | train/policy_loss = -41.85040283203125 | train/value_loss = 1852.0758056640625 | time/iterations = 1000 | rollout/ep_rew_mean = 43.55 | rollout/ep_len_mean = 43.55 | time/fps = 68 | time/time_elapsed = 73 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.0010719895362854004 | train/entropy_loss = -0.5551344156265259 | train/policy_loss = -41.85040283203125 | train/value_loss = 1852.0758056640625 | time/iterations = 1000 | rollout/ep_rew_mean = 43.55 | rollout/ep_len_mean = 43.55 | time/fps = 68 | time/time_elapsed = 73 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:04: [A2C[worker: 4]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0001265406608581543 | train/entropy_loss = -0.6416239738464355 | train/policy_loss = 0.7211331129074097 | train/value_loss = 2.5047407150268555 | time/iterations = 1100 | rollout/ep_rew_mean = 48.43 | rollout/ep_len_mean = 48.43 | time/fps = 71 | time/time_elapsed = 76 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0001265406608581543 | train/entropy_loss = -0.6416239738464355 | train/policy_loss = 0.7211331129074097 | train/value_loss = 2.5047407150268555 | time/iterations = 1100 | rollout/ep_rew_mean = 48.43 | rollout/ep_len_mean = 48.43 | time/fps = 71 | time/time_elapsed = 76 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:04: [A2C[worker: 0]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0006551742553710938 | train/entropy_loss = -0.5471089482307434 | train/policy_loss = 0.7310547232627869 | train/value_loss = 2.137202739715576 | time/iterations = 1100 | rollout/ep_rew_mean = 65.87951807228916 | rollout/ep_len_mean = 65.87951807228916 | time/fps = 70 | time/time_elapsed = 77 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0006551742553710938 | train/entropy_loss = -0.5471089482307434 | train/policy_loss = 0.7310547232627869 | train/value_loss = 2.137202739715576 | time/iterations = 1100 | rollout/ep_rew_mean = 65.87951807228916 | rollout/ep_len_mean = 65.87951807228916 | time/fps = 70 | time/time_elapsed = 77 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:04: [A2C[worker: 1]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0008833408355712891 | train/entropy_loss = -0.5460157990455627 | train/policy_loss = 0.877358615398407 | train/value_loss = 2.024580478668213 | time/iterations = 1100 | rollout/ep_rew_mean = 77.4225352112676 | rollout/ep_len_mean = 77.4225352112676 | time/fps = 70 | time/time_elapsed = 77 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0008833408355712891 | train/entropy_loss = -0.5460157990455627 | train/policy_loss = 0.877358615398407 | train/value_loss = 2.024580478668213 | time/iterations = 1100 | rollout/ep_rew_mean = 77.4225352112676 | rollout/ep_len_mean = 77.4225352112676 | time/fps = 70 | time/time_elapsed = 77 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:04: [A2C[worker: 2]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0014206171035766602 | train/entropy_loss = -0.40027159452438354 | train/policy_loss = 1.769018530845642 | train/value_loss = 3.1534273624420166 | time/iterations = 1100 | rollout/ep_rew_mean = 35.17 | rollout/ep_len_mean = 35.17 | time/fps = 70 | time/time_elapsed = 78 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0014206171035766602 | train/entropy_loss = -0.40027159452438354 | train/policy_loss = 1.769018530845642 | train/value_loss = 3.1534273624420166 | time/iterations = 1100 | rollout/ep_rew_mean = 35.17 | rollout/ep_len_mean = 35.17 | time/fps = 70 | time/time_elapsed = 78 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:04: [A2C[worker: 3]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.00010156631469726562 | train/entropy_loss = -0.6741737723350525 | train/policy_loss = 0.820309042930603 | train/value_loss = 2.6575536727905273 | time/iterations = 1100 | rollout/ep_rew_mean = 47.09 | rollout/ep_len_mean = 47.09 | time/fps = 69 | time/time_elapsed = 79 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.00010156631469726562 | train/entropy_loss = -0.6741737723350525 | train/policy_loss = 0.820309042930603 | train/value_loss = 2.6575536727905273 | time/iterations = 1100 | rollout/ep_rew_mean = 47.09 | rollout/ep_len_mean = 47.09 | time/fps = 69 | time/time_elapsed = 79 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:04: [A2C[worker: 4]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 5.91278076171875e-05 | train/entropy_loss = -0.5402065515518188 | train/policy_loss = 0.5685919523239136 | train/value_loss = 2.133406162261963 | time/iterations = 1200 | rollout/ep_rew_mean = 51.67 | rollout/ep_len_mean = 51.67 | time/fps = 72 | time/time_elapsed = 82 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 5.91278076171875e-05 | train/entropy_loss = -0.5402065515518188 | train/policy_loss = 0.5685919523239136 | train/value_loss = 2.133406162261963 | time/iterations = 1200 | rollout/ep_rew_mean = 51.67 | rollout/ep_len_mean = 51.67 | time/fps = 72 | time/time_elapsed = 82 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:04: [A2C[worker: 1]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.003794550895690918 | train/entropy_loss = -0.434924453496933 | train/policy_loss = 1.13424813747406 | train/value_loss = 1.6227813959121704 | time/iterations = 1200 | rollout/ep_rew_mean = 79.52054794520548 | rollout/ep_len_mean = 79.52054794520548 | time/fps = 71 | time/time_elapsed = 83 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.003794550895690918 | train/entropy_loss = -0.434924453496933 | train/policy_loss = 1.13424813747406 | train/value_loss = 1.6227813959121704 | time/iterations = 1200 | rollout/ep_rew_mean = 79.52054794520548 | rollout/ep_len_mean = 79.52054794520548 | time/fps = 71 | time/time_elapsed = 83 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:04: [A2C[worker: 0]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00013327598571777344 | train/entropy_loss = -0.49769195914268494 | train/policy_loss = 0.8595694303512573 | train/value_loss = 1.7311878204345703 | time/iterations = 1200 | rollout/ep_rew_mean = 68.8452380952381 | rollout/ep_len_mean = 68.8452380952381 | time/fps = 71 | time/time_elapsed = 83 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00013327598571777344 | train/entropy_loss = -0.49769195914268494 | train/policy_loss = 0.8595694303512573 | train/value_loss = 1.7311878204345703 | time/iterations = 1200 | rollout/ep_rew_mean = 68.8452380952381 | rollout/ep_len_mean = 68.8452380952381 | time/fps = 71 | time/time_elapsed = 83 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:04: [A2C[worker: 2]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.0012224912643432617 | train/entropy_loss = -0.6082350611686707 | train/policy_loss = 0.5871256589889526 | train/value_loss = 2.6694185733795166 | time/iterations = 1200 | rollout/ep_rew_mean = 38.56 | rollout/ep_len_mean = 38.56 | time/fps = 71 | time/time_elapsed = 84 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.0012224912643432617 | train/entropy_loss = -0.6082350611686707 | train/policy_loss = 0.5871256589889526 | train/value_loss = 2.6694185733795166 | time/iterations = 1200 | rollout/ep_rew_mean = 38.56 | rollout/ep_len_mean = 38.56 | time/fps = 71 | time/time_elapsed = 84 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:04: [A2C[worker: 3]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.0006107091903686523 | train/entropy_loss = -0.6570874452590942 | train/policy_loss = 0.6260218620300293 | train/value_loss = 2.231214761734009 | time/iterations = 1200 | rollout/ep_rew_mean = 50.96 | rollout/ep_len_mean = 50.96 | time/fps = 70 | time/time_elapsed = 85 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.0006107091903686523 | train/entropy_loss = -0.6570874452590942 | train/policy_loss = 0.6260218620300293 | train/value_loss = 2.231214761734009 | time/iterations = 1200 | rollout/ep_rew_mean = 50.96 | rollout/ep_len_mean = 50.96 | time/fps = 70 | time/time_elapsed = 85 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:04: [A2C[worker: 4]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0002397298812866211 | train/entropy_loss = -0.5235929489135742 | train/policy_loss = 0.930378258228302 | train/value_loss = 1.7835988998413086 | time/iterations = 1300 | rollout/ep_rew_mean = 54.75 | rollout/ep_len_mean = 54.75 | time/fps = 73 | time/time_elapsed = 88 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0002397298812866211 | train/entropy_loss = -0.5235929489135742 | train/policy_loss = 0.930378258228302 | train/value_loss = 1.7835988998413086 | time/iterations = 1300 | rollout/ep_rew_mean = 54.75 | rollout/ep_len_mean = 54.75 | time/fps = 73 | time/time_elapsed = 88 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:04: [A2C[worker: 0]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.0006383061408996582 | train/entropy_loss = -0.5383577346801758 | train/policy_loss = 0.47434574365615845 | train/value_loss = 1.3729300498962402 | time/iterations = 1300 | rollout/ep_rew_mean = 74.48837209302326 | rollout/ep_len_mean = 74.48837209302326 | time/fps = 72 | time/time_elapsed = 90 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.0006383061408996582 | train/entropy_loss = -0.5383577346801758 | train/policy_loss = 0.47434574365615845 | train/value_loss = 1.3729300498962402 | time/iterations = 1300 | rollout/ep_rew_mean = 74.48837209302326 | rollout/ep_len_mean = 74.48837209302326 | time/fps = 72 | time/time_elapsed = 90 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:04: [A2C[worker: 1]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.00038617849349975586 | train/entropy_loss = -0.3042911887168884 | train/policy_loss = 1.147226333618164 | train/value_loss = 1.30494225025177 | time/iterations = 1300 | rollout/ep_rew_mean = 84.5657894736842 | rollout/ep_len_mean = 84.5657894736842 | time/fps = 72 | time/time_elapsed = 89 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.00038617849349975586 | train/entropy_loss = -0.3042911887168884 | train/policy_loss = 1.147226333618164 | train/value_loss = 1.30494225025177 | time/iterations = 1300 | rollout/ep_rew_mean = 84.5657894736842 | rollout/ep_len_mean = 84.5657894736842 | time/fps = 72 | time/time_elapsed = 89 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:04: [A2C[worker: 2]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0037059783935546875 | train/entropy_loss = -0.6250969767570496 | train/policy_loss = -22.01401138305664 | train/value_loss = 2621.413330078125 | time/iterations = 1300 | rollout/ep_rew_mean = 42.8 | rollout/ep_len_mean = 42.8 | time/fps = 71 | time/time_elapsed = 90 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0037059783935546875 | train/entropy_loss = -0.6250969767570496 | train/policy_loss = -22.01401138305664 | train/value_loss = 2621.413330078125 | time/iterations = 1300 | rollout/ep_rew_mean = 42.8 | rollout/ep_len_mean = 42.8 | time/fps = 71 | time/time_elapsed = 90 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:04: [A2C[worker: 3]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -4.887580871582031e-05 | train/entropy_loss = -0.5773941278457642 | train/policy_loss = 0.39283695816993713 | train/value_loss = 1.8412449359893799 | time/iterations = 1300 | rollout/ep_rew_mean = 55.26 | rollout/ep_len_mean = 55.26 | time/fps = 70 | time/time_elapsed = 91 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -4.887580871582031e-05 | train/entropy_loss = -0.5773941278457642 | train/policy_loss = 0.39283695816993713 | train/value_loss = 1.8412449359893799 | time/iterations = 1300 | rollout/ep_rew_mean = 55.26 | rollout/ep_len_mean = 55.26 | time/fps = 70 | time/time_elapsed = 91 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:04: [A2C[worker: 4]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -4.470348358154297e-05 | train/entropy_loss = -0.6317303776741028 | train/policy_loss = 0.657829999923706 | train/value_loss = 1.4363354444503784 | time/iterations = 1400 | rollout/ep_rew_mean = 58.24 | rollout/ep_len_mean = 58.24 | time/fps = 73 | time/time_elapsed = 95 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -4.470348358154297e-05 | train/entropy_loss = -0.6317303776741028 | train/policy_loss = 0.657829999923706 | train/value_loss = 1.4363354444503784 | time/iterations = 1400 | rollout/ep_rew_mean = 58.24 | rollout/ep_len_mean = 58.24 | time/fps = 73 | time/time_elapsed = 95 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:04: [A2C[worker: 0]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.0013037919998168945 | train/entropy_loss = -0.3010948598384857 | train/policy_loss = 1.0445282459259033 | train/value_loss = 1.0565059185028076 | time/iterations = 1400 | rollout/ep_rew_mean = 79.01149425287356 | rollout/ep_len_mean = 79.01149425287356 | time/fps = 72 | time/time_elapsed = 96 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.0013037919998168945 | train/entropy_loss = -0.3010948598384857 | train/policy_loss = 1.0445282459259033 | train/value_loss = 1.0565059185028076 | time/iterations = 1400 | rollout/ep_rew_mean = 79.01149425287356 | rollout/ep_len_mean = 79.01149425287356 | time/fps = 72 | time/time_elapsed = 96 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:05: [A2C[worker: 1]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.00023812055587768555 | train/entropy_loss = -0.5277656316757202 | train/policy_loss = 0.23498420417308807 | train/value_loss = 1.0111385583877563 | time/iterations = 1400 | rollout/ep_rew_mean = 86.65 | rollout/ep_len_mean = 86.65 | time/fps = 72 | time/time_elapsed = 96 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.00023812055587768555 | train/entropy_loss = -0.5277656316757202 | train/policy_loss = 0.23498420417308807 | train/value_loss = 1.0111385583877563 | time/iterations = 1400 | rollout/ep_rew_mean = 86.65 | rollout/ep_len_mean = 86.65 | time/fps = 72 | time/time_elapsed = 96 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:05: [A2C[worker: 2]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -5.412101745605469e-05 | train/entropy_loss = -0.519539475440979 | train/policy_loss = 0.9243170619010925 | train/value_loss = 1.8674389123916626 | time/iterations = 1400 | rollout/ep_rew_mean = 46.42 | rollout/ep_len_mean = 46.42 | time/fps = 72 | time/time_elapsed = 97 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -5.412101745605469e-05 | train/entropy_loss = -0.519539475440979 | train/policy_loss = 0.9243170619010925 | train/value_loss = 1.8674389123916626 | time/iterations = 1400 | rollout/ep_rew_mean = 46.42 | rollout/ep_len_mean = 46.42 | time/fps = 72 | time/time_elapsed = 97 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:05: [A2C[worker: 3]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 3.331899642944336e-05 | train/entropy_loss = -0.6270254254341125 | train/policy_loss = 0.5650692582130432 | train/value_loss = 1.4742841720581055 | time/iterations = 1400 | rollout/ep_rew_mean = 58.74 | rollout/ep_len_mean = 58.74 | time/fps = 71 | time/time_elapsed = 98 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 3.331899642944336e-05 | train/entropy_loss = -0.6270254254341125 | train/policy_loss = 0.5650692582130432 | train/value_loss = 1.4742841720581055 | time/iterations = 1400 | rollout/ep_rew_mean = 58.74 | rollout/ep_len_mean = 58.74 | time/fps = 71 | time/time_elapsed = 98 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:05: [A2C[worker: 4]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -2.384185791015625e-07 | train/entropy_loss = -0.5629721879959106 | train/policy_loss = 0.4819096624851227 | train/value_loss = 1.1278159618377686 | time/iterations = 1500 | rollout/ep_rew_mean = 63.33 | rollout/ep_len_mean = 63.33 | time/fps = 74 | time/time_elapsed = 101 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -2.384185791015625e-07 | train/entropy_loss = -0.5629721879959106 | train/policy_loss = 0.4819096624851227 | train/value_loss = 1.1278159618377686 | time/iterations = 1500 | rollout/ep_rew_mean = 63.33 | rollout/ep_len_mean = 63.33 | time/fps = 74 | time/time_elapsed = 101 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:05: [A2C[worker: 0]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0005526542663574219 | train/entropy_loss = -0.39356887340545654 | train/policy_loss = 0.8053494691848755 | train/value_loss = 0.7831568717956543 | time/iterations = 1500 | rollout/ep_rew_mean = 82.96629213483146 | rollout/ep_len_mean = 82.96629213483146 | time/fps = 73 | time/time_elapsed = 102 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0005526542663574219 | train/entropy_loss = -0.39356887340545654 | train/policy_loss = 0.8053494691848755 | train/value_loss = 0.7831568717956543 | time/iterations = 1500 | rollout/ep_rew_mean = 82.96629213483146 | rollout/ep_len_mean = 82.96629213483146 | time/fps = 73 | time/time_elapsed = 102 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:05: [A2C[worker: 1]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -3.516674041748047e-05 | train/entropy_loss = -0.4007960259914398 | train/policy_loss = 0.6841002702713013 | train/value_loss = 0.7493652105331421 | time/iterations = 1500 | rollout/ep_rew_mean = 89.33734939759036 | rollout/ep_len_mean = 89.33734939759036 | time/fps = 73 | time/time_elapsed = 102 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -3.516674041748047e-05 | train/entropy_loss = -0.4007960259914398 | train/policy_loss = 0.6841002702713013 | train/value_loss = 0.7493652105331421 | time/iterations = 1500 | rollout/ep_rew_mean = 89.33734939759036 | rollout/ep_len_mean = 89.33734939759036 | time/fps = 73 | time/time_elapsed = 102 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:05: [A2C[worker: 2]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -1.4543533325195312e-05 | train/entropy_loss = -0.6384104490280151 | train/policy_loss = 0.4788106381893158 | train/value_loss = 1.555235505104065 | time/iterations = 1500 | rollout/ep_rew_mean = 49.05 | rollout/ep_len_mean = 49.05 | time/fps = 72 | time/time_elapsed = 103 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -1.4543533325195312e-05 | train/entropy_loss = -0.6384104490280151 | train/policy_loss = 0.4788106381893158 | train/value_loss = 1.555235505104065 | time/iterations = 1500 | rollout/ep_rew_mean = 49.05 | rollout/ep_len_mean = 49.05 | time/fps = 72 | time/time_elapsed = 103 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:05: [A2C[worker: 3]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -8.547306060791016e-05 | train/entropy_loss = -0.6409732103347778 | train/policy_loss = 0.483442485332489 | train/value_loss = 1.1670035123825073 | time/iterations = 1500 | rollout/ep_rew_mean = 63.97 | rollout/ep_len_mean = 63.97 | time/fps = 71 | time/time_elapsed = 104 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -8.547306060791016e-05 | train/entropy_loss = -0.6409732103347778 | train/policy_loss = 0.483442485332489 | train/value_loss = 1.1670035123825073 | time/iterations = 1500 | rollout/ep_rew_mean = 63.97 | rollout/ep_len_mean = 63.97 | time/fps = 71 | time/time_elapsed = 104 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:05: [A2C[worker: 4]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -3.719329833984375e-05 | train/entropy_loss = -0.6144925951957703 | train/policy_loss = 0.3446594774723053 | train/value_loss = 0.8460689783096313 | time/iterations = 1600 | rollout/ep_rew_mean = 66.15 | rollout/ep_len_mean = 66.15 | time/fps = 74 | time/time_elapsed = 107 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -3.719329833984375e-05 | train/entropy_loss = -0.6144925951957703 | train/policy_loss = 0.3446594774723053 | train/value_loss = 0.8460689783096313 | time/iterations = 1600 | rollout/ep_rew_mean = 66.15 | rollout/ep_len_mean = 66.15 | time/fps = 74 | time/time_elapsed = 107 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:05: [A2C[worker: 0]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.0006144046783447266 | train/entropy_loss = -0.5419591665267944 | train/policy_loss = 0.3608786463737488 | train/value_loss = 0.5466316938400269 | time/iterations = 1600 | rollout/ep_rew_mean = 84.95555555555555 | rollout/ep_len_mean = 84.95555555555555 | time/fps = 73 | time/time_elapsed = 108 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.0006144046783447266 | train/entropy_loss = -0.5419591665267944 | train/policy_loss = 0.3608786463737488 | train/value_loss = 0.5466316938400269 | time/iterations = 1600 | rollout/ep_rew_mean = 84.95555555555555 | rollout/ep_len_mean = 84.95555555555555 | time/fps = 73 | time/time_elapsed = 108 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:05: [A2C[worker: 1]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.0035498738288879395 | train/entropy_loss = -0.39484772086143494 | train/policy_loss = 0.4491041302680969 | train/value_loss = 0.5261693000793457 | time/iterations = 1600 | rollout/ep_rew_mean = 91.9080459770115 | rollout/ep_len_mean = 91.9080459770115 | time/fps = 73 | time/time_elapsed = 108 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.0035498738288879395 | train/entropy_loss = -0.39484772086143494 | train/policy_loss = 0.4491041302680969 | train/value_loss = 0.5261693000793457 | time/iterations = 1600 | rollout/ep_rew_mean = 91.9080459770115 | rollout/ep_len_mean = 91.9080459770115 | time/fps = 73 | time/time_elapsed = 108 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:05: [A2C[worker: 2]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 1.8417835235595703e-05 | train/entropy_loss = -0.3939523994922638 | train/policy_loss = 0.5356603264808655 | train/value_loss = 1.2522186040878296 | time/iterations = 1600 | rollout/ep_rew_mean = 51.95 | rollout/ep_len_mean = 51.95 | time/fps = 72 | time/time_elapsed = 109 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 1.8417835235595703e-05 | train/entropy_loss = -0.3939523994922638 | train/policy_loss = 0.5356603264808655 | train/value_loss = 1.2522186040878296 | time/iterations = 1600 | rollout/ep_rew_mean = 51.95 | rollout/ep_len_mean = 51.95 | time/fps = 72 | time/time_elapsed = 109 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:05: [A2C[worker: 3]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 6.031990051269531e-05 | train/entropy_loss = -0.6515109539031982 | train/policy_loss = 0.5399571657180786 | train/value_loss = 0.8788954019546509 | time/iterations = 1600 | rollout/ep_rew_mean = 66.23 | rollout/ep_len_mean = 66.23 | time/fps = 72 | time/time_elapsed = 110 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 6.031990051269531e-05 | train/entropy_loss = -0.6515109539031982 | train/policy_loss = 0.5399571657180786 | train/value_loss = 0.8788954019546509 | time/iterations = 1600 | rollout/ep_rew_mean = 66.23 | rollout/ep_len_mean = 66.23 | time/fps = 72 | time/time_elapsed = 110 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:05: [A2C[worker: 4]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -0.00013720989227294922 | train/entropy_loss = -0.5969778895378113 | train/policy_loss = 0.3404299318790436 | train/value_loss = 0.6137812733650208 | time/iterations = 1700 | rollout/ep_rew_mean = 71.93 | rollout/ep_len_mean = 71.93 | time/fps = 74 | time/time_elapsed = 113 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -0.00013720989227294922 | train/entropy_loss = -0.5969778895378113 | train/policy_loss = 0.3404299318790436 | train/value_loss = 0.6137812733650208 | time/iterations = 1700 | rollout/ep_rew_mean = 71.93 | rollout/ep_len_mean = 71.93 | time/fps = 74 | time/time_elapsed = 113 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:05: [A2C[worker: 0]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -4.363059997558594e-05 | train/entropy_loss = -0.5183632969856262 | train/policy_loss = 0.16970346868038177 | train/value_loss = 0.3559702932834625 | time/iterations = 1700 | rollout/ep_rew_mean = 89.1195652173913 | rollout/ep_len_mean = 89.1195652173913 | time/fps = 74 | time/time_elapsed = 114 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -4.363059997558594e-05 | train/entropy_loss = -0.5183632969856262 | train/policy_loss = 0.16970346868038177 | train/value_loss = 0.3559702932834625 | time/iterations = 1700 | rollout/ep_rew_mean = 89.1195652173913 | rollout/ep_len_mean = 89.1195652173913 | time/fps = 74 | time/time_elapsed = 114 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:05: [A2C[worker: 1]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 2.6106834411621094e-05 | train/entropy_loss = -0.5417013764381409 | train/policy_loss = 0.21899256110191345 | train/value_loss = 0.343730628490448 | time/iterations = 1700 | rollout/ep_rew_mean = 94.13483146067416 | rollout/ep_len_mean = 94.13483146067416 | time/fps = 74 | time/time_elapsed = 114 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 2.6106834411621094e-05 | train/entropy_loss = -0.5417013764381409 | train/policy_loss = 0.21899256110191345 | train/value_loss = 0.343730628490448 | time/iterations = 1700 | rollout/ep_rew_mean = 94.13483146067416 | rollout/ep_len_mean = 94.13483146067416 | time/fps = 74 | time/time_elapsed = 114 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:05: [A2C[worker: 2]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 2.5987625122070312e-05 | train/entropy_loss = -0.5634048581123352 | train/policy_loss = 0.45077019929885864 | train/value_loss = 0.9652835726737976 | time/iterations = 1700 | rollout/ep_rew_mean = 55.77 | rollout/ep_len_mean = 55.77 | time/fps = 73 | time/time_elapsed = 115 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 2.5987625122070312e-05 | train/entropy_loss = -0.5634048581123352 | train/policy_loss = 0.45077019929885864 | train/value_loss = 0.9652835726737976 | time/iterations = 1700 | rollout/ep_rew_mean = 55.77 | rollout/ep_len_mean = 55.77 | time/fps = 73 | time/time_elapsed = 115 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:05: [A2C[worker: 3]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -3.826618194580078e-05 | train/entropy_loss = -0.616480827331543 | train/policy_loss = 0.26579156517982483 | train/value_loss = 0.636154055595398 | time/iterations = 1700 | rollout/ep_rew_mean = 72.6 | rollout/ep_len_mean = 72.6 | time/fps = 72 | time/time_elapsed = 116 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -3.826618194580078e-05 | train/entropy_loss = -0.616480827331543 | train/policy_loss = 0.26579156517982483 | train/value_loss = 0.636154055595398 | time/iterations = 1700 | rollout/ep_rew_mean = 72.6 | rollout/ep_len_mean = 72.6 | time/fps = 72 | time/time_elapsed = 116 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:05: [A2C[worker: 4]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 3.17692756652832e-05 | train/entropy_loss = -0.5487431287765503 | train/policy_loss = 0.2559928297996521 | train/value_loss = 0.4128130376338959 | time/iterations = 1800 | rollout/ep_rew_mean = 75.4 | rollout/ep_len_mean = 75.4 | time/fps = 75 | time/time_elapsed = 119 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 3.17692756652832e-05 | train/entropy_loss = -0.5487431287765503 | train/policy_loss = 0.2559928297996521 | train/value_loss = 0.4128130376338959 | time/iterations = 1800 | rollout/ep_rew_mean = 75.4 | rollout/ep_len_mean = 75.4 | time/fps = 75 | time/time_elapsed = 119 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:05: [A2C[worker: 0]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 0.0002471804618835449 | train/entropy_loss = -0.5083008408546448 | train/policy_loss = 0.09616847336292267 | train/value_loss = 0.20609581470489502 | time/iterations = 1800 | rollout/ep_rew_mean = 94.5 | rollout/ep_len_mean = 94.5 | time/fps = 74 | time/time_elapsed = 120 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 0.0002471804618835449 | train/entropy_loss = -0.5083008408546448 | train/policy_loss = 0.09616847336292267 | train/value_loss = 0.20609581470489502 | time/iterations = 1800 | rollout/ep_rew_mean = 94.5 | rollout/ep_len_mean = 94.5 | time/fps = 74 | time/time_elapsed = 120 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:05: [A2C[worker: 1]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -4.565715789794922e-05 | train/entropy_loss = -0.5405830144882202 | train/policy_loss = 0.16117288172245026 | train/value_loss = 0.20194248855113983 | time/iterations = 1800 | rollout/ep_rew_mean = 95.13978494623656 | rollout/ep_len_mean = 95.13978494623656 | time/fps = 74 | time/time_elapsed = 121 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -4.565715789794922e-05 | train/entropy_loss = -0.5405830144882202 | train/policy_loss = 0.16117288172245026 | train/value_loss = 0.20194248855113983 | time/iterations = 1800 | rollout/ep_rew_mean = 95.13978494623656 | rollout/ep_len_mean = 95.13978494623656 | time/fps = 74 | time/time_elapsed = 121 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:05: [A2C[worker: 2]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -2.4318695068359375e-05 | train/entropy_loss = -0.30234771966934204 | train/policy_loss = 0.6040060520172119 | train/value_loss = 0.7058995366096497 | time/iterations = 1800 | rollout/ep_rew_mean = 60.6 | rollout/ep_len_mean = 60.6 | time/fps = 73 | time/time_elapsed = 122 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -2.4318695068359375e-05 | train/entropy_loss = -0.30234771966934204 | train/policy_loss = 0.6040060520172119 | train/value_loss = 0.7058995366096497 | time/iterations = 1800 | rollout/ep_rew_mean = 60.6 | rollout/ep_len_mean = 60.6 | time/fps = 73 | time/time_elapsed = 122 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:05: [A2C[worker: 3]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 1.531839370727539e-05 | train/entropy_loss = -0.578125 | train/policy_loss = 0.3322591781616211 | train/value_loss = 0.43294787406921387 | time/iterations = 1800 | rollout/ep_rew_mean = 77.06 | rollout/ep_len_mean = 77.06 | time/fps = 73 | time/time_elapsed = 122 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 1.531839370727539e-05 | train/entropy_loss = -0.578125 | train/policy_loss = 0.3322591781616211 | train/value_loss = 0.43294787406921387 | time/iterations = 1800 | rollout/ep_rew_mean = 77.06 | rollout/ep_len_mean = 77.06 | time/fps = 73 | time/time_elapsed = 122 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:05: [A2C[worker: 4]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -7.033348083496094e-06 | train/entropy_loss = -0.5545161366462708 | train/policy_loss = -9.515450477600098 | train/value_loss = 2763.290283203125 | time/iterations = 1900 | rollout/ep_rew_mean = 80.3 | rollout/ep_len_mean = 80.3 | time/fps = 75 | time/time_elapsed = 125 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -7.033348083496094e-06 | train/entropy_loss = -0.5545161366462708 | train/policy_loss = -9.515450477600098 | train/value_loss = 2763.290283203125 | time/iterations = 1900 | rollout/ep_rew_mean = 80.3 | rollout/ep_len_mean = 80.3 | time/fps = 75 | time/time_elapsed = 125 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:05: [A2C[worker: 0]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 3.0279159545898438e-05 | train/entropy_loss = -0.5306060910224915 | train/policy_loss = 0.11524113267660141 | train/value_loss = 0.09487800300121307 | time/iterations = 1900 | rollout/ep_rew_mean = 98.16842105263157 | rollout/ep_len_mean = 98.16842105263157 | time/fps = 74 | time/time_elapsed = 126 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 3.0279159545898438e-05 | train/entropy_loss = -0.5306060910224915 | train/policy_loss = 0.11524113267660141 | train/value_loss = 0.09487800300121307 | time/iterations = 1900 | rollout/ep_rew_mean = 98.16842105263157 | rollout/ep_len_mean = 98.16842105263157 | time/fps = 74 | time/time_elapsed = 126 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:05: [A2C[worker: 1]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.00011450052261352539 | train/entropy_loss = -0.516208291053772 | train/policy_loss = 0.14234991371631622 | train/value_loss = 0.09731733053922653 | time/iterations = 1900 | rollout/ep_rew_mean = 97.3917525773196 | rollout/ep_len_mean = 97.3917525773196 | time/fps = 74 | time/time_elapsed = 127 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.00011450052261352539 | train/entropy_loss = -0.516208291053772 | train/policy_loss = 0.14234991371631622 | train/value_loss = 0.09731733053922653 | time/iterations = 1900 | rollout/ep_rew_mean = 97.3917525773196 | rollout/ep_len_mean = 97.3917525773196 | time/fps = 74 | time/time_elapsed = 127 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:05: [A2C[worker: 2]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -2.944469451904297e-05 | train/entropy_loss = -0.44743403792381287 | train/policy_loss = 0.3532154858112335 | train/value_loss = 0.48589086532592773 | time/iterations = 1900 | rollout/ep_rew_mean = 62.24 | rollout/ep_len_mean = 62.24 | time/fps = 73 | time/time_elapsed = 128 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -2.944469451904297e-05 | train/entropy_loss = -0.44743403792381287 | train/policy_loss = 0.3532154858112335 | train/value_loss = 0.48589086532592773 | time/iterations = 1900 | rollout/ep_rew_mean = 62.24 | rollout/ep_len_mean = 62.24 | time/fps = 73 | time/time_elapsed = 128 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:05: [A2C[worker: 3]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 5.894899368286133e-05 | train/entropy_loss = -0.589726448059082 | train/policy_loss = 0.32548683881759644 | train/value_loss = 0.26211830973625183 | time/iterations = 1900 | rollout/ep_rew_mean = 78.73 | rollout/ep_len_mean = 78.73 | time/fps = 73 | time/time_elapsed = 129 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 5.894899368286133e-05 | train/entropy_loss = -0.589726448059082 | train/policy_loss = 0.32548683881759644 | train/value_loss = 0.26211830973625183 | time/iterations = 1900 | rollout/ep_rew_mean = 78.73 | rollout/ep_len_mean = 78.73 | time/fps = 73 | time/time_elapsed = 129 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:05: ... trained! [0m
    INFO:rlberry_logger:... trained!
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:05: Saved ExperimentManager(A2C) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(A2C) using pickle.
    [38;21m[INFO] 15:05: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-03-17_19f36ffd/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-03-17_19f36ffd/manager_obj.pickle'
    [38;21m[INFO] 15:05: Running ExperimentManager fit() for PPO with n_fit = 5 and max_workers = None. [0m
    INFO:rlberry_logger:Running ExperimentManager fit() for PPO with n_fit = 5 and max_workers = None.
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:05:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        3           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        3           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:05:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        0           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        0           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:05:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        4           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        4           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:05:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        2           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        2           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:05:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        1           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        1           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:06: [PPO[worker: 4]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.045454545454547 | rollout/ep_len_mean = 23.045454545454547 | time/fps = 152 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.045454545454547 | rollout/ep_len_mean = 23.045454545454547 | time/fps = 152 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:06: [PPO[worker: 3]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.63953488372093 | rollout/ep_len_mean = 23.63953488372093 | time/fps = 153 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 23.63953488372093 | rollout/ep_len_mean = 23.63953488372093 | time/fps = 153 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:06: [PPO[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.428571428571427 | rollout/ep_len_mean = 22.428571428571427 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.428571428571427 | rollout/ep_len_mean = 22.428571428571427 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:06: [PPO[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.913978494623656 | rollout/ep_len_mean = 21.913978494623656 | time/fps = 152 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.913978494623656 | rollout/ep_len_mean = 21.913978494623656 | time/fps = 152 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:06: [PPO[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 24.19047619047619 | rollout/ep_len_mean = 24.19047619047619 | time/fps = 149 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 24.19047619047619 | rollout/ep_len_mean = 24.19047619047619 | time/fps = 149 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    [38;21m[INFO] 15:06: [PPO[worker: 4]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 28.33 | rollout/ep_len_mean = 28.33 | time/fps = 112 | time/time_elapsed = 36 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6855821546167136 | train/policy_gradient_loss = -0.019578370334056672 | train/value_loss = 57.91945840120316 | train/approx_kl = 0.009054658934473991 | train/clip_fraction = 0.117626953125 | train/loss = 8.426912307739258 | train/explained_variance = 0.009046494960784912 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 28.33 | rollout/ep_len_mean = 28.33 | time/fps = 112 | time/time_elapsed = 36 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6855821546167136 | train/policy_gradient_loss = -0.019578370334056672 | train/value_loss = 57.91945840120316 | train/approx_kl = 0.009054658934473991 | train/clip_fraction = 0.117626953125 | train/loss = 8.426912307739258 | train/explained_variance = 0.009046494960784912 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:06: [PPO[worker: 3]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.28 | rollout/ep_len_mean = 26.28 | time/fps = 111 | time/time_elapsed = 36 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6865503489971161 | train/policy_gradient_loss = -0.015592032196582294 | train/value_loss = 57.32514564394951 | train/approx_kl = 0.007778831757605076 | train/clip_fraction = 0.0990234375 | train/loss = 7.951833248138428 | train/explained_variance = -0.00044083595275878906 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.28 | rollout/ep_len_mean = 26.28 | time/fps = 111 | time/time_elapsed = 36 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6865503489971161 | train/policy_gradient_loss = -0.015592032196582294 | train/value_loss = 57.32514564394951 | train/approx_kl = 0.007778831757605076 | train/clip_fraction = 0.0990234375 | train/loss = 7.951833248138428 | train/explained_variance = -0.00044083595275878906 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:06: [PPO[worker: 2]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.07 | rollout/ep_len_mean = 27.07 | time/fps = 110 | time/time_elapsed = 37 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6863669069483876 | train/policy_gradient_loss = -0.014665822236565873 | train/value_loss = 53.02105034291744 | train/approx_kl = 0.008388923481106758 | train/clip_fraction = 0.094189453125 | train/loss = 8.274557113647461 | train/explained_variance = -0.0012385845184326172 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.07 | rollout/ep_len_mean = 27.07 | time/fps = 110 | time/time_elapsed = 37 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6863669069483876 | train/policy_gradient_loss = -0.014665822236565873 | train/value_loss = 53.02105034291744 | train/approx_kl = 0.008388923481106758 | train/clip_fraction = 0.094189453125 | train/loss = 8.274557113647461 | train/explained_variance = -0.0012385845184326172 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:06: [PPO[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.55 | rollout/ep_len_mean = 25.55 | time/fps = 110 | time/time_elapsed = 37 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6856023507192731 | train/policy_gradient_loss = -0.01833865966764279 | train/value_loss = 48.832897648215294 | train/approx_kl = 0.009716484695672989 | train/clip_fraction = 0.122021484375 | train/loss = 6.524239540100098 | train/explained_variance = -0.004844188690185547 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.55 | rollout/ep_len_mean = 25.55 | time/fps = 110 | time/time_elapsed = 37 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6856023507192731 | train/policy_gradient_loss = -0.01833865966764279 | train/value_loss = 48.832897648215294 | train/approx_kl = 0.009716484695672989 | train/clip_fraction = 0.122021484375 | train/loss = 6.524239540100098 | train/explained_variance = -0.004844188690185547 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:06: [PPO[worker: 1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.93 | rollout/ep_len_mean = 26.93 | time/fps = 109 | time/time_elapsed = 37 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6870815070345998 | train/policy_gradient_loss = -0.010852844186592847 | train/value_loss = 57.2336449354887 | train/approx_kl = 0.008126066997647285 | train/clip_fraction = 0.0751953125 | train/loss = 9.966599464416504 | train/explained_variance = -0.006066322326660156 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.93 | rollout/ep_len_mean = 26.93 | time/fps = 109 | time/time_elapsed = 37 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6870815070345998 | train/policy_gradient_loss = -0.010852844186592847 | train/value_loss = 57.2336449354887 | train/approx_kl = 0.008126066997647285 | train/clip_fraction = 0.0751953125 | train/loss = 9.966599464416504 | train/explained_variance = -0.006066322326660156 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: [PPO[worker: 4]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 37.43 | rollout/ep_len_mean = 37.43 | time/fps = 101 | time/time_elapsed = 60 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.666085897758603 | train/policy_gradient_loss = -0.018712871163734233 | train/value_loss = 34.757980370521544 | train/approx_kl = 0.01057757344096899 | train/clip_fraction = 0.06826171875 | train/loss = 10.268289566040039 | train/explained_variance = 0.05594611167907715 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 37.43 | rollout/ep_len_mean = 37.43 | time/fps = 101 | time/time_elapsed = 60 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.666085897758603 | train/policy_gradient_loss = -0.018712871163734233 | train/value_loss = 34.757980370521544 | train/approx_kl = 0.01057757344096899 | train/clip_fraction = 0.06826171875 | train/loss = 10.268289566040039 | train/explained_variance = 0.05594611167907715 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: [PPO[worker: 0]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 32.83 | rollout/ep_len_mean = 32.83 | time/fps = 99 | time/time_elapsed = 61 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6693092837929726 | train/policy_gradient_loss = -0.019600948303559563 | train/value_loss = 35.54562155604363 | train/approx_kl = 0.00984528660774231 | train/clip_fraction = 0.072607421875 | train/loss = 12.980842590332031 | train/explained_variance = 0.14364826679229736 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 32.83 | rollout/ep_len_mean = 32.83 | time/fps = 99 | time/time_elapsed = 61 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6693092837929726 | train/policy_gradient_loss = -0.019600948303559563 | train/value_loss = 35.54562155604363 | train/approx_kl = 0.00984528660774231 | train/clip_fraction = 0.072607421875 | train/loss = 12.980842590332031 | train/explained_variance = 0.14364826679229736 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: [PPO[worker: 3]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 34.44 | rollout/ep_len_mean = 34.44 | time/fps = 100 | time/time_elapsed = 61 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6678574515506626 | train/policy_gradient_loss = -0.01743735173367895 | train/value_loss = 34.57134313583374 | train/approx_kl = 0.009694496169686317 | train/clip_fraction = 0.05771484375 | train/loss = 16.37115478515625 | train/explained_variance = 0.05414217710494995 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 34.44 | rollout/ep_len_mean = 34.44 | time/fps = 100 | time/time_elapsed = 61 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6678574515506626 | train/policy_gradient_loss = -0.01743735173367895 | train/value_loss = 34.57134313583374 | train/approx_kl = 0.009694496169686317 | train/clip_fraction = 0.05771484375 | train/loss = 16.37115478515625 | train/explained_variance = 0.05414217710494995 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: [PPO[worker: 2]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 35.37 | rollout/ep_len_mean = 35.37 | time/fps = 99 | time/time_elapsed = 61 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6652133556082844 | train/policy_gradient_loss = -0.014915495341119822 | train/value_loss = 37.66022880673408 | train/approx_kl = 0.00888967514038086 | train/clip_fraction = 0.05234375 | train/loss = 10.17378044128418 | train/explained_variance = 0.08586394786834717 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 35.37 | rollout/ep_len_mean = 35.37 | time/fps = 99 | time/time_elapsed = 61 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6652133556082844 | train/policy_gradient_loss = -0.014915495341119822 | train/value_loss = 37.66022880673408 | train/approx_kl = 0.00888967514038086 | train/clip_fraction = 0.05234375 | train/loss = 10.17378044128418 | train/explained_variance = 0.08586394786834717 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: [PPO[worker: 1]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 36.25 | rollout/ep_len_mean = 36.25 | time/fps = 99 | time/time_elapsed = 62 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6642959631979466 | train/policy_gradient_loss = -0.020860919190454297 | train/value_loss = 35.729093956947324 | train/approx_kl = 0.011025937274098396 | train/clip_fraction = 0.080517578125 | train/loss = 13.510440826416016 | train/explained_variance = 0.10118263959884644 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 36.25 | rollout/ep_len_mean = 36.25 | time/fps = 99 | time/time_elapsed = 62 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6642959631979466 | train/policy_gradient_loss = -0.020860919190454297 | train/value_loss = 35.729093956947324 | train/approx_kl = 0.011025937274098396 | train/clip_fraction = 0.080517578125 | train/loss = 13.510440826416016 | train/explained_variance = 0.10118263959884644 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: [PPO[worker: 4]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 48.85 | rollout/ep_len_mean = 48.85 | time/fps = 97 | time/time_elapsed = 84 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.635801306180656 | train/policy_gradient_loss = -0.016553470183862374 | train/value_loss = 55.51804388761521 | train/approx_kl = 0.008969226852059364 | train/clip_fraction = 0.08681640625 | train/loss = 24.498998641967773 | train/explained_variance = 0.21174871921539307 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 48.85 | rollout/ep_len_mean = 48.85 | time/fps = 97 | time/time_elapsed = 84 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.635801306180656 | train/policy_gradient_loss = -0.016553470183862374 | train/value_loss = 55.51804388761521 | train/approx_kl = 0.008969226852059364 | train/clip_fraction = 0.08681640625 | train/loss = 24.498998641967773 | train/explained_variance = 0.21174871921539307 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: [PPO[worker: 3]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.67 | rollout/ep_len_mean = 47.67 | time/fps = 96 | time/time_elapsed = 84 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6390980303287506 | train/policy_gradient_loss = -0.01797724827192724 | train/value_loss = 58.0338671207428 | train/approx_kl = 0.009383758530020714 | train/clip_fraction = 0.086669921875 | train/loss = 26.77962303161621 | train/explained_variance = 0.21048128604888916 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.67 | rollout/ep_len_mean = 47.67 | time/fps = 96 | time/time_elapsed = 84 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6390980303287506 | train/policy_gradient_loss = -0.01797724827192724 | train/value_loss = 58.0338671207428 | train/approx_kl = 0.009383758530020714 | train/clip_fraction = 0.086669921875 | train/loss = 26.77962303161621 | train/explained_variance = 0.21048128604888916 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: [PPO[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 42.45 | rollout/ep_len_mean = 42.45 | time/fps = 97 | time/time_elapsed = 84 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6278021160513163 | train/policy_gradient_loss = -0.026102848345180972 | train/value_loss = 47.80316988825798 | train/approx_kl = 0.012307563796639442 | train/clip_fraction = 0.14658203125 | train/loss = 25.745498657226562 | train/explained_variance = 0.2764056324958801 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 42.45 | rollout/ep_len_mean = 42.45 | time/fps = 97 | time/time_elapsed = 84 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6278021160513163 | train/policy_gradient_loss = -0.026102848345180972 | train/value_loss = 47.80316988825798 | train/approx_kl = 0.012307563796639442 | train/clip_fraction = 0.14658203125 | train/loss = 25.745498657226562 | train/explained_variance = 0.2764056324958801 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: [PPO[worker: 2]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 44.41 | rollout/ep_len_mean = 44.41 | time/fps = 96 | time/time_elapsed = 84 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6274249570444226 | train/policy_gradient_loss = -0.01951213801803533 | train/value_loss = 49.91790453195572 | train/approx_kl = 0.007597693242132664 | train/clip_fraction = 0.0869140625 | train/loss = 17.09599494934082 | train/explained_variance = 0.2093888521194458 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 44.41 | rollout/ep_len_mean = 44.41 | time/fps = 96 | time/time_elapsed = 84 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6274249570444226 | train/policy_gradient_loss = -0.01951213801803533 | train/value_loss = 49.91790453195572 | train/approx_kl = 0.007597693242132664 | train/clip_fraction = 0.0869140625 | train/loss = 17.09599494934082 | train/explained_variance = 0.2093888521194458 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: [PPO[worker: 1]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.09 | rollout/ep_len_mean = 47.09 | time/fps = 95 | time/time_elapsed = 85 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6337788594886661 | train/policy_gradient_loss = -0.018271777502377518 | train/value_loss = 59.01572444438934 | train/approx_kl = 0.007316542323678732 | train/clip_fraction = 0.073291015625 | train/loss = 22.2635555267334 | train/explained_variance = 0.21362674236297607 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.09 | rollout/ep_len_mean = 47.09 | time/fps = 95 | time/time_elapsed = 85 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6337788594886661 | train/policy_gradient_loss = -0.018271777502377518 | train/value_loss = 59.01572444438934 | train/approx_kl = 0.007316542323678732 | train/clip_fraction = 0.073291015625 | train/loss = 22.2635555267334 | train/explained_variance = 0.21362674236297607 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:07: ... trained! [0m
    INFO:rlberry_logger:... trained!
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:07: Saved ExperimentManager(PPO) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(PPO) using pickle.
    [38;21m[INFO] 15:07: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-03-17_68454fc8/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-03-17_68454fc8/manager_obj.pickle'
    [38;21m[INFO] 15:07: Saved ExperimentManager(A2C) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(A2C) using pickle.
    [38;21m[INFO] 15:07: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-03-17_19f36ffd/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-03-17_19f36ffd/manager_obj.pickle'
    [38;21m[INFO] 15:07: Saved ExperimentManager(PPO) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(PPO) using pickle.
    [38;21m[INFO] 15:07: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-03-17_68454fc8/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-03-17_68454fc8/manager_obj.pickle'
    [38;21m[INFO] 15:07: Evaluating agent 0 [0m
    INFO:rlberry_logger:Evaluating agent 0
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:07: Evaluating agent 1 [0m
    INFO:rlberry_logger:Evaluating agent 1
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:08: Evaluating agent 2 [0m
    INFO:rlberry_logger:Evaluating agent 2
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:08: Evaluating agent 3 [0m
    INFO:rlberry_logger:Evaluating agent 3
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:08: Evaluating agent 4 [0m
    INFO:rlberry_logger:Evaluating agent 4
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:08: Evaluating agent 0 [0m
    INFO:rlberry_logger:Evaluating agent 0
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:08: Evaluating agent 1 [0m
    INFO:rlberry_logger:Evaluating agent 1
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:09: Evaluating agent 2 [0m
    INFO:rlberry_logger:Evaluating agent 2
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:09: Evaluating agent 3 [0m
    INFO:rlberry_logger:Evaluating agent 3
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:09: Evaluating agent 4 [0m
    INFO:rlberry_logger:Evaluating agent 4
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:09: Running ExperimentManager fit() for A2C with n_fit = 5 and max_workers = None. [0m
    INFO:rlberry_logger:Running ExperimentManager fit() for A2C with n_fit = 5 and max_workers = None.


    Step 1


    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:09:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        2           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        2           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:09:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        4           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        4           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:09:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        3           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        3           0.001               500
    [38;21m[INFO] 15:09:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        0           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        0           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:09:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        1           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        1           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:09: [A2C[worker: 4]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.18642711639404297 | train/entropy_loss = -0.6907178163528442 | train/policy_loss = 0.4677669405937195 | train/value_loss = 25.563091278076172 | time/iterations = 100 | rollout/ep_rew_mean = 28.88235294117647 | rollout/ep_len_mean = 28.88235294117647 | time/fps = 74 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.18642711639404297 | train/entropy_loss = -0.6907178163528442 | train/policy_loss = 0.4677669405937195 | train/value_loss = 25.563091278076172 | time/iterations = 100 | rollout/ep_rew_mean = 28.88235294117647 | rollout/ep_len_mean = 28.88235294117647 | time/fps = 74 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:09: [A2C[worker: 0]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.45247167348861694 | train/entropy_loss = -0.673658013343811 | train/policy_loss = 1.8011510372161865 | train/value_loss = 8.262370109558105 | time/iterations = 100 | rollout/ep_rew_mean = 44.9 | rollout/ep_len_mean = 44.9 | time/fps = 73 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.45247167348861694 | train/entropy_loss = -0.673658013343811 | train/policy_loss = 1.8011510372161865 | train/value_loss = 8.262370109558105 | time/iterations = 100 | rollout/ep_rew_mean = 44.9 | rollout/ep_len_mean = 44.9 | time/fps = 73 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:09: [A2C[worker: 2]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -1.3078675270080566 | train/entropy_loss = -0.6542801260948181 | train/policy_loss = 1.7599084377288818 | train/value_loss = 9.81079387664795 | time/iterations = 100 | rollout/ep_rew_mean = 29.0625 | rollout/ep_len_mean = 29.0625 | time/fps = 74 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -1.3078675270080566 | train/entropy_loss = -0.6542801260948181 | train/policy_loss = 1.7599084377288818 | train/value_loss = 9.81079387664795 | time/iterations = 100 | rollout/ep_rew_mean = 29.0625 | rollout/ep_len_mean = 29.0625 | time/fps = 74 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    [38;21m[INFO] 15:09: [A2C[worker: 3]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.8413596302270889 | train/entropy_loss = -0.6585522890090942 | train/policy_loss = -0.1458371877670288 | train/value_loss = 1.0055694580078125 | time/iterations = 100 | rollout/ep_rew_mean = 13.694444444444445 | rollout/ep_len_mean = 13.694444444444445 | time/fps = 73 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.8413596302270889 | train/entropy_loss = -0.6585522890090942 | train/policy_loss = -0.1458371877670288 | train/value_loss = 1.0055694580078125 | time/iterations = 100 | rollout/ep_rew_mean = 13.694444444444445 | rollout/ep_len_mean = 13.694444444444445 | time/fps = 73 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:09: [A2C[worker: 1]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.01315528154373169 | train/entropy_loss = -0.6931458711624146 | train/policy_loss = 1.93643319606781 | train/value_loss = 9.361917495727539 | time/iterations = 100 | rollout/ep_rew_mean = 18.76923076923077 | rollout/ep_len_mean = 18.76923076923077 | time/fps = 72 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.01315528154373169 | train/entropy_loss = -0.6931458711624146 | train/policy_loss = 1.93643319606781 | train/value_loss = 9.361917495727539 | time/iterations = 100 | rollout/ep_rew_mean = 18.76923076923077 | rollout/ep_len_mean = 18.76923076923077 | time/fps = 72 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    [38;21m[INFO] 15:09: [A2C[worker: 4]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.1804349422454834 | train/entropy_loss = -0.6912150979042053 | train/policy_loss = 1.8180732727050781 | train/value_loss = 9.33709716796875 | time/iterations = 200 | rollout/ep_rew_mean = 27.333333333333332 | rollout/ep_len_mean = 27.333333333333332 | time/fps = 71 | time/time_elapsed = 13 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.1804349422454834 | train/entropy_loss = -0.6912150979042053 | train/policy_loss = 1.8180732727050781 | train/value_loss = 9.33709716796875 | time/iterations = 200 | rollout/ep_rew_mean = 27.333333333333332 | rollout/ep_len_mean = 27.333333333333332 | time/fps = 71 | time/time_elapsed = 13 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:09: [A2C[worker: 0]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.10016942024230957 | train/entropy_loss = -0.6692067384719849 | train/policy_loss = 1.291132926940918 | train/value_loss = 5.620176315307617 | time/iterations = 200 | rollout/ep_rew_mean = 37.57692307692308 | rollout/ep_len_mean = 37.57692307692308 | time/fps = 70 | time/time_elapsed = 14 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.10016942024230957 | train/entropy_loss = -0.6692067384719849 | train/policy_loss = 1.291132926940918 | train/value_loss = 5.620176315307617 | time/iterations = 200 | rollout/ep_rew_mean = 37.57692307692308 | rollout/ep_len_mean = 37.57692307692308 | time/fps = 70 | time/time_elapsed = 14 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:09: [A2C[worker: 3]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.15797924995422363 | train/entropy_loss = -0.6892672777175903 | train/policy_loss = 1.6069138050079346 | train/value_loss = 8.544564247131348 | time/iterations = 200 | rollout/ep_rew_mean = 17.5 | rollout/ep_len_mean = 17.5 | time/fps = 69 | time/time_elapsed = 14 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.15797924995422363 | train/entropy_loss = -0.6892672777175903 | train/policy_loss = 1.6069138050079346 | train/value_loss = 8.544564247131348 | time/iterations = 200 | rollout/ep_rew_mean = 17.5 | rollout/ep_len_mean = 17.5 | time/fps = 69 | time/time_elapsed = 14 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:09: [A2C[worker: 2]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.1084526777267456 | train/entropy_loss = -0.5661053657531738 | train/policy_loss = 1.2847734689712524 | train/value_loss = 5.54800271987915 | time/iterations = 200 | rollout/ep_rew_mean = 35.642857142857146 | rollout/ep_len_mean = 35.642857142857146 | time/fps = 69 | time/time_elapsed = 14 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.1084526777267456 | train/entropy_loss = -0.5661053657531738 | train/policy_loss = 1.2847734689712524 | train/value_loss = 5.54800271987915 | time/iterations = 200 | rollout/ep_rew_mean = 35.642857142857146 | rollout/ep_len_mean = 35.642857142857146 | time/fps = 69 | time/time_elapsed = 14 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:09: [A2C[worker: 1]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.5823498964309692 | train/entropy_loss = -0.6928364634513855 | train/policy_loss = 1.9771530628204346 | train/value_loss = 10.936906814575195 | time/iterations = 200 | rollout/ep_rew_mean = 18.60377358490566 | rollout/ep_len_mean = 18.60377358490566 | time/fps = 68 | time/time_elapsed = 14 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.5823498964309692 | train/entropy_loss = -0.6928364634513855 | train/policy_loss = 1.9771530628204346 | train/value_loss = 10.936906814575195 | time/iterations = 200 | rollout/ep_rew_mean = 18.60377358490566 | rollout/ep_len_mean = 18.60377358490566 | time/fps = 68 | time/time_elapsed = 14 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:10: [A2C[worker: 3]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.10999506711959839 | train/entropy_loss = -0.6892181634902954 | train/policy_loss = 1.4866740703582764 | train/value_loss = 6.2296600341796875 | time/iterations = 300 | rollout/ep_rew_mean = 18.469135802469136 | rollout/ep_len_mean = 18.469135802469136 | time/fps = 68 | time/time_elapsed = 22 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.10999506711959839 | train/entropy_loss = -0.6892181634902954 | train/policy_loss = 1.4866740703582764 | train/value_loss = 6.2296600341796875 | time/iterations = 300 | rollout/ep_rew_mean = 18.469135802469136 | rollout/ep_len_mean = 18.469135802469136 | time/fps = 68 | time/time_elapsed = 22 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:10: [A2C[worker: 0]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.07676029205322266 | train/entropy_loss = -0.6414602994918823 | train/policy_loss = 1.25242018699646 | train/value_loss = 5.673001766204834 | time/iterations = 300 | rollout/ep_rew_mean = 32.06521739130435 | rollout/ep_len_mean = 32.06521739130435 | time/fps = 68 | time/time_elapsed = 21 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.07676029205322266 | train/entropy_loss = -0.6414602994918823 | train/policy_loss = 1.25242018699646 | train/value_loss = 5.673001766204834 | time/iterations = 300 | rollout/ep_rew_mean = 32.06521739130435 | rollout/ep_len_mean = 32.06521739130435 | time/fps = 68 | time/time_elapsed = 21 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:10: [A2C[worker: 4]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.004576981067657471 | train/entropy_loss = -0.6915091276168823 | train/policy_loss = 1.6600682735443115 | train/value_loss = 6.924845218658447 | time/iterations = 300 | rollout/ep_rew_mean = 26.285714285714285 | rollout/ep_len_mean = 26.285714285714285 | time/fps = 68 | time/time_elapsed = 21 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.004576981067657471 | train/entropy_loss = -0.6915091276168823 | train/policy_loss = 1.6600682735443115 | train/value_loss = 6.924845218658447 | time/iterations = 300 | rollout/ep_rew_mean = 26.285714285714285 | rollout/ep_len_mean = 26.285714285714285 | time/fps = 68 | time/time_elapsed = 21 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:10: [A2C[worker: 2]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.026686787605285645 | train/entropy_loss = -0.6308630108833313 | train/policy_loss = 1.0583724975585938 | train/value_loss = 6.197861671447754 | time/iterations = 300 | rollout/ep_rew_mean = 35.357142857142854 | rollout/ep_len_mean = 35.357142857142854 | time/fps = 67 | time/time_elapsed = 22 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.026686787605285645 | train/entropy_loss = -0.6308630108833313 | train/policy_loss = 1.0583724975585938 | train/value_loss = 6.197861671447754 | time/iterations = 300 | rollout/ep_rew_mean = 35.357142857142854 | rollout/ep_len_mean = 35.357142857142854 | time/fps = 67 | time/time_elapsed = 22 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:10: [A2C[worker: 1]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.08796954154968262 | train/entropy_loss = -0.6501177549362183 | train/policy_loss = 1.219123125076294 | train/value_loss = 7.665058135986328 | time/iterations = 300 | rollout/ep_rew_mean = 21.08823529411765 | rollout/ep_len_mean = 21.08823529411765 | time/fps = 66 | time/time_elapsed = 22 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.08796954154968262 | train/entropy_loss = -0.6501177549362183 | train/policy_loss = 1.219123125076294 | train/value_loss = 7.665058135986328 | time/iterations = 300 | rollout/ep_rew_mean = 21.08823529411765 | rollout/ep_len_mean = 21.08823529411765 | time/fps = 66 | time/time_elapsed = 22 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:10: [A2C[worker: 3]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.008651137351989746 | train/entropy_loss = -0.6721038818359375 | train/policy_loss = 1.5308561325073242 | train/value_loss = 6.4369659423828125 | time/iterations = 400 | rollout/ep_rew_mean = 21.311827956989248 | rollout/ep_len_mean = 21.311827956989248 | time/fps = 68 | time/time_elapsed = 29 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.008651137351989746 | train/entropy_loss = -0.6721038818359375 | train/policy_loss = 1.5308561325073242 | train/value_loss = 6.4369659423828125 | time/iterations = 400 | rollout/ep_rew_mean = 21.311827956989248 | rollout/ep_len_mean = 21.311827956989248 | time/fps = 68 | time/time_elapsed = 29 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:10: [A2C[worker: 4]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.009793758392333984 | train/entropy_loss = -0.6899396181106567 | train/policy_loss = 1.5478688478469849 | train/value_loss = 6.152771949768066 | time/iterations = 400 | rollout/ep_rew_mean = 27.356164383561644 | rollout/ep_len_mean = 27.356164383561644 | time/fps = 68 | time/time_elapsed = 29 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.009793758392333984 | train/entropy_loss = -0.6899396181106567 | train/policy_loss = 1.5478688478469849 | train/value_loss = 6.152771949768066 | time/iterations = 400 | rollout/ep_rew_mean = 27.356164383561644 | rollout/ep_len_mean = 27.356164383561644 | time/fps = 68 | time/time_elapsed = 29 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:10: [A2C[worker: 0]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.3753969669342041 | train/entropy_loss = -0.5877801775932312 | train/policy_loss = 0.7385711669921875 | train/value_loss = 6.3294525146484375 | time/iterations = 400 | rollout/ep_rew_mean = 30.106060606060606 | rollout/ep_len_mean = 30.106060606060606 | time/fps = 68 | time/time_elapsed = 29 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.3753969669342041 | train/entropy_loss = -0.5877801775932312 | train/policy_loss = 0.7385711669921875 | train/value_loss = 6.3294525146484375 | time/iterations = 400 | rollout/ep_rew_mean = 30.106060606060606 | rollout/ep_len_mean = 30.106060606060606 | time/fps = 68 | time/time_elapsed = 29 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:10: [A2C[worker: 2]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.062445759773254395 | train/entropy_loss = -0.5823730826377869 | train/policy_loss = 1.0034661293029785 | train/value_loss = 5.74527645111084 | time/iterations = 400 | rollout/ep_rew_mean = 36.574074074074076 | rollout/ep_len_mean = 36.574074074074076 | time/fps = 68 | time/time_elapsed = 29 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.062445759773254395 | train/entropy_loss = -0.5823730826377869 | train/policy_loss = 1.0034661293029785 | train/value_loss = 5.74527645111084 | time/iterations = 400 | rollout/ep_rew_mean = 36.574074074074076 | rollout/ep_len_mean = 36.574074074074076 | time/fps = 68 | time/time_elapsed = 29 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:10: [A2C[worker: 1]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.025402307510375977 | train/entropy_loss = -0.43576890230178833 | train/policy_loss = 2.0088400840759277 | train/value_loss = 6.401949882507324 | time/iterations = 400 | rollout/ep_rew_mean = 25.666666666666668 | rollout/ep_len_mean = 25.666666666666668 | time/fps = 67 | time/time_elapsed = 29 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.025402307510375977 | train/entropy_loss = -0.43576890230178833 | train/policy_loss = 2.0088400840759277 | train/value_loss = 6.401949882507324 | time/iterations = 400 | rollout/ep_rew_mean = 25.666666666666668 | rollout/ep_len_mean = 25.666666666666668 | time/fps = 67 | time/time_elapsed = 29 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:10: [A2C[worker: 3]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.004522919654846191 | train/entropy_loss = -0.6823202967643738 | train/policy_loss = 1.2793045043945312 | train/value_loss = 5.683351516723633 | time/iterations = 500 | rollout/ep_rew_mean = 23.72 | rollout/ep_len_mean = 23.72 | time/fps = 69 | time/time_elapsed = 36 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.004522919654846191 | train/entropy_loss = -0.6823202967643738 | train/policy_loss = 1.2793045043945312 | train/value_loss = 5.683351516723633 | time/iterations = 500 | rollout/ep_rew_mean = 23.72 | rollout/ep_len_mean = 23.72 | time/fps = 69 | time/time_elapsed = 36 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:10: [A2C[worker: 4]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.005380988121032715 | train/entropy_loss = -0.685744047164917 | train/policy_loss = 1.3542277812957764 | train/value_loss = 5.510058403015137 | time/iterations = 500 | rollout/ep_rew_mean = 28.569767441860463 | rollout/ep_len_mean = 28.569767441860463 | time/fps = 69 | time/time_elapsed = 36 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.005380988121032715 | train/entropy_loss = -0.685744047164917 | train/policy_loss = 1.3542277812957764 | train/value_loss = 5.510058403015137 | time/iterations = 500 | rollout/ep_rew_mean = 28.569767441860463 | rollout/ep_len_mean = 28.569767441860463 | time/fps = 69 | time/time_elapsed = 36 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:10: [A2C[worker: 2]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.012939214706420898 | train/entropy_loss = -0.5823730230331421 | train/policy_loss = 1.0283818244934082 | train/value_loss = 5.6837639808654785 | time/iterations = 500 | rollout/ep_rew_mean = 38.375 | rollout/ep_len_mean = 38.375 | time/fps = 68 | time/time_elapsed = 36 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.012939214706420898 | train/entropy_loss = -0.5823730230331421 | train/policy_loss = 1.0283818244934082 | train/value_loss = 5.6837639808654785 | time/iterations = 500 | rollout/ep_rew_mean = 38.375 | rollout/ep_len_mean = 38.375 | time/fps = 68 | time/time_elapsed = 36 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:10: [A2C[worker: 0]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.04369664192199707 | train/entropy_loss = -0.5499756336212158 | train/policy_loss = 1.370962381362915 | train/value_loss = 5.990866184234619 | time/iterations = 500 | rollout/ep_rew_mean = 29.547619047619047 | rollout/ep_len_mean = 29.547619047619047 | time/fps = 68 | time/time_elapsed = 36 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.04369664192199707 | train/entropy_loss = -0.5499756336212158 | train/policy_loss = 1.370962381362915 | train/value_loss = 5.990866184234619 | time/iterations = 500 | rollout/ep_rew_mean = 29.547619047619047 | rollout/ep_len_mean = 29.547619047619047 | time/fps = 68 | time/time_elapsed = 36 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:10: [A2C[worker: 1]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.003229975700378418 | train/entropy_loss = -0.666698157787323 | train/policy_loss = 1.243027687072754 | train/value_loss = 5.507551670074463 | time/iterations = 500 | rollout/ep_rew_mean = 30.0 | rollout/ep_len_mean = 30.0 | time/fps = 66 | time/time_elapsed = 37 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.003229975700378418 | train/entropy_loss = -0.666698157787323 | train/policy_loss = 1.243027687072754 | train/value_loss = 5.507551670074463 | time/iterations = 500 | rollout/ep_rew_mean = 30.0 | rollout/ep_len_mean = 30.0 | time/fps = 66 | time/time_elapsed = 37 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:10: [A2C[worker: 3]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.007139325141906738 | train/entropy_loss = -0.6681389808654785 | train/policy_loss = 1.3790162801742554 | train/value_loss = 5.190242290496826 | time/iterations = 600 | rollout/ep_rew_mean = 26.78 | rollout/ep_len_mean = 26.78 | time/fps = 69 | time/time_elapsed = 42 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.007139325141906738 | train/entropy_loss = -0.6681389808654785 | train/policy_loss = 1.3790162801742554 | train/value_loss = 5.190242290496826 | time/iterations = 600 | rollout/ep_rew_mean = 26.78 | rollout/ep_len_mean = 26.78 | time/fps = 69 | time/time_elapsed = 42 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:10: [A2C[worker: 4]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.03279531002044678 | train/entropy_loss = -0.5321053266525269 | train/policy_loss = 1.628596544265747 | train/value_loss = 4.92462158203125 | time/iterations = 600 | rollout/ep_rew_mean = 30.71875 | rollout/ep_len_mean = 30.71875 | time/fps = 69 | time/time_elapsed = 43 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.03279531002044678 | train/entropy_loss = -0.5321053266525269 | train/policy_loss = 1.628596544265747 | train/value_loss = 4.92462158203125 | time/iterations = 600 | rollout/ep_rew_mean = 30.71875 | rollout/ep_len_mean = 30.71875 | time/fps = 69 | time/time_elapsed = 43 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:10: [A2C[worker: 2]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0023932456970214844 | train/entropy_loss = -0.5413261651992798 | train/policy_loss = 0.7782852053642273 | train/value_loss = 4.924276828765869 | time/iterations = 600 | rollout/ep_rew_mean = 43.63235294117647 | rollout/ep_len_mean = 43.63235294117647 | time/fps = 69 | time/time_elapsed = 43 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0023932456970214844 | train/entropy_loss = -0.5413261651992798 | train/policy_loss = 0.7782852053642273 | train/value_loss = 4.924276828765869 | time/iterations = 600 | rollout/ep_rew_mean = 43.63235294117647 | rollout/ep_len_mean = 43.63235294117647 | time/fps = 69 | time/time_elapsed = 43 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:10: [A2C[worker: 0]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0076007843017578125 | train/entropy_loss = -0.6300274133682251 | train/policy_loss = 0.906743049621582 | train/value_loss = 5.166604042053223 | time/iterations = 600 | rollout/ep_rew_mean = 30.0 | rollout/ep_len_mean = 30.0 | time/fps = 68 | time/time_elapsed = 43 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0076007843017578125 | train/entropy_loss = -0.6300274133682251 | train/policy_loss = 0.906743049621582 | train/value_loss = 5.166604042053223 | time/iterations = 600 | rollout/ep_rew_mean = 30.0 | rollout/ep_len_mean = 30.0 | time/fps = 68 | time/time_elapsed = 43 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:10: [A2C[worker: 1]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.026357650756835938 | train/entropy_loss = -0.6566885113716125 | train/policy_loss = 1.0192025899887085 | train/value_loss = 4.904099464416504 | time/iterations = 600 | rollout/ep_rew_mean = 32.55555555555556 | rollout/ep_len_mean = 32.55555555555556 | time/fps = 67 | time/time_elapsed = 44 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.026357650756835938 | train/entropy_loss = -0.6566885113716125 | train/policy_loss = 1.0192025899887085 | train/value_loss = 4.904099464416504 | time/iterations = 600 | rollout/ep_rew_mean = 32.55555555555556 | rollout/ep_len_mean = 32.55555555555556 | time/fps = 67 | time/time_elapsed = 44 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:10: [A2C[worker: 3]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.029261231422424316 | train/entropy_loss = -0.5920466780662537 | train/policy_loss = 1.6800129413604736 | train/value_loss = 4.529644012451172 | time/iterations = 700 | rollout/ep_rew_mean = 29.73 | rollout/ep_len_mean = 29.73 | time/fps = 70 | time/time_elapsed = 49 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.029261231422424316 | train/entropy_loss = -0.5920466780662537 | train/policy_loss = 1.6800129413604736 | train/value_loss = 4.529644012451172 | time/iterations = 700 | rollout/ep_rew_mean = 29.73 | rollout/ep_len_mean = 29.73 | time/fps = 70 | time/time_elapsed = 49 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:10: [A2C[worker: 4]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.006057322025299072 | train/entropy_loss = -0.581157386302948 | train/policy_loss = 1.166274070739746 | train/value_loss = 4.416700839996338 | time/iterations = 700 | rollout/ep_rew_mean = 33.5 | rollout/ep_len_mean = 33.5 | time/fps = 70 | time/time_elapsed = 49 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.006057322025299072 | train/entropy_loss = -0.581157386302948 | train/policy_loss = 1.166274070739746 | train/value_loss = 4.416700839996338 | time/iterations = 700 | rollout/ep_rew_mean = 33.5 | rollout/ep_len_mean = 33.5 | time/fps = 70 | time/time_elapsed = 49 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:10: [A2C[worker: 2]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.0045580267906188965 | train/entropy_loss = -0.5794894099235535 | train/policy_loss = 1.1010726690292358 | train/value_loss = 4.24985408782959 | time/iterations = 700 | rollout/ep_rew_mean = 46.06849315068493 | rollout/ep_len_mean = 46.06849315068493 | time/fps = 69 | time/time_elapsed = 50 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.0045580267906188965 | train/entropy_loss = -0.5794894099235535 | train/policy_loss = 1.1010726690292358 | train/value_loss = 4.24985408782959 | time/iterations = 700 | rollout/ep_rew_mean = 46.06849315068493 | rollout/ep_len_mean = 46.06849315068493 | time/fps = 69 | time/time_elapsed = 50 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:10: [A2C[worker: 0]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.0034548044204711914 | train/entropy_loss = -0.6483399868011475 | train/policy_loss = 0.8774197697639465 | train/value_loss = 4.509138584136963 | time/iterations = 700 | rollout/ep_rew_mean = 30.27 | rollout/ep_len_mean = 30.27 | time/fps = 69 | time/time_elapsed = 50 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.0034548044204711914 | train/entropy_loss = -0.6483399868011475 | train/policy_loss = 0.8774197697639465 | train/value_loss = 4.509138584136963 | time/iterations = 700 | rollout/ep_rew_mean = 30.27 | rollout/ep_len_mean = 30.27 | time/fps = 69 | time/time_elapsed = 50 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:10: [A2C[worker: 1]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 7.909536361694336e-05 | train/entropy_loss = -0.6238623857498169 | train/policy_loss = 1.2989298105239868 | train/value_loss = 4.320833683013916 | time/iterations = 700 | rollout/ep_rew_mean = 35.224489795918366 | rollout/ep_len_mean = 35.224489795918366 | time/fps = 68 | time/time_elapsed = 51 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 7.909536361694336e-05 | train/entropy_loss = -0.6238623857498169 | train/policy_loss = 1.2989298105239868 | train/value_loss = 4.320833683013916 | time/iterations = 700 | rollout/ep_rew_mean = 35.224489795918366 | rollout/ep_len_mean = 35.224489795918366 | time/fps = 68 | time/time_elapsed = 51 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:10: [A2C[worker: 3]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.01001882553100586 | train/entropy_loss = -0.6652368307113647 | train/policy_loss = 1.111027479171753 | train/value_loss = 4.29358434677124 | time/iterations = 800 | rollout/ep_rew_mean = 33.16 | rollout/ep_len_mean = 33.16 | time/fps = 71 | time/time_elapsed = 56 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.01001882553100586 | train/entropy_loss = -0.6652368307113647 | train/policy_loss = 1.111027479171753 | train/value_loss = 4.29358434677124 | time/iterations = 800 | rollout/ep_rew_mean = 33.16 | rollout/ep_len_mean = 33.16 | time/fps = 71 | time/time_elapsed = 56 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:10: [A2C[worker: 4]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.00047838687896728516 | train/entropy_loss = -0.4655945897102356 | train/policy_loss = -19.256349563598633 | train/value_loss = 1361.687744140625 | time/iterations = 800 | rollout/ep_rew_mean = 37.07 | rollout/ep_len_mean = 37.07 | time/fps = 70 | time/time_elapsed = 56 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.00047838687896728516 | train/entropy_loss = -0.4655945897102356 | train/policy_loss = -19.256349563598633 | train/value_loss = 1361.687744140625 | time/iterations = 800 | rollout/ep_rew_mean = 37.07 | rollout/ep_len_mean = 37.07 | time/fps = 70 | time/time_elapsed = 56 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:10: [A2C[worker: 2]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0012837648391723633 | train/entropy_loss = -0.5026782751083374 | train/policy_loss = 0.9535248875617981 | train/value_loss = 3.7278473377227783 | time/iterations = 800 | rollout/ep_rew_mean = 51.81818181818182 | rollout/ep_len_mean = 51.81818181818182 | time/fps = 70 | time/time_elapsed = 56 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0012837648391723633 | train/entropy_loss = -0.5026782751083374 | train/policy_loss = 0.9535248875617981 | train/value_loss = 3.7278473377227783 | time/iterations = 800 | rollout/ep_rew_mean = 51.81818181818182 | rollout/ep_len_mean = 51.81818181818182 | time/fps = 70 | time/time_elapsed = 56 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:10: [A2C[worker: 0]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0021058320999145508 | train/entropy_loss = -0.5611809492111206 | train/policy_loss = 1.0986288785934448 | train/value_loss = 4.1130266189575195 | time/iterations = 800 | rollout/ep_rew_mean = 31.96 | rollout/ep_len_mean = 31.96 | time/fps = 69 | time/time_elapsed = 57 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0021058320999145508 | train/entropy_loss = -0.5611809492111206 | train/policy_loss = 1.0986288785934448 | train/value_loss = 4.1130266189575195 | time/iterations = 800 | rollout/ep_rew_mean = 31.96 | rollout/ep_len_mean = 31.96 | time/fps = 69 | time/time_elapsed = 57 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:10: [A2C[worker: 1]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.0030864477157592773 | train/entropy_loss = -0.5590025186538696 | train/policy_loss = 1.441819190979004 | train/value_loss = 3.7370200157165527 | time/iterations = 800 | rollout/ep_rew_mean = 39.23 | rollout/ep_len_mean = 39.23 | time/fps = 68 | time/time_elapsed = 58 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.0030864477157592773 | train/entropy_loss = -0.5590025186538696 | train/policy_loss = 1.441819190979004 | train/value_loss = 3.7370200157165527 | time/iterations = 800 | rollout/ep_rew_mean = 39.23 | rollout/ep_len_mean = 39.23 | time/fps = 68 | time/time_elapsed = 58 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:10: [A2C[worker: 3]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0038623809814453125 | train/entropy_loss = -0.5975967645645142 | train/policy_loss = 1.1042768955230713 | train/value_loss = 3.834260940551758 | time/iterations = 900 | rollout/ep_rew_mean = 34.95 | rollout/ep_len_mean = 34.95 | time/fps = 71 | time/time_elapsed = 62 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0038623809814453125 | train/entropy_loss = -0.5975967645645142 | train/policy_loss = 1.1042768955230713 | train/value_loss = 3.834260940551758 | time/iterations = 900 | rollout/ep_rew_mean = 34.95 | rollout/ep_len_mean = 34.95 | time/fps = 71 | time/time_elapsed = 62 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:10: [A2C[worker: 4]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0005697011947631836 | train/entropy_loss = -0.6188932061195374 | train/policy_loss = 0.9269369840621948 | train/value_loss = 3.3753273487091064 | time/iterations = 900 | rollout/ep_rew_mean = 40.21 | rollout/ep_len_mean = 40.21 | time/fps = 71 | time/time_elapsed = 63 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0005697011947631836 | train/entropy_loss = -0.6188932061195374 | train/policy_loss = 0.9269369840621948 | train/value_loss = 3.3753273487091064 | time/iterations = 900 | rollout/ep_rew_mean = 40.21 | rollout/ep_len_mean = 40.21 | time/fps = 71 | time/time_elapsed = 63 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:10: [A2C[worker: 2]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0016261935234069824 | train/entropy_loss = -0.5243479609489441 | train/policy_loss = 0.44117680191993713 | train/value_loss = 3.202864170074463 | time/iterations = 900 | rollout/ep_rew_mean = 55.6 | rollout/ep_len_mean = 55.6 | time/fps = 70 | time/time_elapsed = 63 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.0016261935234069824 | train/entropy_loss = -0.5243479609489441 | train/policy_loss = 0.44117680191993713 | train/value_loss = 3.202864170074463 | time/iterations = 900 | rollout/ep_rew_mean = 55.6 | rollout/ep_len_mean = 55.6 | time/fps = 70 | time/time_elapsed = 63 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:10: [A2C[worker: 0]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.00245821475982666 | train/entropy_loss = -0.6555637121200562 | train/policy_loss = 0.9260631799697876 | train/value_loss = 3.6484603881835938 | time/iterations = 900 | rollout/ep_rew_mean = 35.6 | rollout/ep_len_mean = 35.6 | time/fps = 70 | time/time_elapsed = 63 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.00245821475982666 | train/entropy_loss = -0.6555637121200562 | train/policy_loss = 0.9260631799697876 | train/value_loss = 3.6484603881835938 | time/iterations = 900 | rollout/ep_rew_mean = 35.6 | rollout/ep_len_mean = 35.6 | time/fps = 70 | time/time_elapsed = 63 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:10: [A2C[worker: 1]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0001627206802368164 | train/entropy_loss = -0.5737690925598145 | train/policy_loss = 1.0268001556396484 | train/value_loss = 3.2679035663604736 | time/iterations = 900 | rollout/ep_rew_mean = 42.38 | rollout/ep_len_mean = 42.38 | time/fps = 69 | time/time_elapsed = 64 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0001627206802368164 | train/entropy_loss = -0.5737690925598145 | train/policy_loss = 1.0268001556396484 | train/value_loss = 3.2679035663604736 | time/iterations = 900 | rollout/ep_rew_mean = 42.38 | rollout/ep_len_mean = 42.38 | time/fps = 69 | time/time_elapsed = 64 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:10: [A2C[worker: 3]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.003239870071411133 | train/entropy_loss = -0.5872007608413696 | train/policy_loss = 0.8767257928848267 | train/value_loss = 3.350132465362549 | time/iterations = 1000 | rollout/ep_rew_mean = 37.83 | rollout/ep_len_mean = 37.83 | time/fps = 72 | time/time_elapsed = 69 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.003239870071411133 | train/entropy_loss = -0.5872007608413696 | train/policy_loss = 0.8767257928848267 | train/value_loss = 3.350132465362549 | time/iterations = 1000 | rollout/ep_rew_mean = 37.83 | rollout/ep_len_mean = 37.83 | time/fps = 72 | time/time_elapsed = 69 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:10: [A2C[worker: 4]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.002167820930480957 | train/entropy_loss = -0.642724871635437 | train/policy_loss = 0.8404825329780579 | train/value_loss = 2.895968198776245 | time/iterations = 1000 | rollout/ep_rew_mean = 44.8 | rollout/ep_len_mean = 44.8 | time/fps = 71 | time/time_elapsed = 69 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.002167820930480957 | train/entropy_loss = -0.642724871635437 | train/policy_loss = 0.8404825329780579 | train/value_loss = 2.895968198776245 | time/iterations = 1000 | rollout/ep_rew_mean = 44.8 | rollout/ep_len_mean = 44.8 | time/fps = 71 | time/time_elapsed = 69 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:10: [A2C[worker: 2]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.004864990711212158 | train/entropy_loss = -0.42632031440734863 | train/policy_loss = 0.6649324297904968 | train/value_loss = 2.720348834991455 | time/iterations = 1000 | rollout/ep_rew_mean = 58.75903614457831 | rollout/ep_len_mean = 58.75903614457831 | time/fps = 71 | time/time_elapsed = 70 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.004864990711212158 | train/entropy_loss = -0.42632031440734863 | train/policy_loss = 0.6649324297904968 | train/value_loss = 2.720348834991455 | time/iterations = 1000 | rollout/ep_rew_mean = 58.75903614457831 | rollout/ep_len_mean = 58.75903614457831 | time/fps = 71 | time/time_elapsed = 70 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:10: [A2C[worker: 0]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0012153387069702148 | train/entropy_loss = -0.6796613931655884 | train/policy_loss = 0.9365519285202026 | train/value_loss = 3.1349785327911377 | time/iterations = 1000 | rollout/ep_rew_mean = 37.99 | rollout/ep_len_mean = 37.99 | time/fps = 70 | time/time_elapsed = 70 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0012153387069702148 | train/entropy_loss = -0.6796613931655884 | train/policy_loss = 0.9365519285202026 | train/value_loss = 3.1349785327911377 | time/iterations = 1000 | rollout/ep_rew_mean = 37.99 | rollout/ep_len_mean = 37.99 | time/fps = 70 | time/time_elapsed = 70 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:10: [A2C[worker: 1]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.000780940055847168 | train/entropy_loss = -0.537676990032196 | train/policy_loss = 1.2888538837432861 | train/value_loss = 2.822640895843506 | time/iterations = 1000 | rollout/ep_rew_mean = 45.5 | rollout/ep_len_mean = 45.5 | time/fps = 69 | time/time_elapsed = 71 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.000780940055847168 | train/entropy_loss = -0.537676990032196 | train/policy_loss = 1.2888538837432861 | train/value_loss = 2.822640895843506 | time/iterations = 1000 | rollout/ep_rew_mean = 45.5 | rollout/ep_len_mean = 45.5 | time/fps = 69 | time/time_elapsed = 71 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:10: [A2C[worker: 3]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0008558034896850586 | train/entropy_loss = -0.6026290655136108 | train/policy_loss = 0.7427920699119568 | train/value_loss = 2.8690831661224365 | time/iterations = 1100 | rollout/ep_rew_mean = 42.77 | rollout/ep_len_mean = 42.77 | time/fps = 72 | time/time_elapsed = 75 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0008558034896850586 | train/entropy_loss = -0.6026290655136108 | train/policy_loss = 0.7427920699119568 | train/value_loss = 2.8690831661224365 | time/iterations = 1100 | rollout/ep_rew_mean = 42.77 | rollout/ep_len_mean = 42.77 | time/fps = 72 | time/time_elapsed = 75 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:10: [A2C[worker: 4]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.00026148557662963867 | train/entropy_loss = -0.6533389091491699 | train/policy_loss = 0.8296110033988953 | train/value_loss = 2.4890694618225098 | time/iterations = 1100 | rollout/ep_rew_mean = 46.89 | rollout/ep_len_mean = 46.89 | time/fps = 72 | time/time_elapsed = 75 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.00026148557662963867 | train/entropy_loss = -0.6533389091491699 | train/policy_loss = 0.8296110033988953 | train/value_loss = 2.4890694618225098 | time/iterations = 1100 | rollout/ep_rew_mean = 46.89 | rollout/ep_len_mean = 46.89 | time/fps = 72 | time/time_elapsed = 75 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:10: [A2C[worker: 2]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0009788274765014648 | train/entropy_loss = -0.35958442091941833 | train/policy_loss = 0.7916496396064758 | train/value_loss = 2.2737746238708496 | time/iterations = 1100 | rollout/ep_rew_mean = 63.56976744186046 | rollout/ep_len_mean = 63.56976744186046 | time/fps = 71 | time/time_elapsed = 76 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0009788274765014648 | train/entropy_loss = -0.35958442091941833 | train/policy_loss = 0.7916496396064758 | train/value_loss = 2.2737746238708496 | time/iterations = 1100 | rollout/ep_rew_mean = 63.56976744186046 | rollout/ep_len_mean = 63.56976744186046 | time/fps = 71 | time/time_elapsed = 76 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:11: [A2C[worker: 0]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0007870197296142578 | train/entropy_loss = -0.5438396334648132 | train/policy_loss = 1.454790472984314 | train/value_loss = 2.6734135150909424 | time/iterations = 1100 | rollout/ep_rew_mean = 40.48 | rollout/ep_len_mean = 40.48 | time/fps = 71 | time/time_elapsed = 76 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0007870197296142578 | train/entropy_loss = -0.5438396334648132 | train/policy_loss = 1.454790472984314 | train/value_loss = 2.6734135150909424 | time/iterations = 1100 | rollout/ep_rew_mean = 40.48 | rollout/ep_len_mean = 40.48 | time/fps = 71 | time/time_elapsed = 76 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:11: [A2C[worker: 1]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.00020891427993774414 | train/entropy_loss = -0.6188745498657227 | train/policy_loss = 0.5653238296508789 | train/value_loss = 2.3908233642578125 | time/iterations = 1100 | rollout/ep_rew_mean = 49.5 | rollout/ep_len_mean = 49.5 | time/fps = 70 | time/time_elapsed = 78 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.00020891427993774414 | train/entropy_loss = -0.6188745498657227 | train/policy_loss = 0.5653238296508789 | train/value_loss = 2.3908233642578125 | time/iterations = 1100 | rollout/ep_rew_mean = 49.5 | rollout/ep_len_mean = 49.5 | time/fps = 70 | time/time_elapsed = 78 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:11: [A2C[worker: 3]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00010502338409423828 | train/entropy_loss = -0.46746692061424255 | train/policy_loss = 1.3082832098007202 | train/value_loss = 2.4219207763671875 | time/iterations = 1200 | rollout/ep_rew_mean = 46.39 | rollout/ep_len_mean = 46.39 | time/fps = 73 | time/time_elapsed = 81 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00010502338409423828 | train/entropy_loss = -0.46746692061424255 | train/policy_loss = 1.3082832098007202 | train/value_loss = 2.4219207763671875 | time/iterations = 1200 | rollout/ep_rew_mean = 46.39 | rollout/ep_len_mean = 46.39 | time/fps = 73 | time/time_elapsed = 81 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:11: [A2C[worker: 4]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -2.4557113647460938e-05 | train/entropy_loss = -0.5854228734970093 | train/policy_loss = 0.8072751760482788 | train/value_loss = 2.093355894088745 | time/iterations = 1200 | rollout/ep_rew_mean = 50.42 | rollout/ep_len_mean = 50.42 | time/fps = 72 | time/time_elapsed = 82 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -2.4557113647460938e-05 | train/entropy_loss = -0.5854228734970093 | train/policy_loss = 0.8072751760482788 | train/value_loss = 2.093355894088745 | time/iterations = 1200 | rollout/ep_rew_mean = 50.42 | rollout/ep_len_mean = 50.42 | time/fps = 72 | time/time_elapsed = 82 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:11: [A2C[worker: 2]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00012636184692382812 | train/entropy_loss = -0.46680188179016113 | train/policy_loss = 0.598046600818634 | train/value_loss = 1.8708425760269165 | time/iterations = 1200 | rollout/ep_rew_mean = 66.15730337078652 | rollout/ep_len_mean = 66.15730337078652 | time/fps = 72 | time/time_elapsed = 82 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00012636184692382812 | train/entropy_loss = -0.46680188179016113 | train/policy_loss = 0.598046600818634 | train/value_loss = 1.8708425760269165 | time/iterations = 1200 | rollout/ep_rew_mean = 66.15730337078652 | rollout/ep_len_mean = 66.15730337078652 | time/fps = 72 | time/time_elapsed = 82 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:11: [A2C[worker: 0]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.0020155906677246094 | train/entropy_loss = -0.5868368744850159 | train/policy_loss = 0.49572673439979553 | train/value_loss = 2.243499279022217 | time/iterations = 1200 | rollout/ep_rew_mean = 46.07 | rollout/ep_len_mean = 46.07 | time/fps = 71 | time/time_elapsed = 83 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.0020155906677246094 | train/entropy_loss = -0.5868368744850159 | train/policy_loss = 0.49572673439979553 | train/value_loss = 2.243499279022217 | time/iterations = 1200 | rollout/ep_rew_mean = 46.07 | rollout/ep_len_mean = 46.07 | time/fps = 71 | time/time_elapsed = 83 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:11: [A2C[worker: 1]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 4.738569259643555e-05 | train/entropy_loss = -0.6563912630081177 | train/policy_loss = 0.6471293568611145 | train/value_loss = 1.9692847728729248 | time/iterations = 1200 | rollout/ep_rew_mean = 53.53 | rollout/ep_len_mean = 53.53 | time/fps = 71 | time/time_elapsed = 84 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 4.738569259643555e-05 | train/entropy_loss = -0.6563912630081177 | train/policy_loss = 0.6471293568611145 | train/value_loss = 1.9692847728729248 | time/iterations = 1200 | rollout/ep_rew_mean = 53.53 | rollout/ep_len_mean = 53.53 | time/fps = 71 | time/time_elapsed = 84 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:11: [A2C[worker: 3]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.00027936697006225586 | train/entropy_loss = -0.5301194190979004 | train/policy_loss = 0.5963245630264282 | train/value_loss = 2.021951675415039 | time/iterations = 1300 | rollout/ep_rew_mean = 50.34 | rollout/ep_len_mean = 50.34 | time/fps = 73 | time/time_elapsed = 88 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.00027936697006225586 | train/entropy_loss = -0.5301194190979004 | train/policy_loss = 0.5963245630264282 | train/value_loss = 2.021951675415039 | time/iterations = 1300 | rollout/ep_rew_mean = 50.34 | rollout/ep_len_mean = 50.34 | time/fps = 73 | time/time_elapsed = 88 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:11: [A2C[worker: 2]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.00036847591400146484 | train/entropy_loss = -0.39183780550956726 | train/policy_loss = 0.33890220522880554 | train/value_loss = 1.499272346496582 | time/iterations = 1300 | rollout/ep_rew_mean = 69.83516483516483 | rollout/ep_len_mean = 69.83516483516483 | time/fps = 73 | time/time_elapsed = 89 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.00036847591400146484 | train/entropy_loss = -0.39183780550956726 | train/policy_loss = 0.33890220522880554 | train/value_loss = 1.499272346496582 | time/iterations = 1300 | rollout/ep_rew_mean = 69.83516483516483 | rollout/ep_len_mean = 69.83516483516483 | time/fps = 73 | time/time_elapsed = 89 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:11: [A2C[worker: 4]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -6.0558319091796875e-05 | train/entropy_loss = -0.5779048204421997 | train/policy_loss = 0.762768566608429 | train/value_loss = 1.701123833656311 | time/iterations = 1300 | rollout/ep_rew_mean = 52.04 | rollout/ep_len_mean = 52.04 | time/fps = 73 | time/time_elapsed = 88 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -6.0558319091796875e-05 | train/entropy_loss = -0.5779048204421997 | train/policy_loss = 0.762768566608429 | train/value_loss = 1.701123833656311 | time/iterations = 1300 | rollout/ep_rew_mean = 52.04 | rollout/ep_len_mean = 52.04 | time/fps = 73 | time/time_elapsed = 88 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:11: [A2C[worker: 0]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0007071495056152344 | train/entropy_loss = -0.5344122052192688 | train/policy_loss = 0.40236377716064453 | train/value_loss = 1.8453998565673828 | time/iterations = 1300 | rollout/ep_rew_mean = 50.43 | rollout/ep_len_mean = 50.43 | time/fps = 72 | time/time_elapsed = 89 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0007071495056152344 | train/entropy_loss = -0.5344122052192688 | train/policy_loss = 0.40236377716064453 | train/value_loss = 1.8453998565673828 | time/iterations = 1300 | rollout/ep_rew_mean = 50.43 | rollout/ep_len_mean = 50.43 | time/fps = 72 | time/time_elapsed = 89 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:11: [A2C[worker: 1]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -2.2411346435546875e-05 | train/entropy_loss = -0.48835015296936035 | train/policy_loss = 0.5278348922729492 | train/value_loss = 1.59037184715271 | time/iterations = 1300 | rollout/ep_rew_mean = 59.47 | rollout/ep_len_mean = 59.47 | time/fps = 71 | time/time_elapsed = 90 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -2.2411346435546875e-05 | train/entropy_loss = -0.48835015296936035 | train/policy_loss = 0.5278348922729492 | train/value_loss = 1.59037184715271 | time/iterations = 1300 | rollout/ep_rew_mean = 59.47 | rollout/ep_len_mean = 59.47 | time/fps = 71 | time/time_elapsed = 90 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:11: [A2C[worker: 3]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.0010071992874145508 | train/entropy_loss = -0.40152043104171753 | train/policy_loss = 0.7328354120254517 | train/value_loss = 1.6364408731460571 | time/iterations = 1400 | rollout/ep_rew_mean = 55.62 | rollout/ep_len_mean = 55.62 | time/fps = 74 | time/time_elapsed = 94 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.0010071992874145508 | train/entropy_loss = -0.40152043104171753 | train/policy_loss = 0.7328354120254517 | train/value_loss = 1.6364408731460571 | time/iterations = 1400 | rollout/ep_rew_mean = 55.62 | rollout/ep_len_mean = 55.62 | time/fps = 74 | time/time_elapsed = 94 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:11: [A2C[worker: 2]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.0003075599670410156 | train/entropy_loss = -0.3762759864330292 | train/policy_loss = 0.35749202966690063 | train/value_loss = 1.170969009399414 | time/iterations = 1400 | rollout/ep_rew_mean = 72.19354838709677 | rollout/ep_len_mean = 72.19354838709677 | time/fps = 73 | time/time_elapsed = 95 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.0003075599670410156 | train/entropy_loss = -0.3762759864330292 | train/policy_loss = 0.35749202966690063 | train/value_loss = 1.170969009399414 | time/iterations = 1400 | rollout/ep_rew_mean = 72.19354838709677 | rollout/ep_len_mean = 72.19354838709677 | time/fps = 73 | time/time_elapsed = 95 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:11: [A2C[worker: 4]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.0003737211227416992 | train/entropy_loss = -0.6379235982894897 | train/policy_loss = 0.4677971303462982 | train/value_loss = 1.369717001914978 | time/iterations = 1400 | rollout/ep_rew_mean = 58.44 | rollout/ep_len_mean = 58.44 | time/fps = 73 | time/time_elapsed = 95 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.0003737211227416992 | train/entropy_loss = -0.6379235982894897 | train/policy_loss = 0.4677971303462982 | train/value_loss = 1.369717001914978 | time/iterations = 1400 | rollout/ep_rew_mean = 58.44 | rollout/ep_len_mean = 58.44 | time/fps = 73 | time/time_elapsed = 95 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:11: [A2C[worker: 0]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.00048345327377319336 | train/entropy_loss = -0.6421979069709778 | train/policy_loss = 0.5580103993415833 | train/value_loss = 1.5114226341247559 | time/iterations = 1400 | rollout/ep_rew_mean = 54.36 | rollout/ep_len_mean = 54.36 | time/fps = 72 | time/time_elapsed = 96 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.00048345327377319336 | train/entropy_loss = -0.6421979069709778 | train/policy_loss = 0.5580103993415833 | train/value_loss = 1.5114226341247559 | time/iterations = 1400 | rollout/ep_rew_mean = 54.36 | rollout/ep_len_mean = 54.36 | time/fps = 72 | time/time_elapsed = 96 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:11: [A2C[worker: 1]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.00012803077697753906 | train/entropy_loss = -0.5785781145095825 | train/policy_loss = 0.43702203035354614 | train/value_loss = 1.2515429258346558 | time/iterations = 1400 | rollout/ep_rew_mean = 64.43 | rollout/ep_len_mean = 64.43 | time/fps = 72 | time/time_elapsed = 97 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.00012803077697753906 | train/entropy_loss = -0.5785781145095825 | train/policy_loss = 0.43702203035354614 | train/value_loss = 1.2515429258346558 | time/iterations = 1400 | rollout/ep_rew_mean = 64.43 | rollout/ep_len_mean = 64.43 | time/fps = 72 | time/time_elapsed = 97 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:11: [A2C[worker: 3]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.00029528141021728516 | train/entropy_loss = -0.3190804123878479 | train/policy_loss = 1.2765357494354248 | train/value_loss = 1.3003696203231812 | time/iterations = 1500 | rollout/ep_rew_mean = 58.24 | rollout/ep_len_mean = 58.24 | time/fps = 74 | time/time_elapsed = 100 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.00029528141021728516 | train/entropy_loss = -0.3190804123878479 | train/policy_loss = 1.2765357494354248 | train/value_loss = 1.3003696203231812 | time/iterations = 1500 | rollout/ep_rew_mean = 58.24 | rollout/ep_len_mean = 58.24 | time/fps = 74 | time/time_elapsed = 100 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:11: [A2C[worker: 4]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.0002982020378112793 | train/entropy_loss = -0.5977676510810852 | train/policy_loss = 0.5758155584335327 | train/value_loss = 1.050067663192749 | time/iterations = 1500 | rollout/ep_rew_mean = 62.62 | rollout/ep_len_mean = 62.62 | time/fps = 74 | time/time_elapsed = 101 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.0002982020378112793 | train/entropy_loss = -0.5977676510810852 | train/policy_loss = 0.5758155584335327 | train/value_loss = 1.050067663192749 | time/iterations = 1500 | rollout/ep_rew_mean = 62.62 | rollout/ep_len_mean = 62.62 | time/fps = 74 | time/time_elapsed = 101 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:11: [A2C[worker: 2]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 1.2040138244628906e-05 | train/entropy_loss = -0.5119199156761169 | train/policy_loss = 0.20188558101654053 | train/value_loss = 0.889872670173645 | time/iterations = 1500 | rollout/ep_rew_mean = 77.11458333333333 | rollout/ep_len_mean = 77.11458333333333 | time/fps = 74 | time/time_elapsed = 101 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 1.2040138244628906e-05 | train/entropy_loss = -0.5119199156761169 | train/policy_loss = 0.20188558101654053 | train/value_loss = 0.889872670173645 | time/iterations = 1500 | rollout/ep_rew_mean = 77.11458333333333 | rollout/ep_len_mean = 77.11458333333333 | time/fps = 74 | time/time_elapsed = 101 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:11: [A2C[worker: 0]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0003472566604614258 | train/entropy_loss = -0.5310603976249695 | train/policy_loss = 0.681984543800354 | train/value_loss = 1.2017850875854492 | time/iterations = 1500 | rollout/ep_rew_mean = 58.43 | rollout/ep_len_mean = 58.43 | time/fps = 73 | time/time_elapsed = 102 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0003472566604614258 | train/entropy_loss = -0.5310603976249695 | train/policy_loss = 0.681984543800354 | train/value_loss = 1.2017850875854492 | time/iterations = 1500 | rollout/ep_rew_mean = 58.43 | rollout/ep_len_mean = 58.43 | time/fps = 73 | time/time_elapsed = 102 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:11: [A2C[worker: 1]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.00019299983978271484 | train/entropy_loss = -0.5694940090179443 | train/policy_loss = 0.30151456594467163 | train/value_loss = 0.9517377018928528 | time/iterations = 1500 | rollout/ep_rew_mean = 68.08 | rollout/ep_len_mean = 68.08 | time/fps = 72 | time/time_elapsed = 103 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.00019299983978271484 | train/entropy_loss = -0.5694940090179443 | train/policy_loss = 0.30151456594467163 | train/value_loss = 0.9517377018928528 | time/iterations = 1500 | rollout/ep_rew_mean = 68.08 | rollout/ep_len_mean = 68.08 | time/fps = 72 | time/time_elapsed = 103 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:11: [A2C[worker: 3]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -1.0371208190917969e-05 | train/entropy_loss = -0.5802566409111023 | train/policy_loss = 0.3626578450202942 | train/value_loss = 0.9898215532302856 | time/iterations = 1600 | rollout/ep_rew_mean = 64.47 | rollout/ep_len_mean = 64.47 | time/fps = 74 | time/time_elapsed = 106 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -1.0371208190917969e-05 | train/entropy_loss = -0.5802566409111023 | train/policy_loss = 0.3626578450202942 | train/value_loss = 0.9898215532302856 | time/iterations = 1600 | rollout/ep_rew_mean = 64.47 | rollout/ep_len_mean = 64.47 | time/fps = 74 | time/time_elapsed = 106 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:11: [A2C[worker: 4]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -3.981590270996094e-05 | train/entropy_loss = -0.6060147881507874 | train/policy_loss = 0.28232359886169434 | train/value_loss = 0.7835298776626587 | time/iterations = 1600 | rollout/ep_rew_mean = 67.24 | rollout/ep_len_mean = 67.24 | time/fps = 74 | time/time_elapsed = 107 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -3.981590270996094e-05 | train/entropy_loss = -0.6060147881507874 | train/policy_loss = 0.28232359886169434 | train/value_loss = 0.7835298776626587 | time/iterations = 1600 | rollout/ep_rew_mean = 67.24 | rollout/ep_len_mean = 67.24 | time/fps = 74 | time/time_elapsed = 107 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:11: [A2C[worker: 2]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.00012153387069702148 | train/entropy_loss = -0.3500787615776062 | train/policy_loss = 0.38296499848365784 | train/value_loss = 0.6388632655143738 | time/iterations = 1600 | rollout/ep_rew_mean = 80.07142857142857 | rollout/ep_len_mean = 80.07142857142857 | time/fps = 74 | time/time_elapsed = 107 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 0.00012153387069702148 | train/entropy_loss = -0.3500787615776062 | train/policy_loss = 0.38296499848365784 | train/value_loss = 0.6388632655143738 | time/iterations = 1600 | rollout/ep_rew_mean = 80.07142857142857 | rollout/ep_len_mean = 80.07142857142857 | time/fps = 74 | time/time_elapsed = 107 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:11: [A2C[worker: 0]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 3.0159950256347656e-05 | train/entropy_loss = -0.5429983139038086 | train/policy_loss = -32.428367614746094 | train/value_loss = 2828.096435546875 | time/iterations = 1600 | rollout/ep_rew_mean = 62.28 | rollout/ep_len_mean = 62.28 | time/fps = 73 | time/time_elapsed = 108 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 3.0159950256347656e-05 | train/entropy_loss = -0.5429983139038086 | train/policy_loss = -32.428367614746094 | train/value_loss = 2828.096435546875 | time/iterations = 1600 | rollout/ep_rew_mean = 62.28 | rollout/ep_len_mean = 62.28 | time/fps = 73 | time/time_elapsed = 108 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:11: [A2C[worker: 1]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 2.4080276489257812e-05 | train/entropy_loss = -0.6429725885391235 | train/policy_loss = 0.32282739877700806 | train/value_loss = 0.6948233842849731 | time/iterations = 1600 | rollout/ep_rew_mean = 73.44 | rollout/ep_len_mean = 73.44 | time/fps = 72 | time/time_elapsed = 109 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 2.4080276489257812e-05 | train/entropy_loss = -0.6429725885391235 | train/policy_loss = 0.32282739877700806 | train/value_loss = 0.6948233842849731 | time/iterations = 1600 | rollout/ep_rew_mean = 73.44 | rollout/ep_len_mean = 73.44 | time/fps = 72 | time/time_elapsed = 109 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:11: [A2C[worker: 3]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 1.6927719116210938e-05 | train/entropy_loss = -0.5426797270774841 | train/policy_loss = 0.24019643664360046 | train/value_loss = 0.7405223250389099 | time/iterations = 1700 | rollout/ep_rew_mean = 67.88 | rollout/ep_len_mean = 67.88 | time/fps = 75 | time/time_elapsed = 112 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 1.6927719116210938e-05 | train/entropy_loss = -0.5426797270774841 | train/policy_loss = 0.24019643664360046 | train/value_loss = 0.7405223250389099 | time/iterations = 1700 | rollout/ep_rew_mean = 67.88 | rollout/ep_len_mean = 67.88 | time/fps = 75 | time/time_elapsed = 112 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:11: [A2C[worker: 4]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -0.0003135204315185547 | train/entropy_loss = -0.46096301078796387 | train/policy_loss = 0.3570004105567932 | train/value_loss = 0.565159797668457 | time/iterations = 1700 | rollout/ep_rew_mean = 71.64 | rollout/ep_len_mean = 71.64 | time/fps = 74 | time/time_elapsed = 113 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -0.0003135204315185547 | train/entropy_loss = -0.46096301078796387 | train/policy_loss = 0.3570004105567932 | train/value_loss = 0.565159797668457 | time/iterations = 1700 | rollout/ep_rew_mean = 71.64 | rollout/ep_len_mean = 71.64 | time/fps = 74 | time/time_elapsed = 113 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:11: [A2C[worker: 2]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -1.6689300537109375e-05 | train/entropy_loss = -0.37788447737693787 | train/policy_loss = 0.3995831608772278 | train/value_loss = 0.4341737627983093 | time/iterations = 1700 | rollout/ep_rew_mean = 83.83 | rollout/ep_len_mean = 83.83 | time/fps = 74 | time/time_elapsed = 113 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -1.6689300537109375e-05 | train/entropy_loss = -0.37788447737693787 | train/policy_loss = 0.3995831608772278 | train/value_loss = 0.4341737627983093 | time/iterations = 1700 | rollout/ep_rew_mean = 83.83 | rollout/ep_len_mean = 83.83 | time/fps = 74 | time/time_elapsed = 113 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:11: [A2C[worker: 0]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -4.9591064453125e-05 | train/entropy_loss = -0.5278026461601257 | train/policy_loss = 0.37457919120788574 | train/value_loss = 0.6811152696609497 | time/iterations = 1700 | rollout/ep_rew_mean = 64.74 | rollout/ep_len_mean = 64.74 | time/fps = 74 | time/time_elapsed = 114 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -4.9591064453125e-05 | train/entropy_loss = -0.5278026461601257 | train/policy_loss = 0.37457919120788574 | train/value_loss = 0.6811152696609497 | time/iterations = 1700 | rollout/ep_rew_mean = 64.74 | rollout/ep_len_mean = 64.74 | time/fps = 74 | time/time_elapsed = 114 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:11: [A2C[worker: 1]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 7.468461990356445e-05 | train/entropy_loss = -0.5998739004135132 | train/policy_loss = 0.3350768983364105 | train/value_loss = 0.47716188430786133 | time/iterations = 1700 | rollout/ep_rew_mean = 78.43 | rollout/ep_len_mean = 78.43 | time/fps = 73 | time/time_elapsed = 116 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 7.468461990356445e-05 | train/entropy_loss = -0.5998739004135132 | train/policy_loss = 0.3350768983364105 | train/value_loss = 0.47716188430786133 | time/iterations = 1700 | rollout/ep_rew_mean = 78.43 | rollout/ep_len_mean = 78.43 | time/fps = 73 | time/time_elapsed = 116 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:11: [A2C[worker: 3]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 0.0003592371940612793 | train/entropy_loss = -0.43005475401878357 | train/policy_loss = 0.23602411150932312 | train/value_loss = 0.5289186239242554 | time/iterations = 1800 | rollout/ep_rew_mean = 70.64 | rollout/ep_len_mean = 70.64 | time/fps = 75 | time/time_elapsed = 119 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 0.0003592371940612793 | train/entropy_loss = -0.43005475401878357 | train/policy_loss = 0.23602411150932312 | train/value_loss = 0.5289186239242554 | time/iterations = 1800 | rollout/ep_rew_mean = 70.64 | rollout/ep_len_mean = 70.64 | time/fps = 75 | time/time_elapsed = 119 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:11: [A2C[worker: 4]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -2.384185791015625e-07 | train/entropy_loss = -0.6743599772453308 | train/policy_loss = 0.2998552918434143 | train/value_loss = 0.3737374246120453 | time/iterations = 1800 | rollout/ep_rew_mean = 75.66 | rollout/ep_len_mean = 75.66 | time/fps = 75 | time/time_elapsed = 119 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -2.384185791015625e-07 | train/entropy_loss = -0.6743599772453308 | train/policy_loss = 0.2998552918434143 | train/value_loss = 0.3737374246120453 | time/iterations = 1800 | rollout/ep_rew_mean = 75.66 | rollout/ep_len_mean = 75.66 | time/fps = 75 | time/time_elapsed = 119 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:11: [A2C[worker: 2]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -3.933906555175781e-05 | train/entropy_loss = -0.29142269492149353 | train/policy_loss = 0.537980318069458 | train/value_loss = 0.2667010426521301 | time/iterations = 1800 | rollout/ep_rew_mean = 87.99 | rollout/ep_len_mean = 87.99 | time/fps = 74 | time/time_elapsed = 120 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -3.933906555175781e-05 | train/entropy_loss = -0.29142269492149353 | train/policy_loss = 0.537980318069458 | train/value_loss = 0.2667010426521301 | time/iterations = 1800 | rollout/ep_rew_mean = 87.99 | rollout/ep_len_mean = 87.99 | time/fps = 74 | time/time_elapsed = 120 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:11: [A2C[worker: 0]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -1.4662742614746094e-05 | train/entropy_loss = -0.4666554033756256 | train/policy_loss = 0.590783417224884 | train/value_loss = 0.47439780831336975 | time/iterations = 1800 | rollout/ep_rew_mean = 69.17 | rollout/ep_len_mean = 69.17 | time/fps = 74 | time/time_elapsed = 120 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -1.4662742614746094e-05 | train/entropy_loss = -0.4666554033756256 | train/policy_loss = 0.590783417224884 | train/value_loss = 0.47439780831336975 | time/iterations = 1800 | rollout/ep_rew_mean = 69.17 | rollout/ep_len_mean = 69.17 | time/fps = 74 | time/time_elapsed = 120 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:11: [A2C[worker: 1]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -1.8477439880371094e-05 | train/entropy_loss = -0.6329901814460754 | train/policy_loss = 0.29739221930503845 | train/value_loss = 0.30018362402915955 | time/iterations = 1800 | rollout/ep_rew_mean = 80.65 | rollout/ep_len_mean = 80.65 | time/fps = 73 | time/time_elapsed = 122 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -1.8477439880371094e-05 | train/entropy_loss = -0.6329901814460754 | train/policy_loss = 0.29739221930503845 | train/value_loss = 0.30018362402915955 | time/iterations = 1800 | rollout/ep_rew_mean = 80.65 | rollout/ep_len_mean = 80.65 | time/fps = 73 | time/time_elapsed = 122 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:11: [A2C[worker: 3]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.00012063980102539062 | train/entropy_loss = -0.48388099670410156 | train/policy_loss = 0.19817300140857697 | train/value_loss = 0.3443290591239929 | time/iterations = 1900 | rollout/ep_rew_mean = 72.82 | rollout/ep_len_mean = 72.82 | time/fps = 75 | time/time_elapsed = 125 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.00012063980102539062 | train/entropy_loss = -0.48388099670410156 | train/policy_loss = 0.19817300140857697 | train/value_loss = 0.3443290591239929 | time/iterations = 1900 | rollout/ep_rew_mean = 72.82 | rollout/ep_len_mean = 72.82 | time/fps = 75 | time/time_elapsed = 125 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:11: [A2C[worker: 4]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.00012993812561035156 | train/entropy_loss = -0.626408576965332 | train/policy_loss = 0.23632553219795227 | train/value_loss = 0.22235234081745148 | time/iterations = 1900 | rollout/ep_rew_mean = 80.43 | rollout/ep_len_mean = 80.43 | time/fps = 75 | time/time_elapsed = 126 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.00012993812561035156 | train/entropy_loss = -0.626408576965332 | train/policy_loss = 0.23632553219795227 | train/value_loss = 0.22235234081745148 | time/iterations = 1900 | rollout/ep_rew_mean = 80.43 | rollout/ep_len_mean = 80.43 | time/fps = 75 | time/time_elapsed = 126 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:11: [A2C[worker: 2]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -5.4717063903808594e-05 | train/entropy_loss = -0.4353080689907074 | train/policy_loss = 0.06577054411172867 | train/value_loss = 0.13951340317726135 | time/iterations = 1900 | rollout/ep_rew_mean = 92.18 | rollout/ep_len_mean = 92.18 | time/fps = 74 | time/time_elapsed = 126 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -5.4717063903808594e-05 | train/entropy_loss = -0.4353080689907074 | train/policy_loss = 0.06577054411172867 | train/value_loss = 0.13951340317726135 | time/iterations = 1900 | rollout/ep_rew_mean = 92.18 | rollout/ep_len_mean = 92.18 | time/fps = 74 | time/time_elapsed = 126 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:11: [A2C[worker: 0]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 5.3822994232177734e-05 | train/entropy_loss = -0.5930700302124023 | train/policy_loss = 0.3122915029525757 | train/value_loss = 0.30490121245384216 | time/iterations = 1900 | rollout/ep_rew_mean = 73.97 | rollout/ep_len_mean = 73.97 | time/fps = 74 | time/time_elapsed = 127 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 5.3822994232177734e-05 | train/entropy_loss = -0.5930700302124023 | train/policy_loss = 0.3122915029525757 | train/value_loss = 0.30490121245384216 | time/iterations = 1900 | rollout/ep_rew_mean = 73.97 | rollout/ep_len_mean = 73.97 | time/fps = 74 | time/time_elapsed = 127 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:11: [A2C[worker: 1]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 2.2649765014648438e-06 | train/entropy_loss = -0.5214107632637024 | train/policy_loss = -25.07892608642578 | train/value_loss = 2964.38916015625 | time/iterations = 1900 | rollout/ep_rew_mean = 87.85 | rollout/ep_len_mean = 87.85 | time/fps = 73 | time/time_elapsed = 129 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 2.2649765014648438e-06 | train/entropy_loss = -0.5214107632637024 | train/policy_loss = -25.07892608642578 | train/value_loss = 2964.38916015625 | time/iterations = 1900 | rollout/ep_rew_mean = 87.85 | rollout/ep_len_mean = 87.85 | time/fps = 73 | time/time_elapsed = 129 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:11: ... trained! [0m
    INFO:rlberry_logger:... trained!
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:11: Saved ExperimentManager(A2C) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(A2C) using pickle.
    [38;21m[INFO] 15:11: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-09-36_93291f77/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-09-36_93291f77/manager_obj.pickle'
    [38;21m[INFO] 15:11: Running ExperimentManager fit() for PPO with n_fit = 5 and max_workers = None. [0m
    INFO:rlberry_logger:Running ExperimentManager fit() for PPO with n_fit = 5 and max_workers = None.
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:12:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        2           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        2           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:12:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        0           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        0           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:12:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        1           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        1           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:12:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        4           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        4           1               2048
    [38;21m[INFO] 15:12:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        3           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        3           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:12: [PPO[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.752688172043012 | rollout/ep_len_mean = 21.752688172043012 | time/fps = 154 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.752688172043012 | rollout/ep_len_mean = 21.752688172043012 | time/fps = 154 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:12: [PPO[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.043478260869566 | rollout/ep_len_mean = 22.043478260869566 | time/fps = 152 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.043478260869566 | rollout/ep_len_mean = 22.043478260869566 | time/fps = 152 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:12: [PPO[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.1875 | rollout/ep_len_mean = 21.1875 | time/fps = 156 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.1875 | rollout/ep_len_mean = 21.1875 | time/fps = 156 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:12: [PPO[worker: 4]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 24.0 | rollout/ep_len_mean = 24.0 | time/fps = 151 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 24.0 | rollout/ep_len_mean = 24.0 | time/fps = 151 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:12: [PPO[worker: 3]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.494505494505493 | rollout/ep_len_mean = 22.494505494505493 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.494505494505493 | rollout/ep_len_mean = 22.494505494505493 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    [38;21m[INFO] 15:13: [PPO[worker: 2]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.85 | rollout/ep_len_mean = 25.85 | time/fps = 107 | time/time_elapsed = 38 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6865362208336592 | train/policy_gradient_loss = -0.012163976750161964 | train/value_loss = 47.44922216832638 | train/approx_kl = 0.008370674215257168 | train/clip_fraction = 0.07109375 | train/loss = 5.725305557250977 | train/explained_variance = -0.010464787483215332 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.85 | rollout/ep_len_mean = 25.85 | time/fps = 107 | time/time_elapsed = 38 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6865362208336592 | train/policy_gradient_loss = -0.012163976750161964 | train/value_loss = 47.44922216832638 | train/approx_kl = 0.008370674215257168 | train/clip_fraction = 0.07109375 | train/loss = 5.725305557250977 | train/explained_variance = -0.010464787483215332 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.79 | rollout/ep_len_mean = 26.79 | time/fps = 108 | time/time_elapsed = 37 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6870102940127254 | train/policy_gradient_loss = -0.010426906173233875 | train/value_loss = 50.4704518288374 | train/approx_kl = 0.00866074301302433 | train/clip_fraction = 0.071923828125 | train/loss = 7.672800064086914 | train/explained_variance = -0.0013566017150878906 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 26.79 | rollout/ep_len_mean = 26.79 | time/fps = 108 | time/time_elapsed = 37 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6870102940127254 | train/policy_gradient_loss = -0.010426906173233875 | train/value_loss = 50.4704518288374 | train/approx_kl = 0.00866074301302433 | train/clip_fraction = 0.071923828125 | train/loss = 7.672800064086914 | train/explained_variance = -0.0013566017150878906 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.18 | rollout/ep_len_mean = 27.18 | time/fps = 108 | time/time_elapsed = 37 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6867541592568159 | train/policy_gradient_loss = -0.011159516370389611 | train/value_loss = 51.5323302000761 | train/approx_kl = 0.007951025851070881 | train/clip_fraction = 0.075146484375 | train/loss = 8.009339332580566 | train/explained_variance = 0.008511781692504883 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.18 | rollout/ep_len_mean = 27.18 | time/fps = 108 | time/time_elapsed = 37 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6867541592568159 | train/policy_gradient_loss = -0.011159516370389611 | train/value_loss = 51.5323302000761 | train/approx_kl = 0.007951025851070881 | train/clip_fraction = 0.075146484375 | train/loss = 8.009339332580566 | train/explained_variance = 0.008511781692504883 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 4]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.41 | rollout/ep_len_mean = 27.41 | time/fps = 106 | time/time_elapsed = 38 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.686777251213789 | train/policy_gradient_loss = -0.013309332387871109 | train/value_loss = 56.742698955535886 | train/approx_kl = 0.008423322811722755 | train/clip_fraction = 0.08642578125 | train/loss = 7.123720169067383 | train/explained_variance = 0.013513624668121338 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.41 | rollout/ep_len_mean = 27.41 | time/fps = 106 | time/time_elapsed = 38 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.686777251213789 | train/policy_gradient_loss = -0.013309332387871109 | train/value_loss = 56.742698955535886 | train/approx_kl = 0.008423322811722755 | train/clip_fraction = 0.08642578125 | train/loss = 7.123720169067383 | train/explained_variance = 0.013513624668121338 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 3]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.35 | rollout/ep_len_mean = 25.35 | time/fps = 103 | time/time_elapsed = 39 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.68677708953619 | train/policy_gradient_loss = -0.011988813986681635 | train/value_loss = 51.26607643663883 | train/approx_kl = 0.00829731859266758 | train/clip_fraction = 0.085205078125 | train/loss = 7.120436191558838 | train/explained_variance = 0.0006332993507385254 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.35 | rollout/ep_len_mean = 25.35 | time/fps = 103 | time/time_elapsed = 39 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.68677708953619 | train/policy_gradient_loss = -0.011988813986681635 | train/value_loss = 51.26607643663883 | train/approx_kl = 0.00829731859266758 | train/clip_fraction = 0.085205078125 | train/loss = 7.120436191558838 | train/explained_variance = 0.0006332993507385254 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 1]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 35.66 | rollout/ep_len_mean = 35.66 | time/fps = 96 | time/time_elapsed = 63 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6660862566903234 | train/policy_gradient_loss = -0.014720242060138843 | train/value_loss = 39.50259827375412 | train/approx_kl = 0.00878126360476017 | train/clip_fraction = 0.04921875 | train/loss = 17.623849868774414 | train/explained_variance = 0.11052852869033813 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 35.66 | rollout/ep_len_mean = 35.66 | time/fps = 96 | time/time_elapsed = 63 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6660862566903234 | train/policy_gradient_loss = -0.014720242060138843 | train/value_loss = 39.50259827375412 | train/approx_kl = 0.00878126360476017 | train/clip_fraction = 0.04921875 | train/loss = 17.623849868774414 | train/explained_variance = 0.11052852869033813 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 2]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.47 | rollout/ep_len_mean = 33.47 | time/fps = 96 | time/time_elapsed = 63 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6659716766327619 | train/policy_gradient_loss = -0.02290084579726681 | train/value_loss = 31.37706240415573 | train/approx_kl = 0.010679353959858418 | train/clip_fraction = 0.089111328125 | train/loss = 10.447492599487305 | train/explained_variance = 0.14599686861038208 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.47 | rollout/ep_len_mean = 33.47 | time/fps = 96 | time/time_elapsed = 63 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6659716766327619 | train/policy_gradient_loss = -0.02290084579726681 | train/value_loss = 31.37706240415573 | train/approx_kl = 0.010679353959858418 | train/clip_fraction = 0.089111328125 | train/loss = 10.447492599487305 | train/explained_variance = 0.14599686861038208 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 0]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 37.16 | rollout/ep_len_mean = 37.16 | time/fps = 96 | time/time_elapsed = 63 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6691582711413503 | train/policy_gradient_loss = -0.014172516761755105 | train/value_loss = 38.93319685459137 | train/approx_kl = 0.00811317190527916 | train/clip_fraction = 0.042626953125 | train/loss = 11.936966896057129 | train/explained_variance = 0.06687164306640625 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 37.16 | rollout/ep_len_mean = 37.16 | time/fps = 96 | time/time_elapsed = 63 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6691582711413503 | train/policy_gradient_loss = -0.014172516761755105 | train/value_loss = 38.93319685459137 | train/approx_kl = 0.00811317190527916 | train/clip_fraction = 0.042626953125 | train/loss = 11.936966896057129 | train/explained_variance = 0.06687164306640625 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 4]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.31 | rollout/ep_len_mean = 33.31 | time/fps = 95 | time/time_elapsed = 64 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6667774656787515 | train/policy_gradient_loss = -0.020987652488111054 | train/value_loss = 35.195351189374925 | train/approx_kl = 0.013082670047879219 | train/clip_fraction = 0.08251953125 | train/loss = 15.075634002685547 | train/explained_variance = 0.05656266212463379 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.31 | rollout/ep_len_mean = 33.31 | time/fps = 95 | time/time_elapsed = 64 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6667774656787515 | train/policy_gradient_loss = -0.020987652488111054 | train/value_loss = 35.195351189374925 | train/approx_kl = 0.013082670047879219 | train/clip_fraction = 0.08251953125 | train/loss = 15.075634002685547 | train/explained_variance = 0.05656266212463379 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 3]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 30.73 | rollout/ep_len_mean = 30.73 | time/fps = 92 | time/time_elapsed = 66 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6701155433431267 | train/policy_gradient_loss = -0.013805805651645641 | train/value_loss = 33.60014774501324 | train/approx_kl = 0.009859509766101837 | train/clip_fraction = 0.052587890625 | train/loss = 10.620658874511719 | train/explained_variance = 0.08080285787582397 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 30.73 | rollout/ep_len_mean = 30.73 | time/fps = 92 | time/time_elapsed = 66 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6701155433431267 | train/policy_gradient_loss = -0.013805805651645641 | train/value_loss = 33.60014774501324 | train/approx_kl = 0.009859509766101837 | train/clip_fraction = 0.052587890625 | train/loss = 10.620658874511719 | train/explained_variance = 0.08080285787582397 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 46.64 | rollout/ep_len_mean = 46.64 | time/fps = 92 | time/time_elapsed = 88 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6431730126962065 | train/policy_gradient_loss = -0.017830393892654683 | train/value_loss = 55.77634754776955 | train/approx_kl = 0.007178150117397308 | train/clip_fraction = 0.070458984375 | train/loss = 26.69325828552246 | train/explained_variance = 0.23610299825668335 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 46.64 | rollout/ep_len_mean = 46.64 | time/fps = 92 | time/time_elapsed = 88 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6431730126962065 | train/policy_gradient_loss = -0.017830393892654683 | train/value_loss = 55.77634754776955 | train/approx_kl = 0.007178150117397308 | train/clip_fraction = 0.070458984375 | train/loss = 26.69325828552246 | train/explained_variance = 0.23610299825668335 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 2]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 44.5 | rollout/ep_len_mean = 44.5 | time/fps = 92 | time/time_elapsed = 88 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6336015397682786 | train/policy_gradient_loss = -0.018510080594569444 | train/value_loss = 50.2428251683712 | train/approx_kl = 0.008075550198554993 | train/clip_fraction = 0.09765625 | train/loss = 27.94938087463379 | train/explained_variance = 0.24914509057998657 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 44.5 | rollout/ep_len_mean = 44.5 | time/fps = 92 | time/time_elapsed = 88 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6336015397682786 | train/policy_gradient_loss = -0.018510080594569444 | train/value_loss = 50.2428251683712 | train/approx_kl = 0.008075550198554993 | train/clip_fraction = 0.09765625 | train/loss = 27.94938087463379 | train/explained_variance = 0.24914509057998657 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 1]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 48.04 | rollout/ep_len_mean = 48.04 | time/fps = 92 | time/time_elapsed = 88 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6348903484642505 | train/policy_gradient_loss = -0.02147019632975571 | train/value_loss = 49.048201990127566 | train/approx_kl = 0.009575752541422844 | train/clip_fraction = 0.093017578125 | train/loss = 18.63315200805664 | train/explained_variance = 0.24517738819122314 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 48.04 | rollout/ep_len_mean = 48.04 | time/fps = 92 | time/time_elapsed = 88 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6348903484642505 | train/policy_gradient_loss = -0.02147019632975571 | train/value_loss = 49.048201990127566 | train/approx_kl = 0.009575752541422844 | train/clip_fraction = 0.093017578125 | train/loss = 18.63315200805664 | train/explained_variance = 0.24517738819122314 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:13: [PPO[worker: 4]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 43.53 | rollout/ep_len_mean = 43.53 | time/fps = 91 | time/time_elapsed = 89 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6313173599541188 | train/policy_gradient_loss = -0.020431764989916702 | train/value_loss = 49.39499318003654 | train/approx_kl = 0.008223217912018299 | train/clip_fraction = 0.10126953125 | train/loss = 20.21065902709961 | train/explained_variance = 0.2483510971069336 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 43.53 | rollout/ep_len_mean = 43.53 | time/fps = 91 | time/time_elapsed = 89 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6313173599541188 | train/policy_gradient_loss = -0.020431764989916702 | train/value_loss = 49.39499318003654 | train/approx_kl = 0.008223217912018299 | train/clip_fraction = 0.10126953125 | train/loss = 20.21065902709961 | train/explained_variance = 0.2483510971069336 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:14: [PPO[worker: 3]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 41.26 | rollout/ep_len_mean = 41.26 | time/fps = 88 | time/time_elapsed = 92 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6344639118760824 | train/policy_gradient_loss = -0.01873804119968554 | train/value_loss = 50.9981625854969 | train/approx_kl = 0.010008599609136581 | train/clip_fraction = 0.107763671875 | train/loss = 26.194978713989258 | train/explained_variance = 0.2077431082725525 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 41.26 | rollout/ep_len_mean = 41.26 | time/fps = 88 | time/time_elapsed = 92 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6344639118760824 | train/policy_gradient_loss = -0.01873804119968554 | train/value_loss = 50.9981625854969 | train/approx_kl = 0.010008599609136581 | train/clip_fraction = 0.107763671875 | train/loss = 26.194978713989258 | train/explained_variance = 0.2077431082725525 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:14: ... trained! [0m
    INFO:rlberry_logger:... trained!
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:14: Saved ExperimentManager(PPO) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(PPO) using pickle.
    [38;21m[INFO] 15:14: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-09-36_033e4e8e/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-09-36_033e4e8e/manager_obj.pickle'
    [38;21m[INFO] 15:14: Saved ExperimentManager(A2C) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(A2C) using pickle.
    [38;21m[INFO] 15:14: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-09-36_93291f77/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-09-36_93291f77/manager_obj.pickle'
    [38;21m[INFO] 15:14: Saved ExperimentManager(PPO) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(PPO) using pickle.
    [38;21m[INFO] 15:14: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-09-36_033e4e8e/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-09-36_033e4e8e/manager_obj.pickle'
    [38;21m[INFO] 15:14: Evaluating agent 0 [0m
    INFO:rlberry_logger:Evaluating agent 0
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:14: Evaluating agent 1 [0m
    INFO:rlberry_logger:Evaluating agent 1
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:14: Evaluating agent 2 [0m
    INFO:rlberry_logger:Evaluating agent 2
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:14: Evaluating agent 3 [0m
    INFO:rlberry_logger:Evaluating agent 3
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:14: Evaluating agent 4 [0m
    INFO:rlberry_logger:Evaluating agent 4
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:14: Evaluating agent 0 [0m
    INFO:rlberry_logger:Evaluating agent 0
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:15: Evaluating agent 1 [0m
    INFO:rlberry_logger:Evaluating agent 1
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:15: Evaluating agent 2 [0m
    INFO:rlberry_logger:Evaluating agent 2
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:15: Evaluating agent 3 [0m
    INFO:rlberry_logger:Evaluating agent 3
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:15: Evaluating agent 4 [0m
    INFO:rlberry_logger:Evaluating agent 4
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:15: Running ExperimentManager fit() for A2C with n_fit = 5 and max_workers = None. [0m
    INFO:rlberry_logger:Running ExperimentManager fit() for A2C with n_fit = 5 and max_workers = None.


    Step 2


    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:15:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        1           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        1           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:15:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        0           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        0           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:15:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        3           0.001               500 [0m
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        3           0.001               500
    [38;21m[INFO] 15:15:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        4           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        4           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:15:                agent_name  worker  train/learning_rate  max_global_step
                                    A2C        2           0.001               500 [0m
    INFO:rlberry_logger:               agent_name  worker  train/learning_rate  max_global_step
                                    A2C        2           0.001               500
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:15: [A2C[worker: 0]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.06784224510192871 | train/entropy_loss = -0.6597503423690796 | train/policy_loss = -2.0614309310913086 | train/value_loss = 44.950923919677734 | time/iterations = 100 | rollout/ep_rew_mean = 37.92307692307692 | rollout/ep_len_mean = 37.92307692307692 | time/fps = 82 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.06784224510192871 | train/entropy_loss = -0.6597503423690796 | train/policy_loss = -2.0614309310913086 | train/value_loss = 44.950923919677734 | time/iterations = 100 | rollout/ep_rew_mean = 37.92307692307692 | rollout/ep_len_mean = 37.92307692307692 | time/fps = 82 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:15: [A2C[worker: 3]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -1.1011767387390137 | train/entropy_loss = -0.6348804831504822 | train/policy_loss = 1.0325791835784912 | train/value_loss = 10.595748901367188 | time/iterations = 100 | rollout/ep_rew_mean = 29.6875 | rollout/ep_len_mean = 29.6875 | time/fps = 82 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -1.1011767387390137 | train/entropy_loss = -0.6348804831504822 | train/policy_loss = 1.0325791835784912 | train/value_loss = 10.595748901367188 | time/iterations = 100 | rollout/ep_rew_mean = 29.6875 | rollout/ep_len_mean = 29.6875 | time/fps = 82 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:15: [A2C[worker: 1]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.05182439088821411 | train/entropy_loss = -0.693057656288147 | train/policy_loss = 1.6723606586456299 | train/value_loss = 7.4091477394104 | time/iterations = 100 | rollout/ep_rew_mean = 20.375 | rollout/ep_len_mean = 20.375 | time/fps = 84 | time/time_elapsed = 5 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.05182439088821411 | train/entropy_loss = -0.693057656288147 | train/policy_loss = 1.6723606586456299 | train/value_loss = 7.4091477394104 | time/iterations = 100 | rollout/ep_rew_mean = 20.375 | rollout/ep_len_mean = 20.375 | time/fps = 84 | time/time_elapsed = 5 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:15: [A2C[worker: 4]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.502903550863266 | train/entropy_loss = -0.5348852872848511 | train/policy_loss = 2.7273004055023193 | train/value_loss = 10.946118354797363 | time/iterations = 100 | rollout/ep_rew_mean = 36.416666666666664 | rollout/ep_len_mean = 36.416666666666664 | time/fps = 83 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = 0.502903550863266 | train/entropy_loss = -0.5348852872848511 | train/policy_loss = 2.7273004055023193 | train/value_loss = 10.946118354797363 | time/iterations = 100 | rollout/ep_rew_mean = 36.416666666666664 | rollout/ep_len_mean = 36.416666666666664 | time/fps = 83 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:16: [A2C[worker: 2]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.046254873275756836 | train/entropy_loss = -0.6866374611854553 | train/policy_loss = 1.6995302438735962 | train/value_loss = 8.241180419921875 | time/iterations = 100 | rollout/ep_rew_mean = 20.869565217391305 | rollout/ep_len_mean = 20.869565217391305 | time/fps = 80 | time/time_elapsed = 6 | time/total_timesteps = 500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 1000 | train/learning_rate = 0.0007 | train/n_updates = 99 | train/explained_variance = -0.046254873275756836 | train/entropy_loss = -0.6866374611854553 | train/policy_loss = 1.6995302438735962 | train/value_loss = 8.241180419921875 | time/iterations = 100 | rollout/ep_rew_mean = 20.869565217391305 | rollout/ep_len_mean = 20.869565217391305 | time/fps = 80 | time/time_elapsed = 6 | time/total_timesteps = 500 |
    [38;21m[INFO] 15:16: [A2C[worker: 0]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.49341583251953125 | train/entropy_loss = -0.5551060438156128 | train/policy_loss = 1.576623797416687 | train/value_loss = 3.3922958374023438 | time/iterations = 200 | rollout/ep_rew_mean = 39.48 | rollout/ep_len_mean = 39.48 | time/fps = 81 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.49341583251953125 | train/entropy_loss = -0.5551060438156128 | train/policy_loss = 1.576623797416687 | train/value_loss = 3.3922958374023438 | time/iterations = 200 | rollout/ep_rew_mean = 39.48 | rollout/ep_len_mean = 39.48 | time/fps = 81 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:16: [A2C[worker: 3]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.09769594669342041 | train/entropy_loss = -0.6649371981620789 | train/policy_loss = 1.6660985946655273 | train/value_loss = 8.332029342651367 | time/iterations = 200 | rollout/ep_rew_mean = 23.232558139534884 | rollout/ep_len_mean = 23.232558139534884 | time/fps = 80 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.09769594669342041 | train/entropy_loss = -0.6649371981620789 | train/policy_loss = 1.6660985946655273 | train/value_loss = 8.332029342651367 | time/iterations = 200 | rollout/ep_rew_mean = 23.232558139534884 | rollout/ep_len_mean = 23.232558139534884 | time/fps = 80 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:16: [A2C[worker: 1]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.06340277194976807 | train/entropy_loss = -0.6927796602249146 | train/policy_loss = 1.7201499938964844 | train/value_loss = 7.941596984863281 | time/iterations = 200 | rollout/ep_rew_mean = 20.5531914893617 | rollout/ep_len_mean = 20.5531914893617 | time/fps = 80 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.06340277194976807 | train/entropy_loss = -0.6927796602249146 | train/policy_loss = 1.7201499938964844 | train/value_loss = 7.941596984863281 | time/iterations = 200 | rollout/ep_rew_mean = 20.5531914893617 | rollout/ep_len_mean = 20.5531914893617 | time/fps = 80 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:16: [A2C[worker: 4]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.1355920433998108 | train/entropy_loss = -0.6015006303787231 | train/policy_loss = 1.047149658203125 | train/value_loss = 7.302009582519531 | time/iterations = 200 | rollout/ep_rew_mean = 44.5 | rollout/ep_len_mean = 44.5 | time/fps = 79 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = 0.1355920433998108 | train/entropy_loss = -0.6015006303787231 | train/policy_loss = 1.047149658203125 | train/value_loss = 7.302009582519531 | time/iterations = 200 | rollout/ep_rew_mean = 44.5 | rollout/ep_len_mean = 44.5 | time/fps = 79 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:16: [A2C[worker: 2]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.11184144020080566 | train/entropy_loss = -0.6772807836532593 | train/policy_loss = 1.5336723327636719 | train/value_loss = 7.939347743988037 | time/iterations = 200 | rollout/ep_rew_mean = 25.894736842105264 | rollout/ep_len_mean = 25.894736842105264 | time/fps = 78 | time/time_elapsed = 12 | time/total_timesteps = 1000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 1500 | train/learning_rate = 0.0007 | train/n_updates = 199 | train/explained_variance = -0.11184144020080566 | train/entropy_loss = -0.6772807836532593 | train/policy_loss = 1.5336723327636719 | train/value_loss = 7.939347743988037 | time/iterations = 200 | rollout/ep_rew_mean = 25.894736842105264 | rollout/ep_len_mean = 25.894736842105264 | time/fps = 78 | time/time_elapsed = 12 | time/total_timesteps = 1000 |
    [38;21m[INFO] 15:16: [A2C[worker: 0]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.1347963809967041 | train/entropy_loss = -0.6389867067337036 | train/policy_loss = 1.3967787027359009 | train/value_loss = 5.62935209274292 | time/iterations = 300 | rollout/ep_rew_mean = 37.05 | rollout/ep_len_mean = 37.05 | time/fps = 80 | time/time_elapsed = 18 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.1347963809967041 | train/entropy_loss = -0.6389867067337036 | train/policy_loss = 1.3967787027359009 | train/value_loss = 5.62935209274292 | time/iterations = 300 | rollout/ep_rew_mean = 37.05 | rollout/ep_len_mean = 37.05 | time/fps = 80 | time/time_elapsed = 18 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:16: [A2C[worker: 3]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.16611027717590332 | train/entropy_loss = -0.6523440480232239 | train/policy_loss = 1.4206634759902954 | train/value_loss = 8.486452102661133 | time/iterations = 300 | rollout/ep_rew_mean = 21.86764705882353 | rollout/ep_len_mean = 21.86764705882353 | time/fps = 79 | time/time_elapsed = 18 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.16611027717590332 | train/entropy_loss = -0.6523440480232239 | train/policy_loss = 1.4206634759902954 | train/value_loss = 8.486452102661133 | time/iterations = 300 | rollout/ep_rew_mean = 21.86764705882353 | rollout/ep_len_mean = 21.86764705882353 | time/fps = 79 | time/time_elapsed = 18 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:16: [A2C[worker: 1]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.011020779609680176 | train/entropy_loss = -0.6345912218093872 | train/policy_loss = 1.7939178943634033 | train/value_loss = 7.512570381164551 | time/iterations = 300 | rollout/ep_rew_mean = 23.03076923076923 | rollout/ep_len_mean = 23.03076923076923 | time/fps = 78 | time/time_elapsed = 19 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = -0.011020779609680176 | train/entropy_loss = -0.6345912218093872 | train/policy_loss = 1.7939178943634033 | train/value_loss = 7.512570381164551 | time/iterations = 300 | rollout/ep_rew_mean = 23.03076923076923 | rollout/ep_len_mean = 23.03076923076923 | time/fps = 78 | time/time_elapsed = 19 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:16: [A2C[worker: 4]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.11584144830703735 | train/entropy_loss = -0.5201119184494019 | train/policy_loss = 0.6796188354492188 | train/value_loss = 5.750647068023682 | time/iterations = 300 | rollout/ep_rew_mean = 46.0 | rollout/ep_len_mean = 46.0 | time/fps = 78 | time/time_elapsed = 19 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.11584144830703735 | train/entropy_loss = -0.5201119184494019 | train/policy_loss = 0.6796188354492188 | train/value_loss = 5.750647068023682 | time/iterations = 300 | rollout/ep_rew_mean = 46.0 | rollout/ep_len_mean = 46.0 | time/fps = 78 | time/time_elapsed = 19 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:16: [A2C[worker: 2]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.11990886926651001 | train/entropy_loss = -0.6334784030914307 | train/policy_loss = 1.74115788936615 | train/value_loss = 5.352802276611328 | time/iterations = 300 | rollout/ep_rew_mean = 27.69811320754717 | rollout/ep_len_mean = 27.69811320754717 | time/fps = 77 | time/time_elapsed = 19 | time/total_timesteps = 1500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 2000 | train/learning_rate = 0.0007 | train/n_updates = 299 | train/explained_variance = 0.11990886926651001 | train/entropy_loss = -0.6334784030914307 | train/policy_loss = 1.74115788936615 | train/value_loss = 5.352802276611328 | time/iterations = 300 | rollout/ep_rew_mean = 27.69811320754717 | rollout/ep_len_mean = 27.69811320754717 | time/fps = 77 | time/time_elapsed = 19 | time/total_timesteps = 1500 |
    [38;21m[INFO] 15:16: [A2C[worker: 0]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.019140243530273438 | train/entropy_loss = -0.5991054177284241 | train/policy_loss = -13.021446228027344 | train/value_loss = 435.7764587402344 | time/iterations = 400 | rollout/ep_rew_mean = 35.607142857142854 | rollout/ep_len_mean = 35.607142857142854 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.019140243530273438 | train/entropy_loss = -0.5991054177284241 | train/policy_loss = -13.021446228027344 | train/value_loss = 435.7764587402344 | time/iterations = 400 | rollout/ep_rew_mean = 35.607142857142854 | rollout/ep_len_mean = 35.607142857142854 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:16: [A2C[worker: 3]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.06505054235458374 | train/entropy_loss = -0.6531654596328735 | train/policy_loss = 0.9774222373962402 | train/value_loss = 5.593251705169678 | time/iterations = 400 | rollout/ep_rew_mean = 22.770114942528735 | rollout/ep_len_mean = 22.770114942528735 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.06505054235458374 | train/entropy_loss = -0.6531654596328735 | train/policy_loss = 0.9774222373962402 | train/value_loss = 5.593251705169678 | time/iterations = 400 | rollout/ep_rew_mean = 22.770114942528735 | rollout/ep_len_mean = 22.770114942528735 | time/fps = 80 | time/time_elapsed = 24 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:16: [A2C[worker: 1]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.0438808798789978 | train/entropy_loss = -0.6380696296691895 | train/policy_loss = 1.6101993322372437 | train/value_loss = 5.975500583648682 | time/iterations = 400 | rollout/ep_rew_mean = 26.958904109589042 | rollout/ep_len_mean = 26.958904109589042 | time/fps = 79 | time/time_elapsed = 25 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.0438808798789978 | train/entropy_loss = -0.6380696296691895 | train/policy_loss = 1.6101993322372437 | train/value_loss = 5.975500583648682 | time/iterations = 400 | rollout/ep_rew_mean = 26.958904109589042 | rollout/ep_len_mean = 26.958904109589042 | time/fps = 79 | time/time_elapsed = 25 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:16: [A2C[worker: 4]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.020477771759033203 | train/entropy_loss = -0.558156430721283 | train/policy_loss = 0.8952496647834778 | train/value_loss = 5.584263801574707 | time/iterations = 400 | rollout/ep_rew_mean = 49.025 | rollout/ep_len_mean = 49.025 | time/fps = 79 | time/time_elapsed = 25 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = -0.020477771759033203 | train/entropy_loss = -0.558156430721283 | train/policy_loss = 0.8952496647834778 | train/value_loss = 5.584263801574707 | time/iterations = 400 | rollout/ep_rew_mean = 49.025 | rollout/ep_len_mean = 49.025 | time/fps = 79 | time/time_elapsed = 25 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:16: [A2C[worker: 2]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.0042961835861206055 | train/entropy_loss = -0.6797277331352234 | train/policy_loss = 1.3575818538665771 | train/value_loss = 5.959181785583496 | time/iterations = 400 | rollout/ep_rew_mean = 30.53846153846154 | rollout/ep_len_mean = 30.53846153846154 | time/fps = 78 | time/time_elapsed = 25 | time/total_timesteps = 2000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 2500 | train/learning_rate = 0.0007 | train/n_updates = 399 | train/explained_variance = 0.0042961835861206055 | train/entropy_loss = -0.6797277331352234 | train/policy_loss = 1.3575818538665771 | train/value_loss = 5.959181785583496 | time/iterations = 400 | rollout/ep_rew_mean = 30.53846153846154 | rollout/ep_len_mean = 30.53846153846154 | time/fps = 78 | time/time_elapsed = 25 | time/total_timesteps = 2000 |
    [38;21m[INFO] 15:16: [A2C[worker: 0]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.014640212059020996 | train/entropy_loss = -0.30123570561408997 | train/policy_loss = 1.4919379949569702 | train/value_loss = 5.562033653259277 | time/iterations = 500 | rollout/ep_rew_mean = 34.34722222222222 | rollout/ep_len_mean = 34.34722222222222 | time/fps = 80 | time/time_elapsed = 31 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.014640212059020996 | train/entropy_loss = -0.30123570561408997 | train/policy_loss = 1.4919379949569702 | train/value_loss = 5.562033653259277 | time/iterations = 500 | rollout/ep_rew_mean = 34.34722222222222 | rollout/ep_len_mean = 34.34722222222222 | time/fps = 80 | time/time_elapsed = 31 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:16: [A2C[worker: 1]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.0020249485969543457 | train/entropy_loss = -0.6286938190460205 | train/policy_loss = 0.9726060628890991 | train/value_loss = 5.488160133361816 | time/iterations = 500 | rollout/ep_rew_mean = 29.070588235294117 | rollout/ep_len_mean = 29.070588235294117 | time/fps = 79 | time/time_elapsed = 31 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.0020249485969543457 | train/entropy_loss = -0.6286938190460205 | train/policy_loss = 0.9726060628890991 | train/value_loss = 5.488160133361816 | time/iterations = 500 | rollout/ep_rew_mean = 29.070588235294117 | rollout/ep_len_mean = 29.070588235294117 | time/fps = 79 | time/time_elapsed = 31 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:16: [A2C[worker: 3]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.03558415174484253 | train/entropy_loss = -0.6505569219589233 | train/policy_loss = 0.970024585723877 | train/value_loss = 5.605032920837402 | time/iterations = 500 | rollout/ep_rew_mean = 21.43 | rollout/ep_len_mean = 21.43 | time/fps = 80 | time/time_elapsed = 31 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.03558415174484253 | train/entropy_loss = -0.6505569219589233 | train/policy_loss = 0.970024585723877 | train/value_loss = 5.605032920837402 | time/iterations = 500 | rollout/ep_rew_mean = 21.43 | rollout/ep_len_mean = 21.43 | time/fps = 80 | time/time_elapsed = 31 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:16: [A2C[worker: 4]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.010056376457214355 | train/entropy_loss = -0.41378074884414673 | train/policy_loss = 1.1489373445510864 | train/value_loss = 5.21596622467041 | time/iterations = 500 | rollout/ep_rew_mean = 51.791666666666664 | rollout/ep_len_mean = 51.791666666666664 | time/fps = 79 | time/time_elapsed = 31 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = -0.010056376457214355 | train/entropy_loss = -0.41378074884414673 | train/policy_loss = 1.1489373445510864 | train/value_loss = 5.21596622467041 | time/iterations = 500 | rollout/ep_rew_mean = 51.791666666666664 | rollout/ep_len_mean = 51.791666666666664 | time/fps = 79 | time/time_elapsed = 31 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:16: [A2C[worker: 2]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.03843444585800171 | train/entropy_loss = -0.5682123303413391 | train/policy_loss = 1.8577531576156616 | train/value_loss = 5.404941082000732 | time/iterations = 500 | rollout/ep_rew_mean = 32.89333333333333 | rollout/ep_len_mean = 32.89333333333333 | time/fps = 78 | time/time_elapsed = 31 | time/total_timesteps = 2500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 3000 | train/learning_rate = 0.0007 | train/n_updates = 499 | train/explained_variance = 0.03843444585800171 | train/entropy_loss = -0.5682123303413391 | train/policy_loss = 1.8577531576156616 | train/value_loss = 5.404941082000732 | time/iterations = 500 | rollout/ep_rew_mean = 32.89333333333333 | rollout/ep_len_mean = 32.89333333333333 | time/fps = 78 | time/time_elapsed = 31 | time/total_timesteps = 2500 |
    [38;21m[INFO] 15:16: [A2C[worker: 3]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0011844635009765625 | train/entropy_loss = -0.6341766715049744 | train/policy_loss = 1.1918209791183472 | train/value_loss = 5.433868408203125 | time/iterations = 600 | rollout/ep_rew_mean = 21.95 | rollout/ep_len_mean = 21.95 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0011844635009765625 | train/entropy_loss = -0.6341766715049744 | train/policy_loss = 1.1918209791183472 | train/value_loss = 5.433868408203125 | time/iterations = 600 | rollout/ep_rew_mean = 21.95 | rollout/ep_len_mean = 21.95 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:16: [A2C[worker: 1]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0062056779861450195 | train/entropy_loss = -0.6601538062095642 | train/policy_loss = 1.169372797012329 | train/value_loss = 4.940157890319824 | time/iterations = 600 | rollout/ep_rew_mean = 31.86021505376344 | rollout/ep_len_mean = 31.86021505376344 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.0062056779861450195 | train/entropy_loss = -0.6601538062095642 | train/policy_loss = 1.169372797012329 | train/value_loss = 4.940157890319824 | time/iterations = 600 | rollout/ep_rew_mean = 31.86021505376344 | rollout/ep_len_mean = 31.86021505376344 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:16: [A2C[worker: 0]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.011767446994781494 | train/entropy_loss = -0.640289306640625 | train/policy_loss = 1.0220626592636108 | train/value_loss = 4.897432327270508 | time/iterations = 600 | rollout/ep_rew_mean = 35.476190476190474 | rollout/ep_len_mean = 35.476190476190474 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = 0.011767446994781494 | train/entropy_loss = -0.640289306640625 | train/policy_loss = 1.0220626592636108 | train/value_loss = 4.897432327270508 | time/iterations = 600 | rollout/ep_rew_mean = 35.476190476190474 | rollout/ep_len_mean = 35.476190476190474 | time/fps = 80 | time/time_elapsed = 37 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:16: [A2C[worker: 4]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.005561232566833496 | train/entropy_loss = -0.4747939109802246 | train/policy_loss = 1.0447986125946045 | train/value_loss = 4.601233005523682 | time/iterations = 600 | rollout/ep_rew_mean = 54.018181818181816 | rollout/ep_len_mean = 54.018181818181816 | time/fps = 79 | time/time_elapsed = 37 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.005561232566833496 | train/entropy_loss = -0.4747939109802246 | train/policy_loss = 1.0447986125946045 | train/value_loss = 4.601233005523682 | time/iterations = 600 | rollout/ep_rew_mean = 54.018181818181816 | rollout/ep_len_mean = 54.018181818181816 | time/fps = 79 | time/time_elapsed = 37 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:16: [A2C[worker: 2]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.006732344627380371 | train/entropy_loss = -0.6716772317886353 | train/policy_loss = 1.05986750125885 | train/value_loss = 4.84399938583374 | time/iterations = 600 | rollout/ep_rew_mean = 34.848837209302324 | rollout/ep_len_mean = 34.848837209302324 | time/fps = 78 | time/time_elapsed = 38 | time/total_timesteps = 3000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 3500 | train/learning_rate = 0.0007 | train/n_updates = 599 | train/explained_variance = -0.006732344627380371 | train/entropy_loss = -0.6716772317886353 | train/policy_loss = 1.05986750125885 | train/value_loss = 4.84399938583374 | time/iterations = 600 | rollout/ep_rew_mean = 34.848837209302324 | rollout/ep_len_mean = 34.848837209302324 | time/fps = 78 | time/time_elapsed = 38 | time/total_timesteps = 3000 |
    [38;21m[INFO] 15:16: [A2C[worker: 3]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.023659467697143555 | train/entropy_loss = -0.6115827560424805 | train/policy_loss = 1.0988785028457642 | train/value_loss = 5.221302509307861 | time/iterations = 700 | rollout/ep_rew_mean = 23.99 | rollout/ep_len_mean = 23.99 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.023659467697143555 | train/entropy_loss = -0.6115827560424805 | train/policy_loss = 1.0988785028457642 | train/value_loss = 5.221302509307861 | time/iterations = 700 | rollout/ep_rew_mean = 23.99 | rollout/ep_len_mean = 23.99 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:16: [A2C[worker: 1]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.0065479278564453125 | train/entropy_loss = -0.6045333743095398 | train/policy_loss = 1.1206281185150146 | train/value_loss = 4.3558735847473145 | time/iterations = 700 | rollout/ep_rew_mean = 34.61224489795919 | rollout/ep_len_mean = 34.61224489795919 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.0065479278564453125 | train/entropy_loss = -0.6045333743095398 | train/policy_loss = 1.1206281185150146 | train/value_loss = 4.3558735847473145 | time/iterations = 700 | rollout/ep_rew_mean = 34.61224489795919 | rollout/ep_len_mean = 34.61224489795919 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:16: [A2C[worker: 0]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.0032600760459899902 | train/entropy_loss = -0.407878577709198 | train/policy_loss = -11.15784740447998 | train/value_loss = 911.1912231445312 | time/iterations = 700 | rollout/ep_rew_mean = 37.170212765957444 | rollout/ep_len_mean = 37.170212765957444 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = 0.0032600760459899902 | train/entropy_loss = -0.407878577709198 | train/policy_loss = -11.15784740447998 | train/value_loss = 911.1912231445312 | time/iterations = 700 | rollout/ep_rew_mean = 37.170212765957444 | rollout/ep_len_mean = 37.170212765957444 | time/fps = 80 | time/time_elapsed = 43 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:16: [A2C[worker: 4]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.004914522171020508 | train/entropy_loss = -0.500377357006073 | train/policy_loss = 0.64014732837677 | train/value_loss = 4.049171447753906 | time/iterations = 700 | rollout/ep_rew_mean = 55.333333333333336 | rollout/ep_len_mean = 55.333333333333336 | time/fps = 79 | time/time_elapsed = 44 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.004914522171020508 | train/entropy_loss = -0.500377357006073 | train/policy_loss = 0.64014732837677 | train/value_loss = 4.049171447753906 | time/iterations = 700 | rollout/ep_rew_mean = 55.333333333333336 | rollout/ep_len_mean = 55.333333333333336 | time/fps = 79 | time/time_elapsed = 44 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:16: [A2C[worker: 2]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.003911614418029785 | train/entropy_loss = -0.6171507835388184 | train/policy_loss = 1.269360899925232 | train/value_loss = 4.353346347808838 | time/iterations = 700 | rollout/ep_rew_mean = 36.04255319148936 | rollout/ep_len_mean = 36.04255319148936 | time/fps = 78 | time/time_elapsed = 44 | time/total_timesteps = 3500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 4000 | train/learning_rate = 0.0007 | train/n_updates = 699 | train/explained_variance = -0.003911614418029785 | train/entropy_loss = -0.6171507835388184 | train/policy_loss = 1.269360899925232 | train/value_loss = 4.353346347808838 | time/iterations = 700 | rollout/ep_rew_mean = 36.04255319148936 | rollout/ep_len_mean = 36.04255319148936 | time/fps = 78 | time/time_elapsed = 44 | time/total_timesteps = 3500 |
    [38;21m[INFO] 15:16: [A2C[worker: 3]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.005255281925201416 | train/entropy_loss = -0.6591740250587463 | train/policy_loss = 0.977618396282196 | train/value_loss = 4.7153706550598145 | time/iterations = 800 | rollout/ep_rew_mean = 26.13 | rollout/ep_len_mean = 26.13 | time/fps = 81 | time/time_elapsed = 49 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.005255281925201416 | train/entropy_loss = -0.6591740250587463 | train/policy_loss = 0.977618396282196 | train/value_loss = 4.7153706550598145 | time/iterations = 800 | rollout/ep_rew_mean = 26.13 | rollout/ep_len_mean = 26.13 | time/fps = 81 | time/time_elapsed = 49 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:16: [A2C[worker: 1]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.002500176429748535 | train/entropy_loss = -0.5927675366401672 | train/policy_loss = 0.6388660669326782 | train/value_loss = 3.793933391571045 | time/iterations = 800 | rollout/ep_rew_mean = 38.58 | rollout/ep_len_mean = 38.58 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.002500176429748535 | train/entropy_loss = -0.5927675366401672 | train/policy_loss = 0.6388660669326782 | train/value_loss = 3.793933391571045 | time/iterations = 800 | rollout/ep_rew_mean = 38.58 | rollout/ep_len_mean = 38.58 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:16: [A2C[worker: 0]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0004057884216308594 | train/entropy_loss = -0.6186630129814148 | train/policy_loss = 0.9697709083557129 | train/value_loss = 3.8345935344696045 | time/iterations = 800 | rollout/ep_rew_mean = 39.56 | rollout/ep_len_mean = 39.56 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.0004057884216308594 | train/entropy_loss = -0.6186630129814148 | train/policy_loss = 0.9697709083557129 | train/value_loss = 3.8345935344696045 | time/iterations = 800 | rollout/ep_rew_mean = 39.56 | rollout/ep_len_mean = 39.56 | time/fps = 80 | time/time_elapsed = 49 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:16: [A2C[worker: 4]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.02655017375946045 | train/entropy_loss = -0.286133348941803 | train/policy_loss = 0.7057732343673706 | train/value_loss = 335.67254638671875 | time/iterations = 800 | rollout/ep_rew_mean = 57.01428571428571 | rollout/ep_len_mean = 57.01428571428571 | time/fps = 79 | time/time_elapsed = 50 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = 0.02655017375946045 | train/entropy_loss = -0.286133348941803 | train/policy_loss = 0.7057732343673706 | train/value_loss = 335.67254638671875 | time/iterations = 800 | rollout/ep_rew_mean = 57.01428571428571 | rollout/ep_len_mean = 57.01428571428571 | time/fps = 79 | time/time_elapsed = 50 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:16: [A2C[worker: 2]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.002364516258239746 | train/entropy_loss = -0.6518915891647339 | train/policy_loss = 0.8045291900634766 | train/value_loss = 3.8379406929016113 | time/iterations = 800 | rollout/ep_rew_mean = 39.2 | rollout/ep_len_mean = 39.2 | time/fps = 78 | time/time_elapsed = 51 | time/total_timesteps = 4000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 4500 | train/learning_rate = 0.0007 | train/n_updates = 799 | train/explained_variance = -0.002364516258239746 | train/entropy_loss = -0.6518915891647339 | train/policy_loss = 0.8045291900634766 | train/value_loss = 3.8379406929016113 | time/iterations = 800 | rollout/ep_rew_mean = 39.2 | rollout/ep_len_mean = 39.2 | time/fps = 78 | time/time_elapsed = 51 | time/total_timesteps = 4000 |
    [38;21m[INFO] 15:16: [A2C[worker: 3]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.00021338462829589844 | train/entropy_loss = -0.6807267665863037 | train/policy_loss = 1.1980390548706055 | train/value_loss = 4.307944297790527 | time/iterations = 900 | rollout/ep_rew_mean = 27.48 | rollout/ep_len_mean = 27.48 | time/fps = 80 | time/time_elapsed = 55 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.00021338462829589844 | train/entropy_loss = -0.6807267665863037 | train/policy_loss = 1.1980390548706055 | train/value_loss = 4.307944297790527 | time/iterations = 900 | rollout/ep_rew_mean = 27.48 | rollout/ep_len_mean = 27.48 | time/fps = 80 | time/time_elapsed = 55 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:16: [A2C[worker: 0]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -3.349781036376953e-05 | train/entropy_loss = -0.6518101692199707 | train/policy_loss = 0.7279996871948242 | train/value_loss = 3.294687271118164 | time/iterations = 900 | rollout/ep_rew_mean = 43.22 | rollout/ep_len_mean = 43.22 | time/fps = 80 | time/time_elapsed = 56 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -3.349781036376953e-05 | train/entropy_loss = -0.6518101692199707 | train/policy_loss = 0.7279996871948242 | train/value_loss = 3.294687271118164 | time/iterations = 900 | rollout/ep_rew_mean = 43.22 | rollout/ep_len_mean = 43.22 | time/fps = 80 | time/time_elapsed = 56 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:16: [A2C[worker: 1]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0003885030746459961 | train/entropy_loss = -0.635242760181427 | train/policy_loss = 0.6818586587905884 | train/value_loss = 3.2691192626953125 | time/iterations = 900 | rollout/ep_rew_mean = 44.05 | rollout/ep_len_mean = 44.05 | time/fps = 80 | time/time_elapsed = 55 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0003885030746459961 | train/entropy_loss = -0.635242760181427 | train/policy_loss = 0.6818586587905884 | train/value_loss = 3.2691192626953125 | time/iterations = 900 | rollout/ep_rew_mean = 44.05 | rollout/ep_len_mean = 44.05 | time/fps = 80 | time/time_elapsed = 55 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:16: [A2C[worker: 4]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0010287761688232422 | train/entropy_loss = -0.517530083656311 | train/policy_loss = 0.4329690933227539 | train/value_loss = 3.049527883529663 | time/iterations = 900 | rollout/ep_rew_mean = 60.0945945945946 | rollout/ep_len_mean = 60.0945945945946 | time/fps = 79 | time/time_elapsed = 56 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = -0.0010287761688232422 | train/entropy_loss = -0.517530083656311 | train/policy_loss = 0.4329690933227539 | train/value_loss = 3.049527883529663 | time/iterations = 900 | rollout/ep_rew_mean = 60.0945945945946 | rollout/ep_len_mean = 60.0945945945946 | time/fps = 79 | time/time_elapsed = 56 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:16: [A2C[worker: 2]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.00017142295837402344 | train/entropy_loss = -0.6514989733695984 | train/policy_loss = 1.001589059829712 | train/value_loss = 3.3113765716552734 | time/iterations = 900 | rollout/ep_rew_mean = 42.86 | rollout/ep_len_mean = 42.86 | time/fps = 78 | time/time_elapsed = 57 | time/total_timesteps = 4500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 5000 | train/learning_rate = 0.0007 | train/n_updates = 899 | train/explained_variance = 0.00017142295837402344 | train/entropy_loss = -0.6514989733695984 | train/policy_loss = 1.001589059829712 | train/value_loss = 3.3113765716552734 | time/iterations = 900 | rollout/ep_rew_mean = 42.86 | rollout/ep_len_mean = 42.86 | time/fps = 78 | time/time_elapsed = 57 | time/total_timesteps = 4500 |
    [38;21m[INFO] 15:16: [A2C[worker: 3]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.00041168928146362305 | train/entropy_loss = -0.6310874819755554 | train/policy_loss = 1.3163658380508423 | train/value_loss = 3.847031831741333 | time/iterations = 1000 | rollout/ep_rew_mean = 28.69 | rollout/ep_len_mean = 28.69 | time/fps = 81 | time/time_elapsed = 61 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 0.00041168928146362305 | train/entropy_loss = -0.6310874819755554 | train/policy_loss = 1.3163658380508423 | train/value_loss = 3.847031831741333 | time/iterations = 1000 | rollout/ep_rew_mean = 28.69 | rollout/ep_len_mean = 28.69 | time/fps = 81 | time/time_elapsed = 61 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:16: [A2C[worker: 0]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 2.4497509002685547e-05 | train/entropy_loss = -0.6405766010284424 | train/policy_loss = 0.7778869867324829 | train/value_loss = 2.835361957550049 | time/iterations = 1000 | rollout/ep_rew_mean = 45.61 | rollout/ep_len_mean = 45.61 | time/fps = 80 | time/time_elapsed = 62 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 2.4497509002685547e-05 | train/entropy_loss = -0.6405766010284424 | train/policy_loss = 0.7778869867324829 | train/value_loss = 2.835361957550049 | time/iterations = 1000 | rollout/ep_rew_mean = 45.61 | rollout/ep_len_mean = 45.61 | time/fps = 80 | time/time_elapsed = 62 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:16: [A2C[worker: 1]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.003565549850463867 | train/entropy_loss = -0.5718713402748108 | train/policy_loss = 0.9199210405349731 | train/value_loss = 2.7869532108306885 | time/iterations = 1000 | rollout/ep_rew_mean = 46.49 | rollout/ep_len_mean = 46.49 | time/fps = 80 | time/time_elapsed = 62 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.003565549850463867 | train/entropy_loss = -0.5718713402748108 | train/policy_loss = 0.9199210405349731 | train/value_loss = 2.7869532108306885 | time/iterations = 1000 | rollout/ep_rew_mean = 46.49 | rollout/ep_len_mean = 46.49 | time/fps = 80 | time/time_elapsed = 62 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:16: [A2C[worker: 4]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 3.993511199951172e-06 | train/entropy_loss = -0.48968857526779175 | train/policy_loss = 0.6242733597755432 | train/value_loss = 2.591146945953369 | time/iterations = 1000 | rollout/ep_rew_mean = 63.32051282051282 | rollout/ep_len_mean = 63.32051282051282 | time/fps = 79 | time/time_elapsed = 62 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = 3.993511199951172e-06 | train/entropy_loss = -0.48968857526779175 | train/policy_loss = 0.6242733597755432 | train/value_loss = 2.591146945953369 | time/iterations = 1000 | rollout/ep_rew_mean = 63.32051282051282 | rollout/ep_len_mean = 63.32051282051282 | time/fps = 79 | time/time_elapsed = 62 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:16: [A2C[worker: 2]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0011034011840820312 | train/entropy_loss = -0.6238595247268677 | train/policy_loss = 0.646612823009491 | train/value_loss = 2.8261396884918213 | time/iterations = 1000 | rollout/ep_rew_mean = 46.71 | rollout/ep_len_mean = 46.71 | time/fps = 78 | time/time_elapsed = 63 | time/total_timesteps = 5000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 5500 | train/learning_rate = 0.0007 | train/n_updates = 999 | train/explained_variance = -0.0011034011840820312 | train/entropy_loss = -0.6238595247268677 | train/policy_loss = 0.646612823009491 | train/value_loss = 2.8261396884918213 | time/iterations = 1000 | rollout/ep_rew_mean = 46.71 | rollout/ep_len_mean = 46.71 | time/fps = 78 | time/time_elapsed = 63 | time/total_timesteps = 5000 |
    [38;21m[INFO] 15:17: [A2C[worker: 3]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0003777742385864258 | train/entropy_loss = -0.5930032730102539 | train/policy_loss = 1.4703848361968994 | train/value_loss = 3.422637462615967 | time/iterations = 1100 | rollout/ep_rew_mean = 31.02 | rollout/ep_len_mean = 31.02 | time/fps = 81 | time/time_elapsed = 67 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = -0.0003777742385864258 | train/entropy_loss = -0.5930032730102539 | train/policy_loss = 1.4703848361968994 | train/value_loss = 3.422637462615967 | time/iterations = 1100 | rollout/ep_rew_mean = 31.02 | rollout/ep_len_mean = 31.02 | time/fps = 81 | time/time_elapsed = 67 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:17: [A2C[worker: 0]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0019857287406921387 | train/entropy_loss = -0.5514726042747498 | train/policy_loss = 1.0775930881500244 | train/value_loss = 2.3963875770568848 | time/iterations = 1100 | rollout/ep_rew_mean = 48.63 | rollout/ep_len_mean = 48.63 | time/fps = 80 | time/time_elapsed = 68 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0019857287406921387 | train/entropy_loss = -0.5514726042747498 | train/policy_loss = 1.0775930881500244 | train/value_loss = 2.3963875770568848 | time/iterations = 1100 | rollout/ep_rew_mean = 48.63 | rollout/ep_len_mean = 48.63 | time/fps = 80 | time/time_elapsed = 68 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:17: [A2C[worker: 1]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0004544854164123535 | train/entropy_loss = -0.6508697271347046 | train/policy_loss = 0.6828605532646179 | train/value_loss = 2.3322129249572754 | time/iterations = 1100 | rollout/ep_rew_mean = 51.27 | rollout/ep_len_mean = 51.27 | time/fps = 80 | time/time_elapsed = 68 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0004544854164123535 | train/entropy_loss = -0.6508697271347046 | train/policy_loss = 0.6828605532646179 | train/value_loss = 2.3322129249572754 | time/iterations = 1100 | rollout/ep_rew_mean = 51.27 | rollout/ep_len_mean = 51.27 | time/fps = 80 | time/time_elapsed = 68 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:17: [A2C[worker: 4]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0007677674293518066 | train/entropy_loss = -0.418527752161026 | train/policy_loss = 0.6401313543319702 | train/value_loss = 2.1692490577697754 | time/iterations = 1100 | rollout/ep_rew_mean = 66.86585365853658 | rollout/ep_len_mean = 66.86585365853658 | time/fps = 79 | time/time_elapsed = 69 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.0007677674293518066 | train/entropy_loss = -0.418527752161026 | train/policy_loss = 0.6401313543319702 | train/value_loss = 2.1692490577697754 | time/iterations = 1100 | rollout/ep_rew_mean = 66.86585365853658 | rollout/ep_len_mean = 66.86585365853658 | time/fps = 79 | time/time_elapsed = 69 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:17: [A2C[worker: 2]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.006498098373413086 | train/entropy_loss = -0.5983108282089233 | train/policy_loss = 1.0919406414031982 | train/value_loss = 2.358607769012451 | time/iterations = 1100 | rollout/ep_rew_mean = 52.59 | rollout/ep_len_mean = 52.59 | time/fps = 78 | time/time_elapsed = 70 | time/total_timesteps = 5500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 6000 | train/learning_rate = 0.0007 | train/n_updates = 1099 | train/explained_variance = 0.006498098373413086 | train/entropy_loss = -0.5983108282089233 | train/policy_loss = 1.0919406414031982 | train/value_loss = 2.358607769012451 | time/iterations = 1100 | rollout/ep_rew_mean = 52.59 | rollout/ep_len_mean = 52.59 | time/fps = 78 | time/time_elapsed = 70 | time/total_timesteps = 5500 |
    [38;21m[INFO] 15:17: [A2C[worker: 3]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00014019012451171875 | train/entropy_loss = -0.6745640635490417 | train/policy_loss = 0.8627058863639832 | train/value_loss = 2.9778554439544678 | time/iterations = 1200 | rollout/ep_rew_mean = 33.66 | rollout/ep_len_mean = 33.66 | time/fps = 81 | time/time_elapsed = 73 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.00014019012451171875 | train/entropy_loss = -0.6745640635490417 | train/policy_loss = 0.8627058863639832 | train/value_loss = 2.9778554439544678 | time/iterations = 1200 | rollout/ep_rew_mean = 33.66 | rollout/ep_len_mean = 33.66 | time/fps = 81 | time/time_elapsed = 73 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:17: [A2C[worker: 0]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.0003578066825866699 | train/entropy_loss = -0.5862897634506226 | train/policy_loss = 0.5465154647827148 | train/value_loss = 1.99076247215271 | time/iterations = 1200 | rollout/ep_rew_mean = 52.56 | rollout/ep_len_mean = 52.56 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.0003578066825866699 | train/entropy_loss = -0.5862897634506226 | train/policy_loss = 0.5465154647827148 | train/value_loss = 1.99076247215271 | time/iterations = 1200 | rollout/ep_rew_mean = 52.56 | rollout/ep_len_mean = 52.56 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:17: [A2C[worker: 1]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.0001087188720703125 | train/entropy_loss = -0.6183658838272095 | train/policy_loss = 0.7108604311943054 | train/value_loss = 1.9346809387207031 | time/iterations = 1200 | rollout/ep_rew_mean = 56.46 | rollout/ep_len_mean = 56.46 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.0001087188720703125 | train/entropy_loss = -0.6183658838272095 | train/policy_loss = 0.7108604311943054 | train/value_loss = 1.9346809387207031 | time/iterations = 1200 | rollout/ep_rew_mean = 56.46 | rollout/ep_len_mean = 56.46 | time/fps = 80 | time/time_elapsed = 74 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:17: [A2C[worker: 4]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.0004311800003051758 | train/entropy_loss = -0.4564032554626465 | train/policy_loss = 0.9832274317741394 | train/value_loss = 1.7684211730957031 | time/iterations = 1200 | rollout/ep_rew_mean = 69.42857142857143 | rollout/ep_len_mean = 69.42857142857143 | time/fps = 79 | time/time_elapsed = 75 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = -0.0004311800003051758 | train/entropy_loss = -0.4564032554626465 | train/policy_loss = 0.9832274317741394 | train/value_loss = 1.7684211730957031 | time/iterations = 1200 | rollout/ep_rew_mean = 69.42857142857143 | rollout/ep_len_mean = 69.42857142857143 | time/fps = 79 | time/time_elapsed = 75 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:17: [A2C[worker: 2]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.0006693601608276367 | train/entropy_loss = -0.6734234094619751 | train/policy_loss = 0.6922272443771362 | train/value_loss = 1.9435784816741943 | time/iterations = 1200 | rollout/ep_rew_mean = 55.67 | rollout/ep_len_mean = 55.67 | time/fps = 78 | time/time_elapsed = 76 | time/total_timesteps = 6000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 6500 | train/learning_rate = 0.0007 | train/n_updates = 1199 | train/explained_variance = 0.0006693601608276367 | train/entropy_loss = -0.6734234094619751 | train/policy_loss = 0.6922272443771362 | train/value_loss = 1.9435784816741943 | time/iterations = 1200 | rollout/ep_rew_mean = 55.67 | rollout/ep_len_mean = 55.67 | time/fps = 78 | time/time_elapsed = 76 | time/total_timesteps = 6000 |
    [38;21m[INFO] 15:17: [A2C[worker: 3]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0008471012115478516 | train/entropy_loss = -0.6321882009506226 | train/policy_loss = 1.0381824970245361 | train/value_loss = 2.590242624282837 | time/iterations = 1300 | rollout/ep_rew_mean = 36.58 | rollout/ep_len_mean = 36.58 | time/fps = 81 | time/time_elapsed = 79 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0008471012115478516 | train/entropy_loss = -0.6321882009506226 | train/policy_loss = 1.0381824970245361 | train/value_loss = 2.590242624282837 | time/iterations = 1300 | rollout/ep_rew_mean = 36.58 | rollout/ep_len_mean = 36.58 | time/fps = 81 | time/time_elapsed = 79 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:17: [A2C[worker: 0]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0005394220352172852 | train/entropy_loss = -0.4100862443447113 | train/policy_loss = 0.8114737272262573 | train/value_loss = 1.6324317455291748 | time/iterations = 1300 | rollout/ep_rew_mean = 55.17 | rollout/ep_len_mean = 55.17 | time/fps = 80 | time/time_elapsed = 80 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -0.0005394220352172852 | train/entropy_loss = -0.4100862443447113 | train/policy_loss = 0.8114737272262573 | train/value_loss = 1.6324317455291748 | time/iterations = 1300 | rollout/ep_rew_mean = 55.17 | rollout/ep_len_mean = 55.17 | time/fps = 80 | time/time_elapsed = 80 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:17: [A2C[worker: 1]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.0008882880210876465 | train/entropy_loss = -0.6421209573745728 | train/policy_loss = 0.6648103594779968 | train/value_loss = 1.5841766595840454 | time/iterations = 1300 | rollout/ep_rew_mean = 59.45 | rollout/ep_len_mean = 59.45 | time/fps = 80 | time/time_elapsed = 81 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = 0.0008882880210876465 | train/entropy_loss = -0.6421209573745728 | train/policy_loss = 0.6648103594779968 | train/value_loss = 1.5841766595840454 | time/iterations = 1300 | rollout/ep_rew_mean = 59.45 | rollout/ep_len_mean = 59.45 | time/fps = 80 | time/time_elapsed = 81 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:17: [A2C[worker: 4]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -1.4781951904296875e-05 | train/entropy_loss = -0.5300630927085876 | train/policy_loss = -4.023325443267822 | train/value_loss = 783.605712890625 | time/iterations = 1300 | rollout/ep_rew_mean = 73.76136363636364 | rollout/ep_len_mean = 73.76136363636364 | time/fps = 79 | time/time_elapsed = 81 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -1.4781951904296875e-05 | train/entropy_loss = -0.5300630927085876 | train/policy_loss = -4.023325443267822 | train/value_loss = 783.605712890625 | time/iterations = 1300 | rollout/ep_rew_mean = 73.76136363636364 | rollout/ep_len_mean = 73.76136363636364 | time/fps = 79 | time/time_elapsed = 81 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:17: [A2C[worker: 2]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -6.747245788574219e-05 | train/entropy_loss = -0.5948551297187805 | train/policy_loss = 0.6244559288024902 | train/value_loss = 1.5669995546340942 | time/iterations = 1300 | rollout/ep_rew_mean = 61.61 | rollout/ep_len_mean = 61.61 | time/fps = 78 | time/time_elapsed = 82 | time/total_timesteps = 6500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 7000 | train/learning_rate = 0.0007 | train/n_updates = 1299 | train/explained_variance = -6.747245788574219e-05 | train/entropy_loss = -0.5948551297187805 | train/policy_loss = 0.6244559288024902 | train/value_loss = 1.5669995546340942 | time/iterations = 1300 | rollout/ep_rew_mean = 61.61 | rollout/ep_len_mean = 61.61 | time/fps = 78 | time/time_elapsed = 82 | time/total_timesteps = 6500 |
    [38;21m[INFO] 15:17: [A2C[worker: 3]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.0002607107162475586 | train/entropy_loss = -0.6787856221199036 | train/policy_loss = 0.8579305410385132 | train/value_loss = 2.239295244216919 | time/iterations = 1400 | rollout/ep_rew_mean = 38.16 | rollout/ep_len_mean = 38.16 | time/fps = 81 | time/time_elapsed = 85 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 0.0002607107162475586 | train/entropy_loss = -0.6787856221199036 | train/policy_loss = 0.8579305410385132 | train/value_loss = 2.239295244216919 | time/iterations = 1400 | rollout/ep_rew_mean = 38.16 | rollout/ep_len_mean = 38.16 | time/fps = 81 | time/time_elapsed = 85 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:17: [A2C[worker: 0]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.00017976760864257812 | train/entropy_loss = -0.5220672488212585 | train/policy_loss = 0.5261040329933167 | train/value_loss = 1.2944307327270508 | time/iterations = 1400 | rollout/ep_rew_mean = 58.05 | rollout/ep_len_mean = 58.05 | time/fps = 80 | time/time_elapsed = 86 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -0.00017976760864257812 | train/entropy_loss = -0.5220672488212585 | train/policy_loss = 0.5261040329933167 | train/value_loss = 1.2944307327270508 | time/iterations = 1400 | rollout/ep_rew_mean = 58.05 | rollout/ep_len_mean = 58.05 | time/fps = 80 | time/time_elapsed = 86 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:17: [A2C[worker: 1]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -8.213520050048828e-05 | train/entropy_loss = -0.5474985241889954 | train/policy_loss = 0.44648393988609314 | train/value_loss = 1.2603166103363037 | time/iterations = 1400 | rollout/ep_rew_mean = 63.82 | rollout/ep_len_mean = 63.82 | time/fps = 80 | time/time_elapsed = 87 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -8.213520050048828e-05 | train/entropy_loss = -0.5474985241889954 | train/policy_loss = 0.44648393988609314 | train/value_loss = 1.2603166103363037 | time/iterations = 1400 | rollout/ep_rew_mean = 63.82 | rollout/ep_len_mean = 63.82 | time/fps = 80 | time/time_elapsed = 87 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:17: [A2C[worker: 4]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -6.079673767089844e-06 | train/entropy_loss = -0.5530027151107788 | train/policy_loss = -21.94296646118164 | train/value_loss = 3411.021484375 | time/iterations = 1400 | rollout/ep_rew_mean = 76.02173913043478 | rollout/ep_len_mean = 76.02173913043478 | time/fps = 79 | time/time_elapsed = 87 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = -6.079673767089844e-06 | train/entropy_loss = -0.5530027151107788 | train/policy_loss = -21.94296646118164 | train/value_loss = 3411.021484375 | time/iterations = 1400 | rollout/ep_rew_mean = 76.02173913043478 | rollout/ep_len_mean = 76.02173913043478 | time/fps = 79 | time/time_elapsed = 87 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:17: [A2C[worker: 2]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 2.753734588623047e-05 | train/entropy_loss = -0.5050080418586731 | train/policy_loss = 0.6447681784629822 | train/value_loss = 1.2228729724884033 | time/iterations = 1400 | rollout/ep_rew_mean = 64.46 | rollout/ep_len_mean = 64.46 | time/fps = 79 | time/time_elapsed = 88 | time/total_timesteps = 7000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 7500 | train/learning_rate = 0.0007 | train/n_updates = 1399 | train/explained_variance = 2.753734588623047e-05 | train/entropy_loss = -0.5050080418586731 | train/policy_loss = 0.6447681784629822 | train/value_loss = 1.2228729724884033 | time/iterations = 1400 | rollout/ep_rew_mean = 64.46 | rollout/ep_len_mean = 64.46 | time/fps = 79 | time/time_elapsed = 88 | time/total_timesteps = 7000 |
    [38;21m[INFO] 15:17: [A2C[worker: 3]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.00011909008026123047 | train/entropy_loss = -0.6850236058235168 | train/policy_loss = 0.7489181756973267 | train/value_loss = 1.9215589761734009 | time/iterations = 1500 | rollout/ep_rew_mean = 39.62 | rollout/ep_len_mean = 39.62 | time/fps = 81 | time/time_elapsed = 92 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.00011909008026123047 | train/entropy_loss = -0.6850236058235168 | train/policy_loss = 0.7489181756973267 | train/value_loss = 1.9215589761734009 | time/iterations = 1500 | rollout/ep_rew_mean = 39.62 | rollout/ep_len_mean = 39.62 | time/fps = 81 | time/time_elapsed = 92 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:17: [A2C[worker: 0]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.00024646520614624023 | train/entropy_loss = -0.4866068363189697 | train/policy_loss = 0.7619408965110779 | train/value_loss = 1.0086603164672852 | time/iterations = 1500 | rollout/ep_rew_mean = 62.88 | rollout/ep_len_mean = 62.88 | time/fps = 80 | time/time_elapsed = 92 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.00024646520614624023 | train/entropy_loss = -0.4866068363189697 | train/policy_loss = 0.7619408965110779 | train/value_loss = 1.0086603164672852 | time/iterations = 1500 | rollout/ep_rew_mean = 62.88 | rollout/ep_len_mean = 62.88 | time/fps = 80 | time/time_elapsed = 92 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:17: [A2C[worker: 1]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0003293752670288086 | train/entropy_loss = -0.4757184088230133 | train/policy_loss = 0.3753967881202698 | train/value_loss = 0.9611436724662781 | time/iterations = 1500 | rollout/ep_rew_mean = 66.15 | rollout/ep_len_mean = 66.15 | time/fps = 80 | time/time_elapsed = 93 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = -0.0003293752670288086 | train/entropy_loss = -0.4757184088230133 | train/policy_loss = 0.3753967881202698 | train/value_loss = 0.9611436724662781 | time/iterations = 1500 | rollout/ep_rew_mean = 66.15 | rollout/ep_len_mean = 66.15 | time/fps = 80 | time/time_elapsed = 93 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:17: [A2C[worker: 4]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.00598984956741333 | train/entropy_loss = -0.4663974344730377 | train/policy_loss = 0.4098740220069885 | train/value_loss = 0.8298931121826172 | time/iterations = 1500 | rollout/ep_rew_mean = 76.87234042553192 | rollout/ep_len_mean = 76.87234042553192 | time/fps = 79 | time/time_elapsed = 93 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 0.00598984956741333 | train/entropy_loss = -0.4663974344730377 | train/policy_loss = 0.4098740220069885 | train/value_loss = 0.8298931121826172 | time/iterations = 1500 | rollout/ep_rew_mean = 76.87234042553192 | rollout/ep_len_mean = 76.87234042553192 | time/fps = 79 | time/time_elapsed = 93 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:17: [A2C[worker: 2]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 3.0994415283203125e-05 | train/entropy_loss = -0.3397093415260315 | train/policy_loss = 0.7903553247451782 | train/value_loss = 0.9350654482841492 | time/iterations = 1500 | rollout/ep_rew_mean = 70.99 | rollout/ep_len_mean = 70.99 | time/fps = 78 | time/time_elapsed = 95 | time/total_timesteps = 7500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 8000 | train/learning_rate = 0.0007 | train/n_updates = 1499 | train/explained_variance = 3.0994415283203125e-05 | train/entropy_loss = -0.3397093415260315 | train/policy_loss = 0.7903553247451782 | train/value_loss = 0.9350654482841492 | time/iterations = 1500 | rollout/ep_rew_mean = 70.99 | rollout/ep_len_mean = 70.99 | time/fps = 78 | time/time_elapsed = 95 | time/total_timesteps = 7500 |
    [38;21m[INFO] 15:17: [A2C[worker: 3]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0021017789840698242 | train/entropy_loss = -0.5292454957962036 | train/policy_loss = 0.290115088224411 | train/value_loss = 1.6142879724502563 | time/iterations = 1600 | rollout/ep_rew_mean = 40.92 | rollout/ep_len_mean = 40.92 | time/fps = 81 | time/time_elapsed = 98 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0021017789840698242 | train/entropy_loss = -0.5292454957962036 | train/policy_loss = 0.290115088224411 | train/value_loss = 1.6142879724502563 | time/iterations = 1600 | rollout/ep_rew_mean = 40.92 | rollout/ep_len_mean = 40.92 | time/fps = 81 | time/time_elapsed = 98 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:17: [A2C[worker: 0]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0001577138900756836 | train/entropy_loss = -0.5802143812179565 | train/policy_loss = 0.24606195092201233 | train/value_loss = 0.7568256855010986 | time/iterations = 1600 | rollout/ep_rew_mean = 65.92 | rollout/ep_len_mean = 65.92 | time/fps = 80 | time/time_elapsed = 98 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0001577138900756836 | train/entropy_loss = -0.5802143812179565 | train/policy_loss = 0.24606195092201233 | train/value_loss = 0.7568256855010986 | time/iterations = 1600 | rollout/ep_rew_mean = 65.92 | rollout/ep_len_mean = 65.92 | time/fps = 80 | time/time_elapsed = 98 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:17: [A2C[worker: 1]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 3.68952751159668e-05 | train/entropy_loss = -0.5150576829910278 | train/policy_loss = 0.2986167371273041 | train/value_loss = 0.7020602822303772 | time/iterations = 1600 | rollout/ep_rew_mean = 71.93 | rollout/ep_len_mean = 71.93 | time/fps = 80 | time/time_elapsed = 99 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = 3.68952751159668e-05 | train/entropy_loss = -0.5150576829910278 | train/policy_loss = 0.2986167371273041 | train/value_loss = 0.7020602822303772 | time/iterations = 1600 | rollout/ep_rew_mean = 71.93 | rollout/ep_len_mean = 71.93 | time/fps = 80 | time/time_elapsed = 99 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:17: [A2C[worker: 4]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0001571178436279297 | train/entropy_loss = -0.49804043769836426 | train/policy_loss = 0.2279295176267624 | train/value_loss = 0.6036964058876038 | time/iterations = 1600 | rollout/ep_rew_mean = 81.46938775510205 | rollout/ep_len_mean = 81.46938775510205 | time/fps = 79 | time/time_elapsed = 100 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0001571178436279297 | train/entropy_loss = -0.49804043769836426 | train/policy_loss = 0.2279295176267624 | train/value_loss = 0.6036964058876038 | time/iterations = 1600 | rollout/ep_rew_mean = 81.46938775510205 | rollout/ep_len_mean = 81.46938775510205 | time/fps = 79 | time/time_elapsed = 100 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:17: [A2C[worker: 2]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0002770423889160156 | train/entropy_loss = -0.4878843426704407 | train/policy_loss = 0.34208714962005615 | train/value_loss = 0.6846947073936462 | time/iterations = 1600 | rollout/ep_rew_mean = 74.21 | rollout/ep_len_mean = 74.21 | time/fps = 79 | time/time_elapsed = 101 | time/total_timesteps = 8000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 8500 | train/learning_rate = 0.0007 | train/n_updates = 1599 | train/explained_variance = -0.0002770423889160156 | train/entropy_loss = -0.4878843426704407 | train/policy_loss = 0.34208714962005615 | train/value_loss = 0.6846947073936462 | time/iterations = 1600 | rollout/ep_rew_mean = 74.21 | rollout/ep_len_mean = 74.21 | time/fps = 79 | time/time_elapsed = 101 | time/total_timesteps = 8000 |
    [38;21m[INFO] 15:17: [A2C[worker: 3]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -1.3232231140136719e-05 | train/entropy_loss = -0.6000055074691772 | train/policy_loss = 0.7648928761482239 | train/value_loss = 1.304478406906128 | time/iterations = 1700 | rollout/ep_rew_mean = 42.88 | rollout/ep_len_mean = 42.88 | time/fps = 81 | time/time_elapsed = 104 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -1.3232231140136719e-05 | train/entropy_loss = -0.6000055074691772 | train/policy_loss = 0.7648928761482239 | train/value_loss = 1.304478406906128 | time/iterations = 1700 | rollout/ep_rew_mean = 42.88 | rollout/ep_len_mean = 42.88 | time/fps = 81 | time/time_elapsed = 104 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:17: [A2C[worker: 0]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 2.7120113372802734e-05 | train/entropy_loss = -0.507287859916687 | train/policy_loss = 0.3262307345867157 | train/value_loss = 0.5406634211540222 | time/iterations = 1700 | rollout/ep_rew_mean = 69.58 | rollout/ep_len_mean = 69.58 | time/fps = 80 | time/time_elapsed = 105 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 2.7120113372802734e-05 | train/entropy_loss = -0.507287859916687 | train/policy_loss = 0.3262307345867157 | train/value_loss = 0.5406634211540222 | time/iterations = 1700 | rollout/ep_rew_mean = 69.58 | rollout/ep_len_mean = 69.58 | time/fps = 80 | time/time_elapsed = 105 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:17: [A2C[worker: 1]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -4.267692565917969e-05 | train/entropy_loss = -0.46631503105163574 | train/policy_loss = 0.24910831451416016 | train/value_loss = 0.48756250739097595 | time/iterations = 1700 | rollout/ep_rew_mean = 76.85 | rollout/ep_len_mean = 76.85 | time/fps = 80 | time/time_elapsed = 105 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = -4.267692565917969e-05 | train/entropy_loss = -0.46631503105163574 | train/policy_loss = 0.24910831451416016 | train/value_loss = 0.48756250739097595 | time/iterations = 1700 | rollout/ep_rew_mean = 76.85 | rollout/ep_len_mean = 76.85 | time/fps = 80 | time/time_elapsed = 105 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:17: [A2C[worker: 4]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 0.001199185848236084 | train/entropy_loss = -0.5061607360839844 | train/policy_loss = 0.22373318672180176 | train/value_loss = 0.39904332160949707 | time/iterations = 1700 | rollout/ep_rew_mean = 84.59 | rollout/ep_len_mean = 84.59 | time/fps = 79 | time/time_elapsed = 106 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 0.001199185848236084 | train/entropy_loss = -0.5061607360839844 | train/policy_loss = 0.22373318672180176 | train/value_loss = 0.39904332160949707 | time/iterations = 1700 | rollout/ep_rew_mean = 84.59 | rollout/ep_len_mean = 84.59 | time/fps = 79 | time/time_elapsed = 106 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:17: [A2C[worker: 2]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 5.97834587097168e-05 | train/entropy_loss = -0.5063344240188599 | train/policy_loss = 0.3865240216255188 | train/value_loss = 0.46397504210472107 | time/iterations = 1700 | rollout/ep_rew_mean = 78.77 | rollout/ep_len_mean = 78.77 | time/fps = 79 | time/time_elapsed = 107 | time/total_timesteps = 8500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 9000 | train/learning_rate = 0.0007 | train/n_updates = 1699 | train/explained_variance = 5.97834587097168e-05 | train/entropy_loss = -0.5063344240188599 | train/policy_loss = 0.3865240216255188 | train/value_loss = 0.46397504210472107 | time/iterations = 1700 | rollout/ep_rew_mean = 78.77 | rollout/ep_len_mean = 78.77 | time/fps = 79 | time/time_elapsed = 107 | time/total_timesteps = 8500 |
    [38;21m[INFO] 15:17: [A2C[worker: 3]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -5.4836273193359375e-06 | train/entropy_loss = -0.5096045732498169 | train/policy_loss = -57.940765380859375 | train/value_loss = 1825.1627197265625 | time/iterations = 1800 | rollout/ep_rew_mean = 47.23 | rollout/ep_len_mean = 47.23 | time/fps = 81 | time/time_elapsed = 110 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -5.4836273193359375e-06 | train/entropy_loss = -0.5096045732498169 | train/policy_loss = -57.940765380859375 | train/value_loss = 1825.1627197265625 | time/iterations = 1800 | rollout/ep_rew_mean = 47.23 | rollout/ep_len_mean = 47.23 | time/fps = 81 | time/time_elapsed = 110 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:17: [A2C[worker: 0]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 4.416704177856445e-05 | train/entropy_loss = -0.3975924253463745 | train/policy_loss = 0.6128240823745728 | train/value_loss = 0.3539719879627228 | time/iterations = 1800 | rollout/ep_rew_mean = 73.76 | rollout/ep_len_mean = 73.76 | time/fps = 80 | time/time_elapsed = 111 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 4.416704177856445e-05 | train/entropy_loss = -0.3975924253463745 | train/policy_loss = 0.6128240823745728 | train/value_loss = 0.3539719879627228 | time/iterations = 1800 | rollout/ep_rew_mean = 73.76 | rollout/ep_len_mean = 73.76 | time/fps = 80 | time/time_elapsed = 111 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:17: [A2C[worker: 1]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -3.6716461181640625e-05 | train/entropy_loss = -0.47606077790260315 | train/policy_loss = 0.3415687084197998 | train/value_loss = 0.3145068883895874 | time/iterations = 1800 | rollout/ep_rew_mean = 80.89 | rollout/ep_len_mean = 80.89 | time/fps = 80 | time/time_elapsed = 112 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = -3.6716461181640625e-05 | train/entropy_loss = -0.47606077790260315 | train/policy_loss = 0.3415687084197998 | train/value_loss = 0.3145068883895874 | time/iterations = 1800 | rollout/ep_rew_mean = 80.89 | rollout/ep_len_mean = 80.89 | time/fps = 80 | time/time_elapsed = 112 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:17: [A2C[worker: 4]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 0.0006428956985473633 | train/entropy_loss = -0.5450716018676758 | train/policy_loss = 0.16278564929962158 | train/value_loss = 0.24435774981975555 | time/iterations = 1800 | rollout/ep_rew_mean = 86.4 | rollout/ep_len_mean = 86.4 | time/fps = 79 | time/time_elapsed = 112 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 0.0006428956985473633 | train/entropy_loss = -0.5450716018676758 | train/policy_loss = 0.16278564929962158 | train/value_loss = 0.24435774981975555 | time/iterations = 1800 | rollout/ep_rew_mean = 86.4 | rollout/ep_len_mean = 86.4 | time/fps = 79 | time/time_elapsed = 112 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:17: [A2C[worker: 2]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 0.00013297796249389648 | train/entropy_loss = -0.5040833950042725 | train/policy_loss = 0.2313179224729538 | train/value_loss = 0.28992730379104614 | time/iterations = 1800 | rollout/ep_rew_mean = 83.99 | rollout/ep_len_mean = 83.99 | time/fps = 78 | time/time_elapsed = 114 | time/total_timesteps = 9000 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 9500 | train/learning_rate = 0.0007 | train/n_updates = 1799 | train/explained_variance = 0.00013297796249389648 | train/entropy_loss = -0.5040833950042725 | train/policy_loss = 0.2313179224729538 | train/value_loss = 0.28992730379104614 | time/iterations = 1800 | rollout/ep_rew_mean = 83.99 | rollout/ep_len_mean = 83.99 | time/fps = 78 | time/time_elapsed = 114 | time/total_timesteps = 9000 |
    [38;21m[INFO] 15:17: [A2C[worker: 3]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.00041234493255615234 | train/entropy_loss = -0.6640589833259583 | train/policy_loss = 0.5048211216926575 | train/value_loss = 0.769464910030365 | time/iterations = 1900 | rollout/ep_rew_mean = 50.03 | rollout/ep_len_mean = 50.03 | time/fps = 81 | time/time_elapsed = 116 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 3]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = 0.00041234493255615234 | train/entropy_loss = -0.6640589833259583 | train/policy_loss = 0.5048211216926575 | train/value_loss = 0.769464910030365 | time/iterations = 1900 | rollout/ep_rew_mean = 50.03 | rollout/ep_len_mean = 50.03 | time/fps = 81 | time/time_elapsed = 116 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:17: [A2C[worker: 0]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.00033795833587646484 | train/entropy_loss = -0.41418904066085815 | train/policy_loss = 0.29809731245040894 | train/value_loss = 0.20698051154613495 | time/iterations = 1900 | rollout/ep_rew_mean = 77.17 | rollout/ep_len_mean = 77.17 | time/fps = 80 | time/time_elapsed = 117 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 0]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.00033795833587646484 | train/entropy_loss = -0.41418904066085815 | train/policy_loss = 0.29809731245040894 | train/value_loss = 0.20698051154613495 | time/iterations = 1900 | rollout/ep_rew_mean = 77.17 | rollout/ep_len_mean = 77.17 | time/fps = 80 | time/time_elapsed = 117 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:17: [A2C[worker: 1]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -1.2278556823730469e-05 | train/entropy_loss = -0.5642617344856262 | train/policy_loss = 0.1860944926738739 | train/value_loss = 0.174888014793396 | time/iterations = 1900 | rollout/ep_rew_mean = 85.17 | rollout/ep_len_mean = 85.17 | time/fps = 80 | time/time_elapsed = 118 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 1]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -1.2278556823730469e-05 | train/entropy_loss = -0.5642617344856262 | train/policy_loss = 0.1860944926738739 | train/value_loss = 0.174888014793396 | time/iterations = 1900 | rollout/ep_rew_mean = 85.17 | rollout/ep_len_mean = 85.17 | time/fps = 80 | time/time_elapsed = 118 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:17: [A2C[worker: 4]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.00026214122772216797 | train/entropy_loss = -0.31037575006484985 | train/policy_loss = 0.21420235931873322 | train/value_loss = 0.12318943440914154 | time/iterations = 1900 | rollout/ep_rew_mean = 93.41 | rollout/ep_len_mean = 93.41 | time/fps = 79 | time/time_elapsed = 118 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 4]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.00026214122772216797 | train/entropy_loss = -0.31037575006484985 | train/policy_loss = 0.21420235931873322 | train/value_loss = 0.12318943440914154 | time/iterations = 1900 | rollout/ep_rew_mean = 93.41 | rollout/ep_len_mean = 93.41 | time/fps = 79 | time/time_elapsed = 118 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:17: [A2C[worker: 2]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.0001533031463623047 | train/entropy_loss = -0.4749812185764313 | train/policy_loss = 0.08823072910308838 | train/value_loss = 0.1588486135005951 | time/iterations = 1900 | rollout/ep_rew_mean = 87.51 | rollout/ep_len_mean = 87.51 | time/fps = 79 | time/time_elapsed = 120 | time/total_timesteps = 9500 |  [0m
    INFO:rlberry_logger:[A2C[worker: 2]] | max_global_step = 10000 | train/learning_rate = 0.0007 | train/n_updates = 1899 | train/explained_variance = -0.0001533031463623047 | train/entropy_loss = -0.4749812185764313 | train/policy_loss = 0.08823072910308838 | train/value_loss = 0.1588486135005951 | time/iterations = 1900 | rollout/ep_rew_mean = 87.51 | rollout/ep_len_mean = 87.51 | time/fps = 79 | time/time_elapsed = 120 | time/total_timesteps = 9500 |
    [38;21m[INFO] 15:17: ... trained! [0m
    INFO:rlberry_logger:... trained!
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:17: Saved ExperimentManager(A2C) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(A2C) using pickle.
    [38;21m[INFO] 15:17: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-15-47_671933e0/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-15-47_671933e0/manager_obj.pickle'
    [38;21m[INFO] 15:17: Running ExperimentManager fit() for PPO with n_fit = 5 and max_workers = None. [0m
    INFO:rlberry_logger:Running ExperimentManager fit() for PPO with n_fit = 5 and max_workers = None.
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:18:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        1           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        1           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:18:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        3           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        3           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:18:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        0           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        0           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:18:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        4           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        4           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:18:                  agent_name  worker  time/iterations  max_global_step
                                      PPO        2           1               2048 [0m
    INFO:rlberry_logger:                 agent_name  worker  time/iterations  max_global_step
                                      PPO        2           1               2048
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:18: [PPO[worker: 3]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.114583333333332 | rollout/ep_len_mean = 21.114583333333332 | time/fps = 153 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.114583333333332 | rollout/ep_len_mean = 21.114583333333332 | time/fps = 153 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:18: [PPO[worker: 4]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.47252747252747 | rollout/ep_len_mean = 22.47252747252747 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.47252747252747 | rollout/ep_len_mean = 22.47252747252747 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    [38;21m[INFO] 15:18: [PPO[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.195652173913043 | rollout/ep_len_mean = 22.195652173913043 | time/fps = 156 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.195652173913043 | rollout/ep_len_mean = 22.195652173913043 | time/fps = 156 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    [38;21m[INFO] 15:18: [PPO[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.270833333333332 | rollout/ep_len_mean = 21.270833333333332 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 21.270833333333332 | rollout/ep_len_mean = 21.270833333333332 | time/fps = 150 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:18: [PPO[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.428571428571427 | rollout/ep_len_mean = 22.428571428571427 | time/fps = 151 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 4096 | time/iterations = 1 | rollout/ep_rew_mean = 22.428571428571427 | rollout/ep_len_mean = 22.428571428571427 | time/fps = 151 | time/time_elapsed = 13 | time/total_timesteps = 2048 | train/learning_rate = 0.0003 |
    [38;21m[INFO] 15:18: [PPO[worker: 3]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.56 | rollout/ep_len_mean = 25.56 | time/fps = 115 | time/time_elapsed = 35 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.685777054913342 | train/policy_gradient_loss = -0.015416811304748989 | train/value_loss = 50.06237749159336 | train/approx_kl = 0.009302661754190922 | train/clip_fraction = 0.10654296875 | train/loss = 6.911357402801514 | train/explained_variance = -0.0029239654541015625 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.56 | rollout/ep_len_mean = 25.56 | time/fps = 115 | time/time_elapsed = 35 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.685777054913342 | train/policy_gradient_loss = -0.015416811304748989 | train/value_loss = 50.06237749159336 | train/approx_kl = 0.009302661754190922 | train/clip_fraction = 0.10654296875 | train/loss = 6.911357402801514 | train/explained_variance = -0.0029239654541015625 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:18: [PPO[worker: 4]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 24.42 | rollout/ep_len_mean = 24.42 | time/fps = 114 | time/time_elapsed = 35 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.686567067541182 | train/policy_gradient_loss = -0.013559106612228788 | train/value_loss = 53.04667126536369 | train/approx_kl = 0.007739779073745012 | train/clip_fraction = 0.09599609375 | train/loss = 5.753210544586182 | train/explained_variance = -0.012129068374633789 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 24.42 | rollout/ep_len_mean = 24.42 | time/fps = 114 | time/time_elapsed = 35 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.686567067541182 | train/policy_gradient_loss = -0.013559106612228788 | train/value_loss = 53.04667126536369 | train/approx_kl = 0.007739779073745012 | train/clip_fraction = 0.09599609375 | train/loss = 5.753210544586182 | train/explained_variance = -0.012129068374633789 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:18: [PPO[worker: 2]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.9 | rollout/ep_len_mean = 25.9 | time/fps = 114 | time/time_elapsed = 35 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6853457026183605 | train/policy_gradient_loss = -0.01876686694158707 | train/value_loss = 49.69416317045689 | train/approx_kl = 0.009763536974787712 | train/clip_fraction = 0.1189453125 | train/loss = 8.091631889343262 | train/explained_variance = -0.0007129907608032227 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.9 | rollout/ep_len_mean = 25.9 | time/fps = 114 | time/time_elapsed = 35 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6853457026183605 | train/policy_gradient_loss = -0.01876686694158707 | train/value_loss = 49.69416317045689 | train/approx_kl = 0.009763536974787712 | train/clip_fraction = 0.1189453125 | train/loss = 8.091631889343262 | train/explained_variance = -0.0007129907608032227 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:18: [PPO[worker: 1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.93 | rollout/ep_len_mean = 25.93 | time/fps = 114 | time/time_elapsed = 35 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6861033642664551 | train/policy_gradient_loss = -0.014860720737488009 | train/value_loss = 51.91080754995346 | train/approx_kl = 0.008546436205506325 | train/clip_fraction = 0.09501953125 | train/loss = 8.072457313537598 | train/explained_variance = -0.0006860494613647461 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 25.93 | rollout/ep_len_mean = 25.93 | time/fps = 114 | time/time_elapsed = 35 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6861033642664551 | train/policy_gradient_loss = -0.014860720737488009 | train/value_loss = 51.91080754995346 | train/approx_kl = 0.008546436205506325 | train/clip_fraction = 0.09501953125 | train/loss = 8.072457313537598 | train/explained_variance = -0.0006860494613647461 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:18: [PPO[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.93 | rollout/ep_len_mean = 27.93 | time/fps = 111 | time/time_elapsed = 36 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.686796179972589 | train/policy_gradient_loss = -0.010411996347829699 | train/value_loss = 52.62666210830211 | train/approx_kl = 0.007990094833076 | train/clip_fraction = 0.067822265625 | train/loss = 6.580994606018066 | train/explained_variance = 0.0019138455390930176 | train/n_updates = 10 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 27.93 | rollout/ep_len_mean = 27.93 | time/fps = 111 | time/time_elapsed = 36 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.686796179972589 | train/policy_gradient_loss = -0.010411996347829699 | train/value_loss = 52.62666210830211 | train/approx_kl = 0.007990094833076 | train/clip_fraction = 0.067822265625 | train/loss = 6.580994606018066 | train/explained_variance = 0.0019138455390930176 | train/n_updates = 10 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: [PPO[worker: 3]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 36.76 | rollout/ep_len_mean = 36.76 | time/fps = 106 | time/time_elapsed = 57 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6674081826582551 | train/policy_gradient_loss = -0.02044952818323509 | train/value_loss = 33.357339683175084 | train/approx_kl = 0.01135399378836155 | train/clip_fraction = 0.0861328125 | train/loss = 10.87477970123291 | train/explained_variance = 0.1124643087387085 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 36.76 | rollout/ep_len_mean = 36.76 | time/fps = 106 | time/time_elapsed = 57 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6674081826582551 | train/policy_gradient_loss = -0.02044952818323509 | train/value_loss = 33.357339683175084 | train/approx_kl = 0.01135399378836155 | train/clip_fraction = 0.0861328125 | train/loss = 10.87477970123291 | train/explained_variance = 0.1124643087387085 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: [PPO[worker: 1]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.9 | rollout/ep_len_mean = 33.9 | time/fps = 104 | time/time_elapsed = 58 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6677466360852122 | train/policy_gradient_loss = -0.019201106773107313 | train/value_loss = 34.60777574777603 | train/approx_kl = 0.010032091289758682 | train/clip_fraction = 0.0677734375 | train/loss = 11.532221794128418 | train/explained_variance = 0.07705909013748169 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 33.9 | rollout/ep_len_mean = 33.9 | time/fps = 104 | time/time_elapsed = 58 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6677466360852122 | train/policy_gradient_loss = -0.019201106773107313 | train/value_loss = 34.60777574777603 | train/approx_kl = 0.010032091289758682 | train/clip_fraction = 0.0677734375 | train/loss = 11.532221794128418 | train/explained_variance = 0.07705909013748169 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: [PPO[worker: 4]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 34.86 | rollout/ep_len_mean = 34.86 | time/fps = 105 | time/time_elapsed = 58 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6639775650575757 | train/policy_gradient_loss = -0.018421680763276528 | train/value_loss = 31.093872064352034 | train/approx_kl = 0.010040415450930595 | train/clip_fraction = 0.07021484375 | train/loss = 13.227946281433105 | train/explained_variance = 0.08318638801574707 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 34.86 | rollout/ep_len_mean = 34.86 | time/fps = 105 | time/time_elapsed = 58 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6639775650575757 | train/policy_gradient_loss = -0.018421680763276528 | train/value_loss = 31.093872064352034 | train/approx_kl = 0.010040415450930595 | train/clip_fraction = 0.07021484375 | train/loss = 13.227946281433105 | train/explained_variance = 0.08318638801574707 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: [PPO[worker: 2]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 31.49 | rollout/ep_len_mean = 31.49 | time/fps = 104 | time/time_elapsed = 58 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6646271903067827 | train/policy_gradient_loss = -0.015234690687793772 | train/value_loss = 34.75618423521519 | train/approx_kl = 0.00909445807337761 | train/clip_fraction = 0.0509765625 | train/loss = 16.92694854736328 | train/explained_variance = 0.08653199672698975 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 31.49 | rollout/ep_len_mean = 31.49 | time/fps = 104 | time/time_elapsed = 58 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6646271903067827 | train/policy_gradient_loss = -0.015234690687793772 | train/value_loss = 34.75618423521519 | train/approx_kl = 0.00909445807337761 | train/clip_fraction = 0.0509765625 | train/loss = 16.92694854736328 | train/explained_variance = 0.08653199672698975 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: [PPO[worker: 0]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 34.97 | rollout/ep_len_mean = 34.97 | time/fps = 102 | time/time_elapsed = 59 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6665260106325149 | train/policy_gradient_loss = -0.01949929333786713 | train/value_loss = 35.88516912460327 | train/approx_kl = 0.010023040696978569 | train/clip_fraction = 0.071728515625 | train/loss = 15.300204277038574 | train/explained_variance = 0.16405224800109863 | train/n_updates = 20 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 8192 | time/iterations = 3 | rollout/ep_rew_mean = 34.97 | rollout/ep_len_mean = 34.97 | time/fps = 102 | time/time_elapsed = 59 | time/total_timesteps = 6144 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6665260106325149 | train/policy_gradient_loss = -0.01949929333786713 | train/value_loss = 35.88516912460327 | train/approx_kl = 0.010023040696978569 | train/clip_fraction = 0.071728515625 | train/loss = 15.300204277038574 | train/explained_variance = 0.16405224800109863 | train/n_updates = 20 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: [PPO[worker: 3]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 51.56 | rollout/ep_len_mean = 51.56 | time/fps = 101 | time/time_elapsed = 80 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6335867114365101 | train/policy_gradient_loss = -0.01827257257536985 | train/value_loss = 54.07645736932754 | train/approx_kl = 0.008562112227082253 | train/clip_fraction = 0.076025390625 | train/loss = 19.734365463256836 | train/explained_variance = 0.28038662672042847 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 3]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 51.56 | rollout/ep_len_mean = 51.56 | time/fps = 101 | time/time_elapsed = 80 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6335867114365101 | train/policy_gradient_loss = -0.01827257257536985 | train/value_loss = 54.07645736932754 | train/approx_kl = 0.008562112227082253 | train/clip_fraction = 0.076025390625 | train/loss = 19.734365463256836 | train/explained_variance = 0.28038662672042847 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: [PPO[worker: 4]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.88 | rollout/ep_len_mean = 47.88 | time/fps = 100 | time/time_elapsed = 81 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6380983158946038 | train/policy_gradient_loss = -0.015608040693041402 | train/value_loss = 52.277515655756 | train/approx_kl = 0.008105376735329628 | train/clip_fraction = 0.073583984375 | train/loss = 17.945486068725586 | train/explained_variance = 0.18586516380310059 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 4]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.88 | rollout/ep_len_mean = 47.88 | time/fps = 100 | time/time_elapsed = 81 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6380983158946038 | train/policy_gradient_loss = -0.015608040693041402 | train/value_loss = 52.277515655756 | train/approx_kl = 0.008105376735329628 | train/clip_fraction = 0.073583984375 | train/loss = 17.945486068725586 | train/explained_variance = 0.18586516380310059 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: [PPO[worker: 1]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.13 | rollout/ep_len_mean = 47.13 | time/fps = 100 | time/time_elapsed = 81 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6364544820040464 | train/policy_gradient_loss = -0.021391361817950382 | train/value_loss = 51.62417317628861 | train/approx_kl = 0.008821746334433556 | train/clip_fraction = 0.110107421875 | train/loss = 23.230304718017578 | train/explained_variance = 0.30747896432876587 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 1]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 47.13 | rollout/ep_len_mean = 47.13 | time/fps = 100 | time/time_elapsed = 81 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6364544820040464 | train/policy_gradient_loss = -0.021391361817950382 | train/value_loss = 51.62417317628861 | train/approx_kl = 0.008821746334433556 | train/clip_fraction = 0.110107421875 | train/loss = 23.230304718017578 | train/explained_variance = 0.30747896432876587 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: [PPO[worker: 2]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 45.08 | rollout/ep_len_mean = 45.08 | time/fps = 100 | time/time_elapsed = 81 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6299627512693405 | train/policy_gradient_loss = -0.018892907629196997 | train/value_loss = 48.79575335383415 | train/approx_kl = 0.007449302822351456 | train/clip_fraction = 0.091943359375 | train/loss = 19.640207290649414 | train/explained_variance = 0.24936705827713013 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 2]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 45.08 | rollout/ep_len_mean = 45.08 | time/fps = 100 | time/time_elapsed = 81 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6299627512693405 | train/policy_gradient_loss = -0.018892907629196997 | train/value_loss = 48.79575335383415 | train/approx_kl = 0.007449302822351456 | train/clip_fraction = 0.091943359375 | train/loss = 19.640207290649414 | train/explained_variance = 0.24936705827713013 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: [PPO[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 46.39 | rollout/ep_len_mean = 46.39 | time/fps = 98 | time/time_elapsed = 82 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.633923995681107 | train/policy_gradient_loss = -0.020698666851967574 | train/value_loss = 49.806022465229034 | train/approx_kl = 0.009045332670211792 | train/clip_fraction = 0.093212890625 | train/loss = 17.598268508911133 | train/explained_variance = 0.2738593816757202 | train/n_updates = 30 | train/clip_range = 0.2 |  [0m
    INFO:rlberry_logger:[PPO[worker: 0]] | max_global_step = 10240 | time/iterations = 4 | rollout/ep_rew_mean = 46.39 | rollout/ep_len_mean = 46.39 | time/fps = 98 | time/time_elapsed = 82 | time/total_timesteps = 8192 | train/learning_rate = 0.0003 | train/entropy_loss = -0.633923995681107 | train/policy_gradient_loss = -0.020698666851967574 | train/value_loss = 49.806022465229034 | train/approx_kl = 0.009045332670211792 | train/clip_fraction = 0.093212890625 | train/loss = 17.598268508911133 | train/explained_variance = 0.2738593816757202 | train/n_updates = 30 | train/clip_range = 0.2 |
    [38;21m[INFO] 15:19: ... trained! [0m
    INFO:rlberry_logger:... trained!
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    /usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/cast.py:1641: DeprecationWarning: np.find_common_type is deprecated.  Please use `np.result_type` or `np.promote_types`.
    See https://numpy.org/devdocs/release/1.25.0-notes.html and the docs for more information.  (Deprecated NumPy 1.25)
      return np.find_common_type(types, [])
    [38;21m[INFO] 15:19: Saved ExperimentManager(PPO) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(PPO) using pickle.
    [38;21m[INFO] 15:19: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-15-47_79c2d608/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-15-47_79c2d608/manager_obj.pickle'
    [38;21m[INFO] 15:19: Saved ExperimentManager(A2C) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(A2C) using pickle.
    [38;21m[INFO] 15:19: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-15-47_671933e0/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/A2C_2024-04-03_15-15-47_671933e0/manager_obj.pickle'
    [38;21m[INFO] 15:19: Saved ExperimentManager(PPO) using pickle. [0m
    INFO:rlberry_logger:Saved ExperimentManager(PPO) using pickle.
    [38;21m[INFO] 15:19: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-15-47_79c2d608/manager_obj.pickle' [0m
    INFO:rlberry_logger:The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_2024-04-03_15-15-47_79c2d608/manager_obj.pickle'
    [38;21m[INFO] 15:19: Evaluating agent 0 [0m
    INFO:rlberry_logger:Evaluating agent 0
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:20: Evaluating agent 1 [0m
    INFO:rlberry_logger:Evaluating agent 1
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:20: Evaluating agent 2 [0m
    INFO:rlberry_logger:Evaluating agent 2
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:20: Evaluating agent 3 [0m
    INFO:rlberry_logger:Evaluating agent 3
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:20: Evaluating agent 4 [0m
    INFO:rlberry_logger:Evaluating agent 4
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:20: Evaluating agent 0 [0m
    INFO:rlberry_logger:Evaluating agent 0
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:20: Evaluating agent 1 [0m
    INFO:rlberry_logger:Evaluating agent 1
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:21: Evaluating agent 2 [0m
    INFO:rlberry_logger:Evaluating agent 2
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:21: Evaluating agent 3 [0m
    INFO:rlberry_logger:Evaluating agent 3
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:21: Evaluating agent 4 [0m
    INFO:rlberry_logger:Evaluating agent 4
    [INFO] Evaluation:INFO:rlberry_logger:[INFO] Evaluation:
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
    .INFO:rlberry_logger:.
      Evaluation finished
    INFO:rlberry_logger:  Evaluation finished

    [38;21m[INFO] 15:21: Test finished [0m
    INFO:rlberry_logger:Test finished
    [38;21m[INFO] 15:21: Results are  [0m
    INFO:rlberry_logger:Results are


    Step 3
    reject
      Agent1 vs Agent2  mean Agent1  mean Agent2  mean diff  std Agent 1  \
    0       A2C vs PPO      278.048      365.285    -87.237   155.561606

       std Agent 2 decisions
    0    53.671506   smaller
    {'A2C': PosixPath('rlberry_data/temp/manager_data/A2C_2024-04-03_15-15-47_671933e0/manager_obj.pickle'), 'PPO': PosixPath('rlberry_data/temp/manager_data/PPO_2024-04-03_15-15-47_79c2d608/manager_obj.pickle')}
