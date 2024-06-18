(logging_page)=

# How to log your experiment

To get informations and readable results about the training of your algorithm, you can use different loggers.
## Set rlberry's logger level
For this example, you will use the "PPO" torch agent from "[StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html)" and wrap it in rlberry Agent. To do that, you need to use [StableBaselinesAgent](rlberry.agents.stable_baselines.StableBaselinesAgent). More information [here](stable_baselines).

```python
from rlberry.envs import gym_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import PPO
from rlberry.manager import ExperimentManager, evaluate_agents


env_id = "CartPole-v1"  # Id of the environment

env_ctor = gym_make  # constructor for the env
env_kwargs = dict(id=env_id)  # give the id of the env inside the kwargs


first_experiment = ExperimentManager(
    StableBaselinesAgent,  # Agent Class to manage stableBaselinesAgents
    (env_ctor, env_kwargs),  # Environment as Tuple(constructor,kwargs)
    init_kwargs=dict(algo_cls=PPO, verbose=1),  # Init value for StableBaselinesAgent
    fit_budget=int(100),  # Budget used to call our agent "fit()"
    eval_kwargs=dict(
        eval_horizon=1000
    ),  # Arguments required to call rlberry.agents.agent.Agent.eval().
    n_fit=1,  # Number of agent instances to fit.
    agent_name="PPO_first_experiment" + env_id,  # Name of the agent
    seed=42,
)

first_experiment.fit()

output = evaluate_agents(
    [first_experiment], n_simulations=5, plot=False
)  # evaluate the experiment on 5 simulations
print(output)
```

```none
[INFO] 09:18: Running ExperimentManager fit() for PPO_first_experimentCartPole-v1 with n_fit = 1 and max_workers = None.
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 23.9     |
|    ep_rew_mean     | 23.9     |
| time/              |          |
|    fps             | 2977     |
|    iterations      | 1        |
|    time_elapsed    | 0        |
|    total_timesteps | 2048     |
---------------------------------
[INFO] 09:18: ... trained!
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
[INFO] 09:18: Saved ExperimentManager(PPO_first_experimentCartPole-v1) using pickle.
[INFO] 09:18: The ExperimentManager was saved in : 'rlberry_data/temp/manager_data/PPO_first_experimentCartPole-v1_2024-04-12_09-18-10_3a9fa8ad/manager_obj.pickle'
[INFO] 09:18: Evaluating PPO_first_experimentCartPole-v1...
[INFO] Evaluation:Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
.....  Evaluation finished
   PPO_first_experimentCartPole-v1
0                             89.0
1                             64.0
2                             82.0
3                            121.0
4                             64.0
```

As you can see, on the previous output, you have the "[INFO]" output (from [ExperimentManager](rlberry.manager.ExperimentManager))


You can choose the verbosity of your logger

But, with [rlberry.logging.set_level(level='...')](rlberry.utils.logging.set_level), you can select the level of your logger to choose what type of information you want.
For examle, you can have only the "CRITICAL" information, for this add this lines on top of the previous code, then run it again :
```python
import rlberry

rlberry.utils.logging.set_level(level="CRITICAL")
```

```none
   PPO_first_experimentCartPole-v1
0                             89.0
1                             64.0
2                             82.0
3                            121.0
4                             64.0
```
As you can see, on the previous output, you don't have the "INFO" output anymore (because it's not a "CRITICAL" output)



## Writer
To keep informations during and after the experiment, rlberry use a 'writer'. The writer is stored inside the [Agent](agent_page), and is updated in its fit() function.

By default (with the [Agent interface](rlberry.agents.Agent)), the writer is [DefaultWriter](rlberry.utils.writers.DefaultWriter).

To keep informations about the environment inside the writer, you can wrap the environment inside [WriterWrapper](rlberry.wrappers.WriterWrapper).


To get the data, saved during an experiment, in a Pandas DataFrame, you can use [plot_writer_data](rlberry.manager.plot_writer_data) on the [ExperimentManager](rlberry.manager.ExperimentManager) (or a list of them).
Example [here](../../auto_examples/demo_bandits/plot_mirror_bandit).

To plot the data, saved during an experiment, you can use [plot_writer_data](rlberry.manager.plot_writer_data) on the [ExperimentManager](rlberry.manager.ExperimentManager) (or a list of them).
Example [here](../../auto_examples/plot_writer_wrapper).
