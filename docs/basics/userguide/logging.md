(logging_page)=

# How to logging your experiment

To get informations and readable result about the training of your algorithm, you can use different logger.

## Set rlberry's logger level

```python
from rlberry.envs import gym_make
from rlberry.agents.torch import PPOAgent
from rlberry.manager import ExperimentManager, evaluate_agents


env_id = "CartPole-v1"  # Id of the environment

env_ctor = gym_make  # constructor for the env
env_kwargs = dict(id=env_id)  # give the id of the env inside the kwargs


first_experiment = ExperimentManager(
    PPOAgent,  # Agent Class
    (env_ctor, env_kwargs),  # Environment as Tuple(constructor,kwargs)
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
[INFO] 15:50: Running ExperimentManager fit() for PPO_first_experimentCartPole-v1 with n_fit = 1 and max_workers = None. 
[INFO] 15:51: ... trained! 
[INFO] 15:51: Evaluating PPO_first_experimentCartPole-v1... 
[INFO] Evaluation:.....  Evaluation finished 
   PPO_first_experimentCartPole-v1
0                             15.0
1                             16.0
2                             16.0
3                             18.0
4                             15.0
```

As you can see, on the previous output, you have the "[INFO]" output (from [ExperimentManager](rlberry.manager.ExperimentManager))


You can choose the verbosity of your logger 

But, with [rlberry.logging.set_level(level='...')](rlberry.utils.logging.set_level), you can select the level of your logger to choose what type of information you want.
For examle, you can have only the "CRITICAL" information, for this add this lines on top of the previous code, then run it again :
```python 
import rlberry
rlberry.utils.logging.set_level(level='CRITICAL')
```

```none
   PPO_first_experimentCartPole-v1
0                             15.0
1                             16.0
2                             16.0
3                             18.0
4                             15.0
```
As you can see, on the previous output, you don't have the "INFO" output anymore (because it's not a "CRITICAL" output)



## Writer
To keep informations during and after the experiment, rlberry use a 'writer'. The writer is stored inside the [Agent](agent_page), and is updated in its fit() function.

By default (with the [Agent interface](agent_page)), the writer is [DefaultWriter](rlberry.utils.writers.DefaultWriter).

To keep informations about the environment inside the writer, you can wrap the environment inside [WriterWrapper](rlberry.wrappers.WriterWrapper).


To get the data, saved during an experiment, in a Pandas DataFrame, you can use [plot_writer_data](rlberry.manager.plot_writer_data) on the [ExperimentManager](rlberry.manager.ExperimentManager) (or a list of them).  
Example [here](../../auto_examples/demo_bandits/plot_mirror_bandit).

To plot the data, saved during an experiment, you can use [plot_writer_data](rlberry.manager.plot_writer_data) on the [ExperimentManager](rlberry.manager.ExperimentManager) (or a list of them).  
Example [here](../../auto_examples/plot_writer_wrapper).



