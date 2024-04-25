(save_load_page)=

# How to save/load an experiment


For this example, we'll use the same code as [ExperimentManager](ExperimentManager_page) (from User Guide), and use the save and load functions.

## how to save an experiment

To save your experiment, you have to train it first (with `fit()`), then you just have to use the `save()` function.

Train the Agent :
```python
from rlberry.envs import gym_make
from rlberry_scool.agents.tabular_rl import QLAgent
from rlberry.manager import ExperimentManager

from rlberry.seeding import Seeder

seeder = Seeder(123)  # seeder initialization

env_id = "FrozenLake-v1"  # Id of the environment
env_ctor = gym_make  # constructor for the env
env_kwargs = dict(
    id=env_id, is_slippery=False
)  # give the id of the env inside the kwargs


experiment_to_save = ExperimentManager(
    QLAgent,  # Agent Class
    (env_ctor, env_kwargs),  # Environment as Tuple(constructor,kwargs)
    init_kwargs=dict(
        gamma=0.95, alpha=0.8, exploration_type="epsilon", exploration_rate=0.25
    ),  # agent args
    fit_budget=int(300000),  # Budget used to call our agent "fit()"
    n_fit=1,  # Number of agent instances to fit.
    seed=seeder,  # to be reproductible
    agent_name="QL" + env_id,  # Name of the agent
    output_dir="./results/",  # where to store the outpus
)

experiment_to_save.fit()
print(experiment_to_save.get_agent_instances()[0].Q)  # print the content of the Q-table
```

```none
[INFO] 11:11: Running ExperimentManager fit() for QLFrozenLake-v1 with n_fit = 1 and max_workers = None.
[INFO] 11:11:                                    agent_name     worker  episode_rewards  max_global_step
                                              QLFrozenLake-v1    0          0.0             178711
[INFO] 11:11: ... trained!
writers.py:108: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  df = pd.concat([df, pd.DataFrame(self._data[tag])], ignore_index=True)
[[0.73509189 0.77378094 0.77378094 0.73509189]
 [0.73509189 0.         0.81450625 0.77378094]
 [0.77378094 0.857375   0.77378094 0.81450625]
 [0.81450625 0.         0.77377103 0.77378092]
 [0.77378094 0.81450625 0.         0.73509189]
 [0.         0.         0.         0.        ]
 [0.         0.9025     0.         0.81450625]
 [0.         0.         0.         0.        ]
 [0.81450625 0.         0.857375   0.77378094]
 [0.81450625 0.9025     0.9025     0.        ]
 [0.857375   0.95       0.         0.857375  ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.9025     0.95       0.857375  ]
 [0.9025     0.95       1.         0.9025    ]
 [0.         0.         0.         0.        ]]
[INFO] 11:11: Saved ExperimentManager(QLFrozenLake-v1) using pickle.
```

After this run, you can see the 'print' of the q-table.
At the end of the fit, the data of this experiment are saved automatically. It will be saved according to the `output_dir` parameter (here `./results/`). If you don't specify the `output_dir` parameter, it will saved by default inside the `rlberry_data/temp/` folder.
(Or you can use temporary folder by importing `tempfile` librrary and using `with tempfile.TemporaryDirectory() as tmpdir:`)

In this folder, you should find :
- `manager_obj.pickle` and folder `agent_handler`, the save of your experiment and your agent.
- `data.csv`, the episodes result during the training process


## How to load a previous experiment?
In this example you will load the experiment saved in the part 1.

To load an experiment previously saved, you need to :
- Locate the file you want to load
- use the function `load()` from the class [ExperimentManager](rlberry.manager.ExperimentManager.load).

```python
import pathlib
from rlberry.envs import gym_make
from rlberry.manager.experiment_manager import ExperimentManager


path_to_load = next(
    pathlib.Path("results").glob("**/manager_obj.pickle")
)  # find the path to the "manager_obj.pickle"

loaded_experiment_manager = ExperimentManager.load(path_to_load)  # load the experiment

print(
    loaded_experiment_manager.get_agent_instances()[0].Q
)  # print the content of the Q-table
```
If you want to test the agent from the loaded Experiment, you can add :

```python
env_id = "FrozenLake-v1"  # Id of the environment
env_ctor = gym_make  # constructor for the env
env_kwargs = dict(
    id=env_id, is_slippery=False
)  # give the id of the env inside the kwargs
test_env = env_ctor(**env_kwargs)  # create the Environment

# test the agent of the experiment on the test environment
observation, info = test_env.reset()
for tt in range(50):
    action = loaded_experiment_manager.get_agent_instances()[0].policy(observation)
    next_observation, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    if done:
        if reward == 1:
            print("Success!")
            break
        else:
            print("Fail! Retry!")
            next_observation, info = test_env.reset()
    observation = next_observation
```

```none
[[0.73509189 0.77378094 0.77378094 0.73509189]
 [0.73509189 0.         0.81450625 0.77378094]
 [0.77378094 0.857375   0.77378094 0.81450625]
 [0.81450625 0.         0.77377103 0.77378092]
 [0.77378094 0.81450625 0.         0.73509189]
 [0.         0.         0.         0.        ]
 [0.         0.9025     0.         0.81450625]
 [0.         0.         0.         0.        ]
 [0.81450625 0.         0.857375   0.77378094]
 [0.81450625 0.9025     0.9025     0.        ]
 [0.857375   0.95       0.         0.857375  ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.9025     0.95       0.857375  ]
 [0.9025     0.95       1.         0.9025    ]
 [0.         0.         0.         0.        ]]
Success!
```

As you can see, we haven't re-fit the experiment, and the q-table is the same as the one previously saved (and the Agent can finish the environment).

## Other informations

The `save` and `load` can be useful for :
- you want to train your agent on a computer, and test/use it on others.
- you have a long training, and you want to do some 'checkpoints'.
- you want to do the training in more than once (only if your agent has "fit(x) then fit(y), is the same as fit(x+y)")
