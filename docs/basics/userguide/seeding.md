(seeding_page)=

# How to seed your experiment

Rlberry has a class [Seeder](rlberry.seeding.seeder.Seeder) that conveniently wraps a [NumPy SeedSequence](https://numpy.org/doc/stable/reference/random/parallel.html), and allows you to create independent random number generators for different objects and threads, using a single [Seeder](rlberry.seeding.seeder.Seeder) instance. It works as follows:

## Basics
Suppose you want generate 5 random Integer between 0 and 9.

If you run this code many time, you should have different outputs.
```python
from rlberry.seeding import Seeder

seeder = Seeder()

result_list = []
for _ in range(5):
    result_list.append(seeder.rng.integers(10))
print(result_list)
```

run 1 :
```none
[9, 3, 4, 8, 4]
```
run 2 :
```none
[2, 0, 6, 3, 9]
```
run 3 :
```none
[7, 3, 8, 1, 1]
```


But if you fix the seed as follow, and run it many time... You should have the same 'random' numbers every time :
```python
from rlberry.seeding import Seeder

seeder = Seeder(123)

result_list = []
for _ in range(5):
    result_list.append(seeder.rng.integers(10))
print(result_list)
```

run 1 :
```none
[9, 1, 0, 7, 4]
```
run 2 :
```none
[9, 1, 0, 7, 4]
```
run 3 :
```none
[9, 1, 0, 7, 4]
```

</br>

## In rlberry

### classic usage
Each [Seeder](rlberry.seeding.seeder.Seeder) instance has a random number generator (rng), see [here](https://numpy.org/doc/stable/reference/random/generator.html) to check the methods available in rng.

[Agent](agent_page) and [Environment](environment_page) `reseed(seeder)` functions should use `seeder.spawn()` that allow to create new independent child generators from the same seeder. So it's a good practice to use single seeder to reseed the Agent or Environment, and they will have their own seeder and rng.

When writing your own agents and inheriting from the Agent class, you should use agent.rng whenever you need to generate random numbers; the same applies to your environments.
This is necessary to ensure reproducibility.

```python
from rlberry.seeding import Seeder

seeder = Seeder(123)  # seeder initialization

from rlberry.envs import gym_make
from rlberry_scool.agents import UCBVIAgent


env = gym_make("MountainCar-v0")
env.reseed(seeder)  # seeder first use

agent = UCBVIAgent(env)
agent.reseed(seeder)  # same seeder

# check that the generated numbers are differents
print("env seeder: ", env.seeder)
print("random sample 1 from env rng: ", env.rng.normal())
print("random sample 2 from env rng: ", env.rng.normal())
print("agent seeder: ", agent.seeder)
print("random sample 1 from agent rng: ", agent.rng.normal())
print("random sample 2 from agent rng: ", agent.rng.normal())
```

```none
env seeder:  Seeder object with: SeedSequence(
    entropy=123,
    spawn_key=(0, 0),
)
random sample 1 from env rng:  -1.567498838741829
random sample 2 from env rng:  0.6356604305460527
agent seeder:  Seeder object with: SeedSequence(
    entropy=123,
    spawn_key=(0, 1),
    n_children_spawned=2,
)
random sample 1 from agent rng:  1.2466559261185188
random sample 2 from agent rng:  0.8402527193117317

```

</br>

### With ExperimentManager
For this part we will use the same code from the [ExperimentManager](ExperimentManager_page) part.
3 runs without the seeder :

```python
from rlberry.envs import gym_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import PPO
from rlberry.manager import ExperimentManager, evaluate_agents


env_id = "CartPole-v1"  # Id of the environment

env_ctor = gym_make  # constructor for the env
env_kwargs = dict(id=env_id)  # give the id of the env inside the kwargs


first_experiment = ExperimentManager(
    StableBaselinesAgent,  # Agent Class
    (env_ctor, env_kwargs),  # Environment as Tuple(constructor,kwargs)
    fit_budget=int(100),  # Budget used to call our agent "fit()"
    init_kwargs=dict(algo_cls=PPO),  # Init value for StableBaselinesAgent
    eval_kwargs=dict(
        eval_horizon=1000
    ),  # Arguments required to call rlberry.agents.agent.Agent.eval().
    n_fit=1,  # Number of agent instances to fit.
    agent_name="PPO_first_experiment" + env_id,  # Name of the agent
)

first_experiment.fit()

output = evaluate_agents(
    [first_experiment], n_simulations=5, plot=False
)  # evaluate the experiment on 5 simulations
print(output)
```


Run 1:
```none
[INFO] 14:47: Running ExperimentManager fit() for PPO_first_experimentCartPole-v1 with n_fit = 1 and max_workers = None.
[INFO] 14:47: ... trained!
[INFO] 14:47: Evaluating PPO_first_experimentCartPole-v1...
[INFO] Evaluation:.....  Evaluation finished
   PPO_first_experimentCartPole-v1
0                             20.8
1                             20.8
2                             21.4
3                             24.3
4                             28.8
```

Run 2 :
```none
[INFO] 14:47: Running ExperimentManager fit() for PPO_first_experimentCartPole-v1 with n_fit = 1 and max_workers = None.
[INFO] 14:47: ... trained!
[INFO] 14:47: Evaluating PPO_first_experimentCartPole-v1...
[INFO] Evaluation:.....  Evaluation finished
   PPO_first_experimentCartPole-v1
0                             25.0
1                             19.3
2                             28.5
3                             26.1
4                             19.0
```

Run 3 :
```none
[INFO] 14:47: Running ExperimentManager fit() for PPO_first_experimentCartPole-v1 with n_fit = 1 and max_workers = None.
[INFO] 14:47: ... trained!
[INFO] 14:47: Evaluating PPO_first_experimentCartPole-v1...
[INFO] Evaluation:.....  Evaluation finished
   PPO_first_experimentCartPole-v1
0                             23.6
1                             19.2
2                             20.5
3                             19.8
4                             16.5
```


</br>

**Without the seeder, the outputs are different (non-reproducible).**

</br>

3 runs with the seeder :

```python
from rlberry.envs import gym_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import PPO
from rlberry.manager import ExperimentManager, evaluate_agents

from rlberry.seeding import Seeder

seeder = Seeder(42)

env_id = "CartPole-v1"  # Id of the environment

env_ctor = gym_make  # constructor for the env
env_kwargs = dict(id=env_id)  # give the id of the env inside the kwargs


first_experiment = ExperimentManager(
    StableBaselinesAgent,  # Agent Class
    (env_ctor, env_kwargs),  # Environment as Tuple(constructor,kwargs)
    fit_budget=int(100),  # Budget used to call our agent "fit()"
    init_kwargs=dict(algo_cls=PPO),  # Init value for StableBaselinesAgent
    eval_kwargs=dict(
        eval_horizon=1000
    ),  # Arguments required to call rlberry.agents.agent.Agent.eval().
    n_fit=1,  # Number of agent instances to fit.
    agent_name="PPO_first_experiment" + env_id,  # Name of the agent
    seed=seeder,
)

first_experiment.fit()

output = evaluate_agents(
    [first_experiment], n_simulations=5, plot=False
)  # evaluate the experiment on 5 simulations
print(output)
```

Run 1:
```none
[INFO] 14:46: Running ExperimentManager fit() for PPO_first_experimentCartPole-v1 with n_fit = 1 and max_workers = None.
[INFO] 14:46: ... trained!
[INFO] 14:46: Evaluating PPO_first_experimentCartPole-v1...
[INFO] Evaluation:.....  Evaluation finished
   PPO_first_experimentCartPole-v1
0                             23.3
1                             19.7
2                             23.0
3                             18.8
4                             19.7
```

Run 2 :
```none
[INFO] 14:46: Running ExperimentManager fit() for PPO_first_experimentCartPole-v1 with n_fit = 1 and max_workers = None.
[INFO] 14:46: ... trained!
[INFO] 14:46: Evaluating PPO_first_experimentCartPole-v1...
[INFO] Evaluation:.....  Evaluation finished
   PPO_first_experimentCartPole-v1
0                             23.3
1                             19.7
2                             23.0
3                             18.8
4                             19.7
```


Run 3 :
```none
[INFO] 14:46: Running ExperimentManager fit() for PPO_first_experimentCartPole-v1 with n_fit = 1 and max_workers = None.
[INFO] 14:46: ... trained!
[INFO] 14:46: Evaluating PPO_first_experimentCartPole-v1...
[INFO] Evaluation:.....  Evaluation finished
   PPO_first_experimentCartPole-v1
0                             23.3
1                             19.7
2                             23.0
3                             18.8
4                             19.7
```


</br>


**With the seeder, the outputs are the same (reproducible).**


</br>




### multi-threading
If you want use multi-threading, a seeder can spawn other seeders that are independent from it.
This is useful to seed two different threads, using seeder1 in the first thread, and seeder2 in the second thread.
```python
from rlberry.seeding import Seeder

seeder = Seeder(123)
seeder1, seeder2 = seeder.spawn(2)

print("random sample 1 from seeder1 rng: ", seeder1.rng.normal())
print("random sample 2 from seeder1 rng: ", seeder1.rng.normal())
print("-----")
print("random sample 1 from seeder2 rng: ", seeder2.rng.normal())
print("random sample 2 from seeder2 rng: ", seeder2.rng.normal())
```
```none
random sample 1 from seeder1 rng:  -0.4732958445958833
random sample 2 from seeder1 rng:  0.5863995575997462
-----
random sample 1 from seeder2 rng:  -0.1722486099076424
random sample 2 from seeder2 rng:  -0.1930990650226178
```

</br>

## External libraries
You can also use a seeder to seed external libraries (such as torch) using the method `set_external_seed`.
It will be usefull if you want reproducibility with external libraries. In this example, you will use `torch` to generate random numbers.

If you run this code many time, you should have different outputs.
```python
import torch

result_list = []
for i in range(5):
    result_list.append(torch.randint(2**32, (1,))[0].item())

print(result_list)
```

run 1 :
```none
[3817148928, 671396126, 2950680447, 791815335, 3335786391]
```
run 2 :
```none
[82990446, 2463687945, 1829003305, 647811387, 3543380778]
```
run 3 :
```none
[3887070615, 363268341, 3607514851, 3881090947, 1018754931]
```



If you add to this code a [Seeder](rlberry.seeding.seeder.Seeder), use the `set_external_seed` method, and re-run it, you should have the same 'random' numbers everytime.

```python
import torch
from rlberry.seeding import set_external_seed
from rlberry.seeding import Seeder

seeder = Seeder(123)

set_external_seed(seeder)
result_list = []
for i in range(5):
    result_list.append(torch.randint(2**32, (1,))[0].item())

print(result_list)
```

run 1 :
```none
[693246422, 3606543353, 433394544, 2194426398, 3928404622]
```
run 2 :
```none
[693246422, 3606543353, 433394544, 2194426398, 3928404622]
```
run 3 :
```none
[693246422, 3606543353, 433394544, 2194426398, 3928404622]
```
