(benchmarks)=

# How to seed your experiment

Rlberry has a tool to download some benchmarks if you need to compare your agent with them.

Currently, the available benchmars are :
 - Pre-trained Reinforcement Learning agents using the rl-baselines3-zoo and Stable Baselines3 ([here](https://github.com/DLR-RM/rl-trained-agents)).

## Download the benchmark
To download the benchmark it's easy, you just have to call the function matching the expected benchmark.
You need to specify the names of the agent and the environment. And If you want overwrite the previous data on this combination. (you can use the `download_path` parameter if you want to download the benchmark in a specific folder).
You can find the API about the benchmark [here](rlberry.benchmarks.benchmark_utils)

```python
from rlberry.benchmarks.benchmark_utils import download_benchmark_from_SB3_zoo

agent_name = "dqn"
environment_name = "PongNoFrameskip-v4_1"

path_with_downloaded_files = download_benchmark_from_SB3_zoo(
    agent_name, environment_name, overwrite=True
)
```

## how to use these benchmarks

in construction ...
