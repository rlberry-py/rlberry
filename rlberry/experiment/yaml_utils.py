from pathlib import Path
from typing import Generator, Tuple
import yaml

from rlberry.seeding.seeding import set_global_seed
from rlberry.stats import AgentStats
from rlberry.utils.factory import load


def read_yaml(path):
    with open(path) as file:
        return yaml.safe_load(file)


def read_agent_config(config_path):
    """
    Read .yaml config file for an Agent instance.

    The file contains the agent class and its parameters.

    TODO: recursive update of base_config.

    Example:

    ``` myagent.yaml
        agent_class: 'rlberry.agents.kernel_based.rs_ucbvi.RSUCBVIAgent'
        gamma: 1.0
        lp_metric: 2
        min_dist: 0.0
        max_repr: 800
        bonus_scale_factor: 1.0
    ```

    Parameters
    ----------
    config_path : str
        yaml file name containing the agent config

    Returns
    -------
    Agent class and dictionary with its init_kwargs
    """
    agent_config = read_yaml(config_path)
    base_config = agent_config.pop("base_config", None)
    base_config = read_yaml(base_config) if base_config else {}
    base_config.update(agent_config)  # TODO: recursive update
    agent_class = load(base_config.pop("agent_class"))
    return agent_class, base_config


def read_env_config(config_path):
    """
    Read .yaml config file for an environment instance.

    The file contains the environment constructor and its params.

    Example:

    ``` env.yaml
        constructor: 'rlberry.envs.benchmarks.grid_exploration.nroom.NRoom'
        params:
            reward_free: false
            array_observation: true
            nrooms: 5
    ```

    Parameters
    ----------
    config_path : str
        yaml file name containing the env config

    Returns
    -------
    Environment instance
    """
    with open(config_path) as file:
        env_config = yaml.safe_load(file)
        if "module_import" in env_config:
            __import__(env_config.pop("module_import"))
        return load(env_config["constructor"])(**env_config["params"])


def parse_experiment_config(path: Path,
                            n_fit: int = 4,
                            n_jobs: int = 4,
                            output_base_dir: str = 'results') -> Generator[Tuple[int, AgentStats], None, None]:
    """
    Read .yaml files. set global seed and convert to AgentStats instances.

    Exemple of experiment config:

    ```experiment.yaml
        description: 'My cool experiment'
        seed: 42
        n_episodes: 1000
        horizon: 50
        train_env: 'env_train.yaml'     # see read_env_config()
        eval_env: 'env_eval.yaml'
        agents:
        - 'agent1.yaml'                 # see read_agent_config()
        - 'agent2.yaml'
    ```

    Parameters
    ----------
    path : Path
        Path to an experiment config
    n_fit : int
        Number of instances of each agent to fit
    n_jobs : int
        Number of parallel jobs
    output_base_dir : str
        Directory where to save AgentStats results.

    Returns
    -------
    seed: int
        global seed
    agent_stats: AgentStats
        the Agent Stats to fit
    """
    with path.open() as file:
        config = yaml.safe_load(file)
        train_env = read_env_config(config["train_env"])
        eval_env = read_env_config(config["eval_env"])
        n_fit = n_fit
        n_jobs = n_jobs

        for agent_path in config["agents"]:
            # set seed before creating AgentStats
            seed = config["seed"]
            set_global_seed(seed)

            agent_name = Path(agent_path).stem
            agent_class, agent_kwargs = read_agent_config(agent_path)
            agent_kwargs.update({
                "horizon": config["horizon"],
                "n_episodes": config["n_episodes"]
            })

            # check if eval_horizon is given
            if "eval_horizon" in config:
                eval_horizon = config["eval_horizon"]
            else:
                eval_horizon = config["horizon"]

            # Process output dir, avoid erasing previous results
            output_dir = Path(output_base_dir) / path.stem / agent_name
            last = 0

            try:
                subdirs = [f for f in output_dir.iterdir() if f.is_dir()]
            except FileNotFoundError:
                subdirs = []

            for dd in subdirs:
                try:
                    idx = int(dd.stem)
                except ValueError:
                    continue
                if idx > last:
                    last = idx

            # append run index to dir
            output_dir = output_dir / str(last+1)

            yield seed, AgentStats(agent_class=agent_class,
                                   init_kwargs=agent_kwargs,
                                   agent_name=agent_name,
                                   train_env=train_env,
                                   eval_env=eval_env,
                                   eval_horizon=eval_horizon,
                                   n_fit=n_fit,
                                   n_jobs=n_jobs,
                                   output_dir=output_dir)


if __name__ == '__main__':
    filename = 'examples/demo_experiment/params_experiment.yaml'
    for (seed, agent_stats) in parse_experiment_config(Path(filename)):
        print(seed)
