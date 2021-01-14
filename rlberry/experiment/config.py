"""
Read .yaml files and convert to AgentStats instances.
"""

from pathlib import Path
from typing import Generator, Tuple
import yaml

from rlberry.stats import AgentStats
from rlberry.utils.factory import load


def env_factory(env_path):
    with open(env_path) as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
        if "module_import" in env_config:
            __import__(env_config.pop("module_import"))
        return load(env_config["constructor"])(**env_config["params"])


def agent_factory(agent_path):
    agent_config = read_agent_config(agent_path)
    base_config = agent_config.pop("base_config", None)
    base_config = read_agent_config(base_config) if base_config else {}
    base_config.update(agent_config)  # TODO: this should be a recursive update
    agent_class = load(base_config.pop("agent_class"))
    return agent_class, base_config


def read_agent_config(agent_path):
    with open(agent_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def parse_experiment_config(path: Path,
                            n_fit: int = 4,
                            n_jobs: int = 4,
                            output_base_dir: str = 'results') -> Generator[Tuple[int, AgentStats], None, None]:
    """
    Read .yaml files and convert to AgentStats instances.

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
        config = yaml.load(file, Loader=yaml.FullLoader)
        train_env = env_factory(config["train_env"])
        eval_env = env_factory(config["eval_env"])
        n_fit = n_fit
        n_jobs = n_jobs
        seed = config["seed"]
        for agent_path in config["agents"]:
            agent_name = Path(agent_path).stem
            agent_class, agent_kwargs = agent_factory(agent_path)
            agent_kwargs.update({
                "horizon": config["horizon"],
                "n_episodes": config["n_episodes"]
            })

            # check if eval_horizon is given
            if "eval_horizon" in config:
                eval_horizon = config["eval_horizon"]
            else:
                eval_horizon = config["horizon"]

            output_dir = Path(output_base_dir) / path.stem / agent_name
            yield seed, AgentStats(agent_class=agent_class,
                                   init_kwargs=agent_kwargs,
                                   agent_name=agent_name,
                                   train_env=train_env,
                                   eval_env=eval_env,
                                   eval_horizon=eval_horizon,
                                   n_fit=n_fit,
                                   n_jobs=n_jobs,
                                   output_dir=output_dir)
