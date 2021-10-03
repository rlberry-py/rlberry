from pathlib import Path
from typing import Generator, Tuple
import yaml
from rlberry.utils.factory import load

_AGENT_KEYS = ('init_kwargs', 'eval_kwargs', 'fit_kwargs')


def read_yaml(path):
    with open(path) as file:
        return yaml.safe_load(file)


def process_agent_yaml(path):
    config = read_yaml(path)
    for key in _AGENT_KEYS:
        if key not in config:
            config[key] = {}
    return config


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
    agent_class
    base_config : dict
        dictionary whose keys are ('agent_class', 'init_kwargs', 'eval_kwargs', 'fit_kwargs')
    """
    agent_config = process_agent_yaml(config_path)
    base_config_yaml = agent_config.pop("base_config", None)

    # TODO: recursive update
    if base_config_yaml is None:
        base_config = agent_config
    else:
        base_config = process_agent_yaml(base_config_yaml)
        for key in _AGENT_KEYS:
            try:
                base_config[key].update(agent_config[key])
            except KeyError:
                base_config[key] = agent_config[key]

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
    Tuple (constructor, kwargs) for the env
    """
    with open(config_path) as file:
        env_config = yaml.safe_load(file)
        return load(env_config["constructor"]), env_config["params"]


def parse_experiment_config(path: Path,
                            n_fit: int = 4,
                            output_base_dir: str = 'results',
                            parallelization: str = 'process') -> Generator[Tuple[int, dict], None, None]:
    """
    Read .yaml files. set global seed and convert to AgentManager instances.

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
    output_base_dir : str
        Directory where to save AgentManager results.

    Returns
    -------
    seed: int
        global seed
    agent_manager_kwargs:
        parameters to create an AgentManager instance.
    """
    with path.open() as file:
        config = yaml.safe_load(file)
        train_env = read_env_config(config["train_env"])
        eval_env = read_env_config(config["eval_env"])
        n_fit = n_fit

        for agent_path in config["agents"]:
            # set seed before creating AgentManager
            seed = config["seed"]

            agent_name = Path(agent_path).stem
            agent_class, agent_config = read_agent_config(agent_path)

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

            # kwargs
            init_kwargs = agent_config['init_kwargs']
            eval_kwargs = agent_config['eval_kwargs']
            fit_kwargs = agent_config['fit_kwargs']

            # check if there are global kwargs
            if 'global_init_kwargs' in config:
                init_kwargs.update(config['global_init_kwargs'])
            if 'global_eval_kwargs' in config:
                eval_kwargs.update(config['global_eval_kwargs'])
            if 'global_fit_kwargs' in config:
                fit_kwargs.update(config['global_fit_kwargs'])

            # pop fit_budget from fit_kwargs
            fit_budget = fit_kwargs.pop('fit_budget')

            # append run index to dir
            output_dir = output_dir / str(last + 1)

            yield seed, dict(
                agent_class=agent_class,
                init_kwargs=init_kwargs,
                eval_kwargs=eval_kwargs,
                fit_budget=fit_budget,
                fit_kwargs=fit_kwargs,
                agent_name=agent_name,
                train_env=train_env,
                eval_env=eval_env,
                n_fit=n_fit,
                output_dir=output_dir,
                parallelization=parallelization,
                seed=seed,
                create_unique_out_dir=False)  # output_dir is already made unique above


if __name__ == '__main__':
    filename = 'examples/demo_experiment/params_experiment.yaml'
    for (seed, agent_manager) in parse_experiment_config(Path(filename)):
        print(seed)
