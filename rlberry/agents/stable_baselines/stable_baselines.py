from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import dill
from stable_baselines3.common import utils
import stable_baselines3.common.logger as sb_logging
from stable_baselines3.common.base_class import BaseAlgorithm as SB3Algorithm
from stable_baselines3.common.policies import BasePolicy as SB3Policy

from rlberry import metadata_utils
from rlberry import types
from rlberry.agents import AgentWithSimplePolicy


import rlberry

logger = rlberry.logger


def is_recordable(value: Any) -> bool:
    if isinstance(value, sb_logging.Video):
        return False
    if isinstance(value, sb_logging.Figure):
        return False
    if isinstance(value, sb_logging.Image):
        return False
    return True


class AgentWriter(sb_logging.KVWriter):
    """
    Wraps rlberry's writer to be compatible with stable_baselines3's Logger.

    Parameters:
    -----------
    writer: Agent's writer
        rlberry's writer to be wrapped.
    """

    def __init__(self, writer: Any):
        self.writer = writer

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        if self.writer is None:
            return

        # Exclude entries with unsupported formats
        for key, value in key_values.items():
            if not is_recordable(value):
                key_excluded[key].append("rlberry")

        # Filter excluded entries
        key_values = sb_logging.filter_excluded_keys(
            key_values, key_excluded, "rlberry"
        )

        # Log to writer
        for key, value in key_values.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        pass


class StableBaselinesAgent(AgentWithSimplePolicy):
    """Wraps an StableBaselines3 Algorithm with a rlberry Agent.

    Notes
    -----
    Other keyword arguments are passed to the algorithm's constructor.

    Parameters
    -----------
    env: gym.Env
        Environment
    algo_cls: stable_baselines3 Algorithm class
        Class of the algorithm to wrap (e.g. A2C)
    policy: str or stable_baselines3 Policy class
        Policy to use (e.g. MlpPolicy)
    verbose: int
        Verbosity level: 0 none, 1 training information, 2 tensorflow debug
    tensorboard_log: str
        Path to the directory where to save the tensorboard logs (if None, no logging)
    eval_env : gym.Env or tuple (constructor, kwargs)
        Environment on which to evaluate the agent. If None, copied from env.
    copy_env : bool
        If true, makes a deep copy of the environment.
    save_envs : bool
        Save and loading the environment with the agent.
    seeder : :class:`~rlberry.seeding.seeder.Seeder`, int, or None
        Seeder/seed for random number generation.
    output_dir : str or Path
        Directory that the agent can use to store data.
    _execution_metadata : ExecutionMetadata, optional
        Extra information about agent execution (e.g. about which is the process id where the agent is running).
        Used by :class:`~rlberry.manager.AgentManager`.
    _default_writer_kwargs : dict, optional
        Parameters to initialize :class:`~rlberry.utils.writers.DefaultWriter` (attribute self.writer).
        Used by :class:`~rlberry.manager.AgentManager`.

    Examples
    --------
    >>> from rlberry.envs import gym_make
    >>> from stable_baselines3 import A2C
    >>> from rlberry.agents import StableBaselinesAgent
    >>> env_ctor, env_kwargs = gym_make, dict(id="CartPole-v1")
    >>> env = env_ctor(**env_kwargs)
    >>> agent = StableBaselinesAgent(env, A2C, "MlpPolicy", verbose=1)
    """

    __rlberry_kwargs = [
        "env",
        "eval_env",
        "copy_env",
        "seeder",
        "output_dir",
        "_execution_metadata",
        "_default_writer_kwargs",
        "_thread_shared_data",
    ]

    def __init__(
        self,
        env: types.Env,
        algo_cls: Type[SB3Algorithm] = None,
        policy: Union[str, Type[SB3Policy]] = "MlpPolicy",
        verbose=0,
        tensorboard_log: Optional[str] = None,
        eval_env: Optional[types.Env] = None,
        copy_env: bool = True,
        save_envs=True,
        seeder: Optional[types.Seeder] = None,
        output_dir: Optional[str] = None,
        _execution_metadata: Optional[metadata_utils.ExecutionMetadata] = None,
        _default_writer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super(StableBaselinesAgent, self).__init__(
            env,
            eval_env=eval_env,
            copy_env=copy_env,
            seeder=seeder,
            output_dir=output_dir,
            _execution_metadata=_execution_metadata,
            _default_writer_kwargs=_default_writer_kwargs,
        )
        self._verbose = verbose
        self._tb_log = tensorboard_log
        self._custom_logger = False

        # Remove rlberry's kwargs and add logging kwargs
        kwargs = {k: v for k, v in kwargs.items() if k not in self.__rlberry_kwargs}
        kwargs["verbose"] = self._verbose
        kwargs["tensorboard_log"] = self._tb_log

        # Generate seed for the algorithm using rlberry's seeding
        seed = self.rng.integers(2**32).item()

        # Initialize the algorithm
        assert algo_cls is not None, "algo_cls must be provided"
        self.algo_cls = algo_cls
        utils.set_random_seed(seed)
        self.wrapped = algo_cls(policy, self.env, seed=seed, **kwargs)

    def set_logger(self, logger):
        """Set the logger to a custom SB3 logger.

        Parameters
        ----------
        logger: stable_baselines3.common.logger.Logger
            The logger to use.
        """
        if logger is not None:
            logger.output_formats.append(AgentWriter(self.writer))
        self.wrapped.set_logger(logger)
        self._custom_logger = True

    def reseed(self, seed_seq=None):
        """Reseed the agent."""
        super().reseed(seed_seq)
        seed = self.rng.integers(2**32).item()
        self.wrapped.set_random_seed(seed)

    def save(self, filename):
        """Save the agent to a file."""
        # Save wrappped RL algorithm
        sb3_file = Path(filename).with_suffix(".zip")
        sb3_file.parent.mkdir(parents=True, exist_ok=True)
        self.wrapped.save(sb3_file)

        # Remove the wrapped algorithm if necessary and save the agent
        if not dill.pickles(self.wrapped):
            self.wrapped = None
        return super(StableBaselinesAgent, self).save(filename)

    @classmethod
    def load(cls, filename, **kwargs):
        """Load agent object."""
        agent = super(StableBaselinesAgent, cls).load(filename, **kwargs)

        # Load the wrapped RL algorithm if necessary
        if agent.wrapped is None:
            sb3_file = Path(filename).with_suffix(".zip")
            agent.wrapped = agent.algo_cls.load(sb3_file)
        return agent

    def fit(
        self,
        budget: int,
        tb_log_name: Optional[str] = None,
        reset_num_timesteps: bool = False,
        **kwargs,
    ):
        """Fit the agent.

        Note
        ----
        This method wraps the :code:`learn` method of the algorithm.
        Logging parameters are processered by rlberry in order to use the
        agent.writer.

        Parameters
        ----------
        budget: int
            Number of timesteps to train the agent for.
        tb_log_name: str
            Name of the log to use in tensorboard.
        reset_num_timesteps: bool
            Whether to reset or not the :code: `num_timesteps` attribute
        """
        # If a logger is not provided, use StableBaselines3's default logger
        if not self._custom_logger:
            if tb_log_name is None:
                tb_log_name = self.wrapped.__class__.__name__
            sb_logger = utils.configure_logger(
                self._verbose, self._tb_log, tb_log_name, reset_num_timesteps
            )
            sb_logger.output_formats.append(AgentWriter(self.writer))
            self.wrapped.set_logger(sb_logger)

        # Fit the algorithm
        self.wrapped.learn(
            total_timesteps=budget,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            **kwargs,
        )

    def policy(self, observation, deterministic=True):
        """Get the policy for the given observation.

        Parameters
        ----------
        observation:
            Observation to get the policy for.
        deterministic: bool
            Whether to return a deterministic policy or not.

        Returns
        -------
        The chosen action.
        """
        action, _ = self.wrapped.predict(observation, deterministic=deterministic)
        return action
