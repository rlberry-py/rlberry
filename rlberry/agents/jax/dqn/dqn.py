"""
Notes
-----
* In a conda environment, it might be necessary to run:
$ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
(see: https://github.com/deepmind/acme/issues/47)
See also: https://stackoverflow.com/a/46833531 to set LD_LIBRARY_PATH automatically
when activating the conda environment.

* For priority updates, see https://github.com/deepmind/reverb/issues/28
"""

import functools
import haiku as hk
import jax
import jax.numpy as jnp
import logging
import numpy as np
import optax
import reverb
import rlberry.agents.jax.nets.common as nets
# import rlax
import tensorflow as tf

from gym import spaces
from rlberry import types
from rlberry.agents import Agent
from rlberry.utils.writers import DefaultWriter


logger = logging.getLogger(__name__)


class DQNAgent(Agent):
    """
    Implementation of Deep Q-Learning using JAX.

    Parameters
    ----------
    env : types.Env
        Environment.
    gamma : float
        Discount factor.
    batch_size : int
        Batch size (in number of chunks).
    chunk_size : int
        Size of trajectory chunks to sample from the buffer.
    online_update_interval : int
        Interval (in number of transitions) between updates of the online network.
    target_update_interval : int
        Interval (in number of transitions) between updates of the target network.
    learning_rate : float
        Optimizer learning rate.
    max_replay_size : int
        Maximum number of transitions in the replay buffer.
    """
    name = "JaxDqnAgent"

    def __init__(
        self,
        env: types.Env,
        gamma: float = 0.99,
        batch_size: int = 64,
        chunk_size: int = 8,
        online_update_interval: int = 64,
        target_update_interval: int = 512,
        learning_rate: float = 0.01,
        max_replay_size: int = 100000,
        **kwargs
    ):
        Agent.__init__(self, env, **kwargs)
        self.rng_key = jax.random.PRNGKey(self.rng.integers(2**32).item())
        self.writer = DefaultWriter(name=self.name)

        # checks
        if chunk_size < 1:
            raise ValueError('chunk_size needs to be >= 1')
        if not isinstance(self.env.observation_space, spaces.Box):
            raise ValueError('DQN only implemented for Box observation spaces.')
        if not isinstance(self.env.action_space, spaces.Discrete):
            raise ValueError('DQN only implemented for Discrete action spaces.')

        # params
        self._gamma = gamma
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._online_update_interval = online_update_interval
        self._target_update_interval = target_update_interval
        self._max_replay_size = max_replay_size

        # define specs
        # TODO: generalize. Observation is taken from reset() because gym is
        # mixing things up (returning double instead of float)
        sample_obs = self.env.reset()
        try:
            self._observation_spec = tf.TensorSpec(
                sample_obs.shape, sample_obs.dtype)
        except AttributeError:   # in case sample_obs has no .shape attribute
            self._observation_spec = tf.TensorSpec(
                self.env.observation_space.shape, self.env.observation_space.dtype)
        self._action_spec = tf.TensorSpec(
            self.env.action_space.shape, self.env.action_space.dtype)

        # initialize replay buffer
        self._reverb_server = None
        self._reverb_client = None
        self._reverb_dataset = None
        self._init_replay_buffer()

        # initialize networks
        net_ctor = functools.partial(
            nets.MLPQNetwork,
            num_actions=self.env.action_space.n,
            hidden_sizes=(64, 64)
        )
        self._q_net = hk.without_apply_rng(
            hk.transform(lambda x: net_ctor()(x))
        )
        self._dummy_obs = jnp.ones(self._observation_spec.shape)
        self.rng_key, subkey1 = jax.random.split(self.rng_key)
        self.rng_key, subkey2 = jax.random.split(self.rng_key)
        self._online_params = self._q_net.init(subkey1, self._dummy_obs)
        self._target_params = self._q_net.init(subkey2, self._dummy_obs)

        # initialize optimizer
        self._optimizer = optax.adam(learning_rate)
        self._optimizer_state = self._optimizer.init(self._online_params)

        # counters
        self.total_timesteps = 0
        self.total_episodes = 0

    def _init_replay_buffer(self):
        self._reverb_server = reverb.Server(
            tables=[
                reverb.Table(
                    name='replay_buffer',
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=self._max_replay_size,
                    rate_limiter=reverb.rate_limiters.MinSize(self._batch_size),
                    signature={
                        'actions': tf.TensorSpec(
                            shape=[self._chunk_size, *self._action_spec.shape],
                            dtype=self._action_spec.dtype),
                        'observations': tf.TensorSpec(
                            shape=[self._chunk_size, *self._observation_spec.shape],
                            dtype=self._observation_spec.dtype),
                        'rewards': tf.TensorSpec(
                            shape=[self._chunk_size, ],
                            dtype=np.float32),
                        'discounts': tf.TensorSpec(
                            shape=[self._chunk_size, ],
                            dtype=np.float32),
                        'next_observations': tf.TensorSpec(
                            shape=[self._chunk_size, *self._observation_spec.shape],
                            dtype=self._observation_spec.dtype),
                    },
                ),
            ],
            port=None
        )
        self._reverb_client = reverb.Client(f'localhost:{self._reverb_server.port}')
        self._reverb_dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=f'localhost:{self._reverb_server.port}',
            table='replay_buffer',
            max_in_flight_samples_per_worker=100)
        logger.info(self._reverb_client.server_info())

    def fit(
        self,
        budget: int,
        **kwargs
    ):
        """
        Train DQN agent.

        Parameters
        ----------
        budget: int
            Number of timesteps to train the agent.
        """
        del kwargs
        timesteps_counter = 0
        episode_timesteps = 0   # timesteps within an episode
        episode_rewards = 0.0
        observation = self.env.reset()
        with self._reverb_client.trajectory_writer(num_keep_alive_refs=self._chunk_size) as reverb_writer:
            while timesteps_counter < budget:
                # replace by exploration policy
                action = self.env.action_space.sample()
                next_obs, reward, done, _ = self.env.step(action)

                # store data
                episode_rewards += reward
                reverb_writer.append(
                    {'action': action,
                     'observation': observation,
                     'reward': np.array(reward, dtype=np.float32),
                     'discount': np.array(self._gamma * (1.0 - done), dtype=np.float32),
                     'next_obs': next_obs})
                # increment counter
                episode_timesteps += 1

                # write to table
                if episode_timesteps >= self._chunk_size:
                    reverb_writer.create_item(
                        table='replay_buffer',
                        priority=1.0,
                        trajectory={
                            'actions': reverb_writer.history['action'][-self._chunk_size:],
                            'observations': reverb_writer.history['observation'][-self._chunk_size:],
                            'rewards': reverb_writer.history['reward'][-self._chunk_size:],
                            'discounts': reverb_writer.history['discount'][-self._chunk_size:],
                            'next_observations': reverb_writer.history['next_obs'][-self._chunk_size:],
                        }
                    )

                # for next iteration
                timesteps_counter += 1
                self.total_timesteps += 1
                observation = next_obs
                if done:
                    episode_timesteps = 0
                    self.total_episodes += 1
                    episode_rewards = 0.0
                    observation = self.env.reset()
                    self.writer.add_scalar('episode_rewards', episode_rewards, self.total_timesteps)
                    reverb_writer.end_episode()

    def eval(self, eval_env, **kwargs):
        return 0.0

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, params, batch):
        return 0.0
