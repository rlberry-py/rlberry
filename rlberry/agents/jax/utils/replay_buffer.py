import gym
import logging
import numpy as np
import reverb
import tensorflow as tf


logger = logging.getLogger(__name__)


class ChunkWriter:
    """Wrapper for reverb's TrajectoryWriter"""
    def __init__(self, reverb_client, chunk_size):
        self.writer = None
        self.chunk_size = chunk_size
        self.client = reverb_client
        self.total_items = 0

    def __enter__(self):
        self.writer = self.client.trajectory_writer(num_keep_alive_refs=self.chunk_size)
        return self

    def __exit__(self, *args, **kwargs):
        self.writer.__exit__(*args, **kwargs)

    def __del__(self):
        self.writer.__del__()

    def end_episode(self):
        return self.writer.end_episode()

    def append(self, *args, **kwargs):
        self.writer.append(*args, **kwargs)
        if self.writer.episode_steps >= self.chunk_size:
            trajectory = dict()
            for key in self.writer.history:
                trajectory[key] = self.writer.history[key][-self.chunk_size:]
            self.writer.create_item(
                table='replay_buffer',
                priority=1.0,
                trajectory=trajectory)
            self.total_items += 1


class ReplayBuffer:
    """Defines an experience replay using reverb.

    Stores chunks of trajectories.

    TODO:
    * Sampling from different tables (priotirized, uniform etc.)

    """
    def __init__(
        self,
        env: gym.Env,
        batch_size: int,
        chunk_size: int,
        max_replay_size: int,
    ):
        if chunk_size < 1:
            raise ValueError('chunk_size needs to be >= 1')

        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._max_replay_size = max_replay_size

        # define specs
        # TODO: generalize. Observation is taken from reset() because gym is
        # mixing things up (returning double instead of float)
        sample_obs = env.reset()
        try:
            self._observation_spec = tf.TensorSpec(
                sample_obs.shape, sample_obs.dtype)
        except AttributeError:   # in case sample_obs has no .shape attribute
            self._observation_spec = tf.TensorSpec(
                env.observation_space.shape, env.observation_space.dtype)
        self._action_spec = tf.TensorSpec(
            env.action_space.shape, env.action_space.dtype)

        self._reverb_server = None
        self._reverb_client = None
        self._reverb_dataset = None
        self._batched_dataset = None
        self._chunk_writer = None
        self._init_replay_buffer()

    @property
    def dataset(self):
        return self._batched_dataset

    def get_writer(self):
        self._chunk_writer = ChunkWriter(self._reverb_client, self._chunk_size)
        return self._chunk_writer

    def sample(self):
        if self._chunk_writer is None:
            raise RuntimeError('Calling sample() without previous call to get_writer()')
        if self._chunk_writer.total_items < self._batch_size:
            return None
        return next(self.dataset)

    def _init_replay_buffer(self):
        self._reverb_server = reverb.Server(
            tables=[
                reverb.Table(
                    name='replay_buffer',
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=self._max_replay_size,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
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
            max_in_flight_samples_per_worker=2 * self._batch_size)
        self._batched_dataset = self._reverb_dataset.batch(self._batch_size, drop_remainder=True).as_numpy_iterator()
        # logger.info(self._reverb_client.server_info())
