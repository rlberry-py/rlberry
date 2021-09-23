"""
Notes
-----

* For priority updates, see https://github.com/deepmind/reverb/issues/28
"""

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

try:
    import reverb
except ImportError as ex:
    logger.error(
        f'[replay_buffer] Could not import reverb: \n   {ex}   \n'
        + ' >>> If you have issues with libpython3.7m.so.1.0, try running: \n'
        + ' >>> $ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib \n'
        + ' >>> in a conda environment, '
        + ' >>> or see https://github.com/deepmind/acme/issues/47 \n'
        + ' >>> See also https://stackoverflow.com/a/46833531 for how to set \n'
        + ' >>> LD_LIBRARY_PATH automatically when activating a conda environment.'
    )
    exit(1)


class ChunkWriter:
    """Wrapper for reverb's TrajectoryWriter"""

    def __init__(self, reverb_client, chunk_size, entries):
        self.writer = None
        self.chunk_size = chunk_size
        self.client = reverb_client
        self.total_items = 0
        self.entries = set(entries)

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
                if key not in self.entries:
                    raise RuntimeError(
                        'Cannot add to replay buffer an item that'
                        f' was not setup with setup_entry() method of ReplayBuffer: {key}')
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
    * Update priorities.

    """

    def __init__(
            self,
            batch_size: int,
            chunk_size: int,
            max_replay_size: int,
    ):
        if chunk_size < 1:
            raise ValueError('chunk_size needs to be >= 1')

        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._max_replay_size = max_replay_size

        self._reverb_server = None
        self._reverb_client = None
        self._reverb_dataset = None
        self._batched_dataset = None
        self._chunk_writer = None
        self._signature = dict()

    @property
    def dataset(self):
        return self._batched_dataset

    def get_writer(self):
        self._chunk_writer = ChunkWriter(self._reverb_client, self._chunk_size, list(self._signature.keys()))
        return self._chunk_writer

    def sample(self):
        if self._chunk_writer is None:
            raise RuntimeError('Calling sample() without previous call to get_writer()')
        if self._chunk_writer.total_items < self._batch_size:
            return None
        return next(self.dataset)

    def setup_entry(self, name, shape, dtype):
        """
        Setup new entry in the replay buffer.

        Parameters
        ----------
        name : str
            Entry name.
        shape :
            Shape of the data. Can be nested.
        dtype :
            Type of the data. Can be nested.
        """
        if name in self._signature:
            raise ValueError(f'Entry {name} already added to the replay buffer.')

        self._signature[name] = tf.TensorSpec(
            shape=[self._chunk_size, *shape],
            dtype=dtype,
        )

    def build(self):
        """Creates reverb server, client and dataset."""
        self._reverb_server = reverb.Server(
            tables=[
                reverb.Table(
                    name='replay_buffer',
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=self._max_replay_size,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                    signature=self._signature,
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
