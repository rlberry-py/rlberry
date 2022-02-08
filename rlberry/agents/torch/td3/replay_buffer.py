"""
Notes
-----

* For priority updates, see https://github.com/deepmind/reverb/issues/28
"""

import jax
import logging
import numpy as np
import tensorflow as tf
import tree
import pathlib
from typing import Union


logger = logging.getLogger(__name__)


try:
    import reverb
except ImportError as ex:
    logger.error(
        f'[ReverbReplayBuffer] Could not import reverb: \n   {ex}   \n'
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
        self.writer.end_episode()

    def _append_to_reverb(self, transition: dict):
        self.writer.append(transition)
        if self.writer.episode_steps >= self.chunk_size:
            trajectory = dict()
            for key in self.writer.history:
                if key not in self.entries:
                    raise RuntimeError(
                        'Cannot add to replay buffer an item that'
                        f' was not setup with setup_entry() method of ReverbReplayBuffer: {key}')
                trajectory[key] = jax.tree_map(lambda x: x[-self.chunk_size:], self.writer.history[key])
            self.writer.create_item(
                table='replay_buffer',
                priority=1.0,
                trajectory=trajectory)
            self.total_items += 1

    def append(self, transition: dict):
        self._append_to_reverb(transition)


class ReverbReplayBuffer:
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
        checkpoint_path: Union[pathlib.Path, str],
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
        self._checkpoint_path = str(checkpoint_path)

    @property
    def dataset(self):
        return self._batched_dataset

    def get_current_size(self):
        return self._reverb_client.server_info()['replay_buffer'].current_size

    def get_writer(self):
        self._chunk_writer = ChunkWriter(
            self._reverb_client,
            self._chunk_size,
            list(self._signature.keys()),
        )
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
        shape : Tuple
            Shape of the data. Can be nested (tuples).
        dtype :
            Type of the data. Can be nested.
        """
        if name in self._signature:
            raise ValueError(f'Entry {name} already added to the replay buffer.')

        # handle possibly nested shapes
        shape_with_chunk = jax.tree_map(
            lambda x: np.array((self._chunk_size,) + tuple(x), dtype=np.int32),
            shape, is_leaf=(lambda y: isinstance(y, tuple)))

        self._signature[name] = tree.map_structure(
            lambda *x: tf.TensorSpec(*x), shape_with_chunk, dtype
        )

    def checkpoint(self):
        """Create checkpoint at self._checkpoint_path"""
        return self._reverb_client.checkpoint()

    def build(self):
        """Creates reverb server, client and dataset."""
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=self._checkpoint_path)
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
            port=None,
            checkpointer=checkpointer
        )
        self._reverb_client = reverb.Client(f'localhost:{self._reverb_server.port}')
        self._reverb_dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=f'localhost:{self._reverb_server.port}',
            table='replay_buffer',
            max_in_flight_samples_per_worker=2 * self._batch_size)
        self._batched_dataset = self._reverb_dataset.batch(self._batch_size, drop_remainder=True).as_numpy_iterator()
        # logger.info(self._reverb_client.server_info())

    #
    # For pickle
    #
    def __getstate__(self):
        # create reverb checkpoint
        self.checkpoint()
        state = self.__dict__.copy()

        # clear reverb data
        state['_reverb_server'] = None
        state['_reverb_client'] = None
        state['_reverb_dataset'] = None
        state['_batched_dataset'] = None
        state['_chunk_writer'] = None
        return state

    def __setstate__(self, newstate):
        # Re-create summary writer with the same logdir
        self.__dict__.update(newstate)
        # rebuild reverb
        self.build()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle
    replay = ReverbReplayBuffer(
        batch_size=1, chunk_size=1, max_replay_size=500000, checkpoint_path='temp/replay_debug')
    replay.setup_entry(name='state', shape=(2,), dtype=np.float32)
    replay.build()
    for _ in range(2):
        initial_buffer_size = replay.get_current_size()
        with replay.get_writer() as writer:
            sampled_from_replay = []
            for ii in range(10000):
                state = (ii + initial_buffer_size) * np.ones((2,), dtype=np.float32)
                state[1] += 0.1
                writer.append(dict(state=state))
                writer.end_episode()
                batch = replay.sample()
                sampled_from_replay.append(batch.data['state'][0][0][0])
        plt.figure()
        plt.plot(sampled_from_replay)

        print("-------------------------------------------------")
        print("save and load replay")
        pickle.dump(replay, open("temp/saved_replay.pickle", "wb"))
        del replay
        replay = pickle.load(open( "temp/saved_replay.pickle", "rb" ) )

    plt.show()
    # print(replay.checkpoint())