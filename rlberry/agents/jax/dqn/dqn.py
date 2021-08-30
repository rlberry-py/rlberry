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

import chex
import functools
import haiku as hk
import jax
import jax.numpy as jnp
import logging
import numpy as np
import optax
import reverb
import rlberry.agents.jax.nets.common as nets
import rlax
import tensorflow as tf

from gym import spaces
from rlberry import types
from rlberry.agents import Agent
from rlberry.utils.writers import DefaultWriter


logger = logging.getLogger(__name__)


@chex.dataclass
class AllParams:
    online: chex.ArrayTree
    target: chex.ArrayTree


@chex.dataclass
class AllStates:
    optimizer: chex.ArrayTree
    learner_steps: int
    actor_steps: int


@chex.dataclass
class ActorOutput:
    actions: chex.Array
    q_values: chex.Array


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
        Interval (in number total number of online updates) between updates of the target network.
    learning_rate : float
        Optimizer learning rate.
    epsilon_init : float
        Initial value of epsilon-greedy exploration.
    epsilon_end : float
        End value of epsilon-greedy exploration.
    epsilon_steps : int
        Number of steps over which annealing over epsilon takes place.
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
        online_update_interval: int = 1,
        target_update_interval: int = 512,
        learning_rate: float = 0.01,
        epsilon_init: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_steps: int = 5000,
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
        self._batched_dataset = None
        self._init_replay_buffer()

        # initialize params
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
        self._all_params = AllParams(
            online=self._q_net.init(subkey1, self._dummy_obs),
            target=self._q_net.init(subkey2, self._dummy_obs)
        )

        # initialize optimizer and states
        self._optimizer = optax.adam(learning_rate)
        self._all_states = AllStates(
            optimizer=self._optimizer.init(self._all_params.online),
            learner_steps=0,
            actor_steps=0,
        )

        # epsilon decay
        self._epsilon_schedule = optax.polynomial_schedule(
            init_value=epsilon_init,
            end_value=epsilon_end,
            transition_steps=epsilon_steps,
            transition_begin=0,
            power=1.0,
        )

        # counters
        self.total_timesteps = 0
        self.total_episodes = 0
        self.buffer_entries = 0

    @property
    def dataset(self):
        return self._batched_dataset

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
        logger.info(self._reverb_client.server_info())

    def can_sample(self):
        return self.buffer_entries >= self._batch_size

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
                self.rng_key, subkey = jax.random.split(self.rng_key)
                actor_out, self._all_states = self.actor_step(
                    self._all_params,
                    self._all_states,
                    observation,
                    subkey,
                    evaluation=False,
                )
                action = actor_out.actions.item()
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
                    self.buffer_entries += 1

                # for next iteration
                timesteps_counter += 1
                self.total_timesteps += 1
                observation = next_obs

                # update
                if self.total_timesteps % self._online_update_interval == 0 and self.can_sample():
                    sample = next(self.dataset)
                    batch = sample.data
                    self._all_params, self._all_states, info = self.learner_step(
                        self._all_params,
                        self._all_states,
                        batch
                    )
                    self.writer.add_scalar('q_loss', info['loss'].item(), self.total_timesteps)
                    self.writer.add_scalar(
                        'learner_steps',
                        self._all_states.learner_steps.item(),
                        self.total_timesteps)

                # check if episode ended
                if done:
                    self.writer.add_scalar('episode_rewards', episode_rewards, self.total_timesteps)
                    reverb_writer.end_episode()
                    episode_timesteps = 0
                    self.total_episodes += 1
                    episode_rewards = 0.0
                    observation = self.env.reset()

    def eval(self, eval_env, **kwargs):
        return 0.0

    @functools.partial(jax.jit, static_argnums=(0,))
    def _loss(self, all_params, batch):
        obs_tm1 = batch['observations']
        a_tm1 = batch['actions']
        r_t = batch['rewards']
        discount_t = batch['discounts']
        obs_t = batch['next_observations']

        # remove time dim
        # TODO: check if done correctly
        a_tm1 = a_tm1.flatten()
        r_t = r_t.flatten()
        discount_t = discount_t.flatten()
        obs_tm1 = jnp.reshape(obs_tm1, (-1, obs_tm1.shape[-1]))
        obs_t = jnp.reshape(obs_t, (-1, obs_t.shape[-1]))

        q_tm1 = self._q_net.apply(all_params.online, obs_tm1)
        q_t_val = self._q_net.apply(all_params.target, obs_t)
        q_t_select = self._q_net.apply(all_params.online, obs_t)

        batched_loss = jax.vmap(rlax.double_q_learning)
        td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t, q_t_val, q_t_select)
        loss = jnp.mean(rlax.l2_loss(td_error))

        info = dict(
            loss=loss
        )
        return loss, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def actor_step(self, all_params, all_states, observation, rng_key, evaluation):
        obs = jnp.expand_dims(observation, 0)            # dummy batch
        q_val = self._q_net.apply(all_params.online, obs)[0]   # remove batch
        epsilon = self._epsilon_schedule(all_states.actor_steps)
        train_action = rlax.epsilon_greedy(epsilon).sample(rng_key, q_val)
        eval_action = rlax.greedy().sample(rng_key, q_val)
        action = jax.lax.select(evaluation, eval_action, train_action)
        return (
            ActorOutput(
                actions=action,
                q_values=q_val),
            AllStates(
                optimizer=all_states.optimizer,
                learner_steps=all_states.learner_steps,
                actor_steps=all_states.actor_steps + 1),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def learner_step(self, all_params, all_states, batch):
        target_params = rlax.periodic_update(
            all_params.online,
            all_params.target,
            all_states.learner_steps,
            self._target_update_interval)
        grad, info = jax.grad(self._loss, has_aux=True)(
            all_params,
            batch)
        updates, optimizer_state = self._optimizer.update(
            grad.online,
            all_states.optimizer)
        online_params = optax.apply_updates(all_params.online, updates)
        return (
            AllParams(
                online=online_params,
                target=target_params),
            AllStates(
                optimizer=optimizer_state,
                learner_steps=all_states.learner_steps + 1,
                actor_steps=all_states.actor_steps),
            info
        )
