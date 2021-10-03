"""
Important
---------

There is something very weird happening when using this agent with the optimize_hyperparams()
method of AgentManager.
If @functools.partial(jax.jit, static_argnums=(0,)) is used as a decorator to jit
the learner_step() and actor_step() methods, the memory use increases a lot
as the number of optuna trials increases.

Using the following instead (in __init__):
        self.actor_step = jax.jit(self._actor_step)
        self.learner_step = jax.jit(self._learner_step)
seems to solve the issue.

However, if we jit the loss function before, as:
        self._loss = jax.jit(self._loss)
        self.actor_step = jax.jit(self._actor_step)
        self.learner_step = jax.jit(self._learner_step)
the memory issue comes back!

It does not seem to be the same issue as this one: https://github.com/google/jax/issues/2072

TODO: try to reproduce this issue with a simple example.
"""

import chex
import functools
import haiku as hk
import jax
import jax.numpy as jnp
import logging
import numpy as np
import optax
import dill
import rlberry.agents.jax.nets.common as nets
import rlax

from gym import spaces
from pathlib import Path
from rlberry import types
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.jax.utils.replay_buffer import ReplayBuffer
from typing import Any, Callable, Mapping, Optional

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


class DQNAgent(AgentWithSimplePolicy):
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
    eval_interval : int
        Interval (in number of transitions) between agent evaluations in fit().
        If None, never evaluate.
    max_episode_length : int
        Maximum length of an episode. If None, episodes will only end if `done = True`
        is returned by env.step().
    lambda_ : float
        Parameter for Peng's Q(lambda). If None, usual Q-learning is used.
    net_constructor : callable
        Constructor for Q network. If None, uses default MLP.
    net_kwargs : dict
        kwargs for network constructor (net_constructor).
    max_gradient_norm : float, default: 100.0
        Maximum gradient norm.
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
            learning_rate: float = 0.001,
            epsilon_init: float = 1.0,
            epsilon_end: float = 0.05,
            epsilon_steps: int = 5000,
            max_replay_size: int = 100000,
            eval_interval: Optional[int] = None,
            max_episode_length: Optional[int] = None,
            lambda_: Optional[float] = None,
            net_constructor: Optional[Callable[..., hk.Module]] = None,
            net_kwargs: Optional[Mapping[str, Any]] = None,
            max_gradient_norm: float = 100.0,
            **kwargs
    ):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        env = self.env
        self.rng_key = jax.random.PRNGKey(self.rng.integers(2 ** 32).item())

        # checks
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
        self._eval_interval = eval_interval
        self._max_episode_length = max_episode_length or np.inf
        self._lambda = lambda_
        self._max_gradient_norm = max_gradient_norm

        #
        # Setup replay buffer
        #

        # define specs
        # TODO: generalize. Observation is taken from reset() because gym is
        # mixing things up (returning double instead of float)
        sample_obs = env.reset()
        try:
            obs_shape, obs_dtype = sample_obs.shape, sample_obs.dtype
        except AttributeError:  # in case sample_obs has no .shape attribute
            obs_shape, obs_dtype = env.observation_space.shape, env.observation_space.dtype
        action_shape, action_dtype = env.action_space.shape, env.action_space.dtype

        self._replay_buffer = ReplayBuffer(
            self._batch_size,
            self._chunk_size,
            self._max_replay_size,
        )
        self._replay_buffer.setup_entry('actions', action_shape, action_dtype)
        self._replay_buffer.setup_entry('observations', obs_shape, obs_dtype)
        self._replay_buffer.setup_entry('next_observations', obs_shape, obs_dtype)
        self._replay_buffer.setup_entry('rewards', (), np.float32)
        self._replay_buffer.setup_entry('discounts', (), np.float32)
        self._replay_buffer.build()

        # initialize network and params
        net_constructor = net_constructor or nets.MLPQNetwork
        net_kwargs = net_kwargs or dict(
            num_actions=self.env.action_space.n,
            hidden_sizes=(64, 64)
        )
        net_ctor = functools.partial(net_constructor, **net_kwargs)
        self._q_net = hk.without_apply_rng(
            hk.transform(lambda x: net_ctor()(x))
        )

        self._dummy_obs = jnp.ones(self.env.observation_space.shape)

        self.rng_key, subkey1 = jax.random.split(self.rng_key)
        self.rng_key, subkey2 = jax.random.split(self.rng_key)

        self._all_params = AllParams(
            online=self._q_net.init(subkey1, self._dummy_obs),
            target=self._q_net.init(subkey2, self._dummy_obs)
        )

        # initialize optimizer and states
        self._optimizer = optax.chain(
            optax.clip_by_global_norm(self._max_gradient_norm),
            optax.adam(learning_rate)
        )
        self._all_states = AllStates(
            optimizer=self._optimizer.init(self._all_params.online),
            learner_steps=jnp.array(0),
            actor_steps=jnp.array(0),
        )

        # epsilon decay
        self._epsilon_schedule = optax.polynomial_schedule(
            init_value=epsilon_init,
            end_value=epsilon_end,
            transition_steps=epsilon_steps,
            transition_begin=0,
            power=1.0,
        )

        # update functions (jit)
        self.actor_step = jax.jit(self._actor_step)
        self.learner_step = jax.jit(self._learner_step)

    def policy(self, observation):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        actor_out, _ = self.actor_step(
            self._all_params,
            self._all_states,
            observation,
            subkey,
            evaluation=True,
        )
        action = actor_out.actions.item()
        return action

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
        episode_rewards = 0.0
        episode_timesteps = 0
        observation = self.env.reset()
        with self._replay_buffer.get_writer() as buffer_writer:
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

                # check max episode length
                done = done and (episode_timesteps < self._max_episode_length)

                # store data
                episode_rewards += reward
                buffer_writer.append(
                    {'actions': action,
                     'observations': observation,
                     'rewards': np.array(reward, dtype=np.float32),
                     'discounts': np.array(self._gamma * (1.0 - done), dtype=np.float32),
                     'next_observations': next_obs})

                # counters and next obs
                timesteps_counter += 1
                episode_timesteps += 1
                observation = next_obs

                # update
                total_timesteps = self._all_states.actor_steps.item()
                if total_timesteps % self._online_update_interval == 0:
                    sample = self._replay_buffer.sample()
                    if sample:
                        batch = sample.data
                        self._all_params, self._all_states, info = self.learner_step(
                            self._all_params,
                            self._all_states,
                            batch
                        )
                        if self.writer:
                            self.writer.add_scalar('q_loss', info['loss'].item(), total_timesteps)
                            self.writer.add_scalar(
                                'learner_steps',
                                self._all_states.learner_steps.item(),
                                total_timesteps)

                # eval
                if self._eval_interval is not None and total_timesteps % self._eval_interval == 0:
                    eval_rewards = self.eval(
                        eval_horizon=self._max_episode_length,
                        n_simimulations=2,
                        gamma=1.0)
                    self.writer.add_scalar(
                        'eval_rewards',
                        eval_rewards,
                        total_timesteps
                    )

                # check if episode ended
                if done:
                    if self.writer:
                        self.writer.add_scalar('episode_rewards', episode_rewards, total_timesteps)
                    buffer_writer.end_episode()
                    episode_rewards = 0.0
                    episode_timesteps = 0
                    observation = self.env.reset()

    def _loss(self, all_params, batch):
        obs_tm1 = batch['observations']
        a_tm1 = batch['actions']
        r_t = batch['rewards']
        discount_t = batch['discounts']
        obs_t = batch['next_observations']

        if self._lambda is None:
            # remove time dim (batch has shape [batch, chunk_size, ...])
            a_tm1 = a_tm1.flatten()
            r_t = r_t.flatten()
            discount_t = discount_t.flatten()
            obs_tm1 = jnp.reshape(obs_tm1, (-1, obs_tm1.shape[-1]))
            obs_t = jnp.reshape(obs_t, (-1, obs_t.shape[-1]))

        q_tm1 = self._q_net.apply(all_params.online, obs_tm1)
        q_t_val = self._q_net.apply(all_params.target, obs_t)
        q_t_select = self._q_net.apply(all_params.online, obs_t)

        if self._lambda is None:
            batched_loss = jax.vmap(rlax.double_q_learning)
            td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t, q_t_val, q_t_select)
        else:
            batched_loss = jax.vmap(rlax.q_lambda)
            batch_lambda = self._lambda * jnp.ones(r_t.shape)
            td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t, q_t_val, batch_lambda)

        loss = jnp.mean(rlax.l2_loss(td_error))

        info = dict(
            loss=loss
        )
        return loss, info

    def _actor_step(self, all_params, all_states, observation, rng_key, evaluation):
        obs = jnp.expand_dims(observation, 0)  # dummy batch
        q_val = self._q_net.apply(all_params.online, obs)[0]  # remove batch
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

    def _learner_step(self, all_params, all_states, batch):
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

    #
    # Custom save/load methods.
    #
    def save(self, filename):
        filename = Path(filename).with_suffix('.pickle')
        filename.parent.mkdir(parents=True, exist_ok=True)

        writer = None
        if dill.pickles(self.writer):
            writer = self.writer

        agent_data = dict(
            rng_key=self.rng_key,
            params=self._all_params,
            states=self._all_states,
            writer=writer,
        )
        with filename.open("wb") as ff:
            dill.dump(agent_data, ff)

        return filename

    @classmethod
    def load(cls, filename, **kwargs):
        filename = Path(filename).with_suffix('.pickle')
        agent = cls(**kwargs)
        with filename.open('rb') as ff:
            agent_data = dill.load(ff)
        agent.key = agent_data['rng_key']
        agent._all_params = agent_data['params']
        agent._all_states = agent_data['states']
        writer = agent_data['writer']
        if writer:
            agent._writer = writer
        return agent

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        gamma = trial.suggest_uniform('gamma', 0.95, 0.99)
        lambda_ = trial.suggest_categorical(
            'lambda_',
            [0.1, 0.5, 0.9, None])
        return dict(
            learning_rate=learning_rate,
            gamma=gamma,
            lambda_=lambda_
        )
