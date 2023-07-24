import inspect
from typing import Callable, Optional, Union

from gymnasium import spaces
import numpy as np
import torch

from rlberry import types
from rlberry.agents import AgentWithSimplePolicy, AgentTorch
from rlberry.agents.torch.utils.training import (
    loss_function_factory,
    model_factory,
    optimizer_factory,
    size_model_config,
)
from rlberry.agents.torch.dqn.dqn_utils import polynomial_schedule, lambda_returns
from rlberry.agents.utils import replay
from rlberry.utils.torch import choose_device
from rlberry.utils.factory import load


import rlberry

logger = rlberry.logger


def default_q_net_fn(env, **kwargs):
    """
    Returns a default Q value network.
    """
    del kwargs
    model_config = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": (64, 64),
        "reshape": False,
    }
    model_config = size_model_config(env, **model_config)
    return model_factory(**model_config)


class DQNAgent(AgentTorch, AgentWithSimplePolicy):
    """DQN Agent based on PyTorch.

    Notes
    -----
    Uses Q(lambda) for computing targets by default. To recover
    the standard DQN, set :code:`lambda_ = 0.0` and :code:`chunk_size = 1`.

    Parameters
    ----------
    env: :class:`~rlberry.types.Env`
        Environment, can be a tuple (constructor, kwargs)
    gamma: float, default = 0.99
        Discount factor.
    batch_size: int, default=32
        Batch size.
    chunk_size: int, default=8
        Length of sub-trajectories sampled from the replay buffer.
    lambda_: float, default=0.5
        Q(lambda) parameter.
    target_update_parameter : int or float
        If int: interval (in number total number of online updates) between updates of the target network.
        If float: soft update coefficient
    device: str
        Torch device, see :func:`~rlberry.utils.torch.choose_device`
    learning_rate : float, default = 1e-3
        Optimizer learning rate.
    loss_function: {"l1", "l2", "smooth_l1"}, default: "l2"
        Loss function used to compute Bellman error.
    epsilon_init: float, default = 1.0
        Initial epsilon value for epsilon-greedy exploration.
    epsilon_final: float, default = 0.1
        Final epsilon value for epsilon-greedy exploration.
    epsilon_decay_interval : int
        After :code:`epsilon_decay` timesteps, epsilon approaches :code:`epsilon_final`.
    optimizer_type : {"ADAM", "RMS_PROP"}
        Optimization algorithm.
    q_net_constructor : Callable, str or None
        Function/constructor that returns a torch module for the Q-network:
        :code:`qnet = q_net_constructor(env, **kwargs)`.

        Module (Q-network) requirements:

        * Input shape = (batch_dim, chunk_size, obs_dims)

        * Ouput shape = (batch_dim, chunk_size, number_of_actions)

        Example: use `rlberry.agents.torch.utils.training.model_factory_from_env`,
         and `q_net_kwargs`
        parameter to modify the neural network::

            model_configs = {
                "type": "MultiLayerPerceptron",
                "layer_sizes": (5, 5),
                "reshape": False,
            }

            agent = DQNAgent(env,
                q_net_constructor=model_factory_from_env,
                q_net_kwargs=model_configs
                )
        If str then it should correspond to the full path to the constructor function,
        e.g.::
            agent = DQNAgent(env,
                q_net_constructor='rlberry.agents.torch.utils.training.model_factory_from_env',
                q_net_kwargs=model_configs
                )

        If None then it is set to MultiLayerPerceptron with 2 hidden layers
        of size 64

    q_net_kwargs : optional, dict
        Parameters for q_net_constructor.
    use_double_dqn : bool, default = False
        If True, use Double DQN.
    use_prioritized_replay : bool, default = False
        If True, use Prioritized Experience Replay.
    train_interval: int
        Update the model every :code:`train_interval` steps.
        If -1, train only at the end of the episodes.
    gradient_steps: int
        How many gradient steps to do at each update.
        If -1, take the number of timesteps since last update.
    max_replay_size : int
        Maximum number of transitions in the replay buffer.
    learning_starts : int
        How many steps of the model to collect transitions for before learning starts
    eval_interval : int, default = None
        Interval (in number of transitions) between agent evaluations in fit().
        If None, never evaluate.
    
    Attributes
    ----------
        gamma : float, default = 0.99
            Discount factor used to discount future rewards in the Bellman equation.

        batch_size : int, default=32
            Batch size used during the training process.

        chunk_size : int, default=8
            Length of sub-trajectories sampled from the replay buffer.

        lambda_ : float, default=0.5
            Q(lambda) parameter used in Q(lambda) algorithm for computing targets.

        target_update_parameter : int or float
            The parameter that controls the update frequency of the target network.
            If int: interval (in number of total online updates) between updates of the target network.
            If float: soft update coefficient, which controls the rate at which the target network approaches
            the online network.

        device : str
            Torch device on which the agent's neural networks are placed. Use "cuda:best" to choose the best
            available GPU device.

        learning_rate : float, default = 1e-3
            Learning rate used by the optimizer during neural network training.

        epsilon_init : float, default = 1.0
            Initial epsilon value for epsilon-greedy exploration. Epsilon-greedy policy is used to balance
            exploration and exploitation during training.

        epsilon_final : float, default = 0.1
            Final epsilon value for epsilon-greedy exploration. Epsilon will approach this value as the agent
            gains more experience.

        epsilon_decay_interval : int
            The number of timesteps after which the epsilon value will approach `epsilon_final`.

        loss_function : {"l1", "l2", "smooth_l1"}, default: "l2"
            The loss function used to compute the Bellman error during training. The available options are
            Mean Absolute Error ("l1"), Mean Squared Error ("l2"), and Smooth L1 Loss ("smooth_l1").

        optimizer_type : {"ADAM", "RMS_PROP"}
            The optimization algorithm used during neural network training. Choose between ADAM and RMS_PROP.

        q_net_constructor : Callable, str or None
            Function/constructor that returns a torch module for the Q-network.
            Example: use `rlberry.agents.torch.utils.training.model_factory_from_env` and `q_net_kwargs`
            parameter to modify the neural network.

        q_net_kwargs : optional, dict
            Parameters for `q_net_constructor`.

        use_double_dqn : bool, default = False
            If True, use Double DQN algorithm, which helps to reduce overestimation bias in Q-value estimates.

        use_prioritized_replay : bool, default = False
            If True, use Prioritized Experience Replay, which prioritizes transitions in the replay buffer
            based on their TD-errors, to improve the learning process.

        train_interval : int
            The agent updates the model every `train_interval` steps. If -1, the agent only trains at the end
            of each episode.

        gradient_steps : int
            The number of gradient steps to perform at each model update. If -1, the number of timesteps since
            the last update will be used.

        max_replay_size : int
            The maximum number of transitions allowed in the replay buffer.

        learning_starts : int
            The number of steps of the model to collect transitions for before learning starts.

        eval_interval : int, default = None
            The interval (in number of transitions) between agent evaluations in the `fit()` method. If None,
            the agent won't evaluate during training.
    """

    name = "DQN"

    def __init__(
        self,
        env: types.Env,
        gamma: float = 0.99,
        batch_size: int = 32,
        chunk_size: int = 8,
        lambda_: float = 0.5,
        target_update_parameter: Union[int, float] = 0.005,
        device: str = "cuda:best",
        learning_rate: float = 1e-3,
        epsilon_init: float = 1.0,
        epsilon_final: float = 0.1,
        epsilon_decay_interval: int = 20_000,
        loss_function: str = "l2",
        optimizer_type: str = "ADAM",
        q_net_constructor: Optional[Callable[..., torch.nn.Module]] = None,
        q_net_kwargs: Optional[dict] = None,
        use_double_dqn: bool = False,
        use_prioritized_replay: bool = False,
        train_interval: int = 10,
        gradient_steps: int = -1,
        max_replay_size: int = 200_000,
        learning_starts: int = 5_000,
        eval_interval: Optional[int] = None,
        **kwargs,
    ):
        # For all parameters, define self.param = param
        _, _, _, values = inspect.getargvalues(inspect.currentframe())

        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        env = self.env
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)

        # DQN parameters

        # Online and target Q networks, torch device
        self._device = choose_device(device)
        if isinstance(q_net_constructor, str):
            q_net_ctor = load(q_net_constructor)
        elif q_net_constructor is None:
            q_net_ctor = default_q_net_fn
        else:
            q_net_ctor = q_net_constructor
        q_net_kwargs = q_net_kwargs or dict()
        self._qnet_online = q_net_ctor(env, **q_net_kwargs).to(self._device)
        self._qnet_target = q_net_ctor(env, **q_net_kwargs).to(self._device)

        # Optimizer and loss
        optimizer_kwargs = {"optimizer_type": optimizer_type, "lr": learning_rate}
        self._optimizer = optimizer_factory(
            self._qnet_online.parameters(), **optimizer_kwargs
        )
        self._loss_function = loss_function_factory(loss_function, reduction="none")

        # Training params
        self._train_interval = train_interval
        self._gradient_steps = gradient_steps
        self._learning_starts = learning_starts
        self._learning_starts = learning_starts
        self._eval_interval = eval_interval

        # Setup replay buffer
        if hasattr(self.env, "_max_episode_steps"):
            max_episode_steps = self.env._max_episode_steps
        else:
            max_episode_steps = np.inf
        self._max_episode_steps = max_episode_steps

        self._replay_buffer = replay.ReplayBuffer(
            max_replay_size=max_replay_size,
            rng=self.rng,
            max_episode_steps=self._max_episode_steps,
            enable_prioritized=use_prioritized_replay,
        )
        self._replay_buffer.setup_entry("observations", np.float32)
        self._replay_buffer.setup_entry("next_observations", np.float32)
        self._replay_buffer.setup_entry("actions", np.int32)
        self._replay_buffer.setup_entry("rewards", np.float32)
        self._replay_buffer.setup_entry("dones", bool)

        # Counters
        self._total_timesteps = 0
        self._total_episodes = 0
        self._total_updates = 0
        self._timesteps_since_last_update = 0

        # epsilon scheduling
        self._epsilon_schedule = polynomial_schedule(
            self.epsilon_init,
            self.epsilon_final,
            power=1.0,
            transition_steps=self.epsilon_decay_interval,
            transition_begin=0,
        )

    @property
    def total_timesteps(self):
        return self._total_timesteps

    def _must_update(self, is_end_of_episode):
        """Returns true if the model must be updated in the current timestep,
        and the number of gradient steps to take"""
        total_timesteps = self._total_timesteps
        n_gradient_steps = self._gradient_steps

        if total_timesteps < self._learning_starts:
            return False, -1

        if n_gradient_steps == -1:
            n_gradient_steps = self._timesteps_since_last_update

        run_update = False
        if self._train_interval == -1:
            run_update = is_end_of_episode
        else:
            run_update = total_timesteps % self._train_interval == 0
        return run_update, n_gradient_steps

    def _update(self, n_gradient_steps):
        """Update networks."""
        if self.use_prioritized_replay:
            sampling_mode = "prioritized"
        else:
            sampling_mode = "uniform"

        for _ in range(n_gradient_steps):
            # Sample a batch
            sampled = self._replay_buffer.sample(
                self.batch_size, self.chunk_size, sampling_mode=sampling_mode
            )
            if not sampled:
                return

            # Update counters
            self._timesteps_since_last_update = 0
            self._total_updates += 1

            batch = sampled.data
            batch_info = sampled.info
            assert batch["rewards"].shape == (self.batch_size, self.chunk_size)

            # Compute targets
            batch_observations = torch.FloatTensor(batch["observations"]).to(
                self._device
            )
            batch_next_observations = torch.FloatTensor(batch["next_observations"]).to(
                self._device
            )
            batch_actions = torch.LongTensor(batch["actions"]).to(self._device)

            target_q_values_tp1 = self._qnet_target(batch_next_observations).detach()
            # Check if double DQN
            if self.use_double_dqn:
                online_q_values_tp1 = self._qnet_online(
                    batch_next_observations
                ).detach()
                a_argmax = online_q_values_tp1.argmax(dim=-1).detach()
            else:
                a_argmax = target_q_values_tp1.argmax(dim=-1).detach()

            v_tp1 = (
                torch.gather(target_q_values_tp1, dim=-1, index=a_argmax[:, :, None])[
                    :, :, 0
                ]
                .cpu()
                .numpy()
            )

            batch_lambda_returns = lambda_returns(
                batch["rewards"],
                self.gamma * (1.0 - np.array(batch["dones"], dtype=np.float32)),
                v_tp1,
                np.array(self.lambda_, dtype=np.float32),
            )
            targets = torch.tensor(batch_lambda_returns).to(self._device)

            # Compute loss
            batch_q_values = self._qnet_online(batch_observations)
            batch_values = torch.gather(
                batch_q_values, dim=-1, index=batch_actions[:, :, None]
            )[
                :, :, 0
            ]  # shape (batch, chunk)

            assert batch_values.shape == targets.shape
            per_element_loss = self._loss_function(batch_values, targets)
            per_batch_element_loss = per_element_loss.mean(dim=1)
            weights = torch.FloatTensor(batch_info["weights"]).to(self._device)
            loss = torch.sum(per_batch_element_loss * weights) / torch.sum(weights)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if self.writer:
                self.writer.add_scalar(
                    "losses/q_loss", loss.item(), self._total_updates
                )

            # update priorities
            if self.use_prioritized_replay:
                new_priorities = per_element_loss.abs().detach().cpu().numpy() + 1e-6
                self._replay_buffer.update_priorities(
                    batch_info["indices"], new_priorities
                )

            # target update
            if self.target_update_parameter > 1:
                if self._total_updates % self.target_update_parameter == 0:
                    self._qnet_target.load_state_dict(self._qnet_online.state_dict())
            else:
                tau = self.target_update_parameter
                for param, target_param in zip(
                    self._qnet_online.parameters(), self._qnet_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters
        ----------
        budget: int
            Number of timesteps to train the agent for.
            One step = one transition in the environment.
        """
        del kwargs
        timesteps_counter = 0
        episode_rewards = 0.0
        episode_timesteps = 0
        observation, info = self.env.reset()
        while timesteps_counter < budget:
            if self.total_timesteps < self._learning_starts:
                action = self.env.action_space.sample()
            else:
                self._timesteps_since_last_update += 1
                action = self._policy(observation, evaluation=False)
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            done = terminated or truncated

            # store data
            episode_rewards += reward
            self._replay_buffer.append(
                {
                    "observations": observation,
                    "actions": action,
                    "rewards": reward,
                    "dones": done,
                    "next_observations": next_observation,
                }
            )

            # counters and next obs
            self._total_timesteps += 1
            timesteps_counter += 1
            episode_timesteps += 1
            observation = next_observation

            # update
            run_update, n_gradient_steps = self._must_update(done)
            if run_update:
                self._update(n_gradient_steps)

            # eval
            total_timesteps = self._total_timesteps
            if (
                self._eval_interval is not None
                and total_timesteps % self._eval_interval == 0
            ):
                eval_rewards = self.eval(
                    eval_horizon=self._max_episode_steps, gamma=1.0
                )
                if self.writer:
                    buffer_size = len(self._replay_buffer)
                    self.writer.add_scalar(
                        "eval_rewards", eval_rewards, total_timesteps
                    )
                    self.writer.add_scalar("buffer_size", buffer_size, total_timesteps)

            # check if episode ended
            if done:
                self._total_episodes += 1
                self._replay_buffer.end_episode()
                if self.writer:
                    self.writer.add_scalar(
                        "episode_rewards", episode_rewards, total_timesteps
                    )
                    self.writer.add_scalar(
                        "total_episodes", self._total_episodes, total_timesteps
                    )
                episode_rewards = 0.0
                episode_timesteps = 0
                observation, info = self.env.reset()

    def _policy(self, observation, evaluation=False):
        epsilon = self._epsilon_schedule(self.total_timesteps)
        if (not evaluation) and self.rng.uniform() < epsilon:
            if self.writer:
                self.writer.add_scalar("epsilon", epsilon, self.total_timesteps)
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                observation = (
                    torch.FloatTensor(observation).to(self._device).unsqueeze(0)
                )
                qvals_tensor = self._qnet_online(observation)[0]
                action = qvals_tensor.argmax().item()
                return action

    def policy(self, observation):
        return self._policy(observation, evaluation=True)
