import numpy as np
import torch
import torch.nn as nn
import logging

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy

# from rlberry.agents.utils.memories import Memory

from rlberry.agents.utils.replay import ReplayBuffer
from rlberry.agents.torch.utils.training import optimizer_factory
from rlberry.agents.torch.utils.models import default_policy_net_fn
from rlberry.agents.torch.utils.models import default_value_net_fn
from rlberry.utils.torch import choose_device
from rlberry.utils.factory import load

# from rlberry.envs import gym_make


logger = logging.getLogger(__name__)


class PPOAgent(AgentWithSimplePolicy):
    """
    Proximal Policy Optimization Agent.

    Policy gradient methods for reinforcement learning, which alternate between
    sampling data through interaction with the environment, and optimizing a
    “surrogate” objective function using stochastic gradient ascent

    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    batch_size : int
        Number of *episodes* to wait before updating the policy.
    horizon : int
        Horizon.
    gamma : double
        Discount factor in [0, 1].
    entr_coef : double
        Entropy coefficient.
    vf_coef : double
        Value function loss coefficient.
    learning_rate : double
        Learning rate.
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    eps_clip : double
        PPO clipping range (epsilon).
    k_epochs : int
        Number of epochs per update.
    policy_net_fn : function(env, **kwargs)
        Function that returns an instance of a policy network (pytorch).
        If None, a default net is used.
    value_net_fn : function(env, **kwargs)
        Function that returns an instance of a value network (pytorch).
        If None, a default net is used.
    policy_net_kwargs : dict
        kwargs for policy_net_fn
    value_net_kwargs : dict
        kwargs for value_net_fn
    device: str
        Device to put the tensors on
    use_bonus : bool, default = False
        If true, compute the environment 'exploration_bonus'
        and add it to the reward. See also UncertaintyEstimatorWrapper.
    uncertainty_estimator_kwargs : dict
        kwargs for UncertaintyEstimatorWrapper

    References
    ----------
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017).
    "Proximal Policy Optimization Algorithms."
    arXiv preprint arXiv:1707.06347.

    Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
    "Trust region policy optimization."
    In International Conference on Machine Learning (pp. 1889-1897).
    """

    name = "PPO"

    def __init__(
        self,
        env,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        entr_coef=0.01,
        vf_coef=0.5,
        learning_rate=0.01,
        optimizer_type="ADAM",
        eps_clip=0.2,
        k_epochs=5,
        use_gae=True,
        gae_lambda=0.95,
        policy_net_fn=None,
        value_net_fn=None,
        policy_net_kwargs=None,
        value_net_kwargs=None,
        device="cuda:best",
        **kwargs
    ):

        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.batch_size = batch_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.entr_coef = entr_coef
        self.vf_coef = vf_coef
        self.learning_rate = learning_rate
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        # self.env = env
        self.optimizer_type = optimizer_type
        self.kwargs = kwargs

        self.normalize_advantages = True  # TODO: turn into argument

        # function approximators
        self.policy_net_kwargs = policy_net_kwargs or {}
        self.value_net_kwargs = value_net_kwargs or {}
        # self.env = env[0](**env[1])

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        #
        if isinstance(policy_net_fn, str):
            self.policy_net_fn = load(policy_net_fn)
        elif policy_net_fn is None:
            self.policy_net_fn = default_policy_net_fn
        else:
            self.policy_net_fn = policy_net_fn

        if isinstance(value_net_fn, str):
            self.value_net_fn = load(value_net_fn)
        elif value_net_fn is None:
            self.value_net_fn = default_value_net_fn
        else:
            self.value_net_fn = value_net_fn

        self.device = choose_device(device)

        self.optimizer_kwargs = {"optimizer_type": optimizer_type, "lr": learning_rate}

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.cat_policy = None  # categorical policy function

        # initialize
        self.reset()

    @classmethod
    def from_config(cls, **kwargs):
        kwargs["policy_net_fn"] = eval(kwargs["policy_net_fn"])
        kwargs["value_net_fn"] = eval(kwargs["value_net_fn"])
        return cls(**kwargs)

    def reset(self, **kwargs):
        self.cat_policy = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(
            self.device
        )
        self.policy_optimizer = optimizer_factory(
            self.cat_policy.parameters(), **self.optimizer_kwargs
        )

        self.value_net = self.value_net_fn(self.env, **self.value_net_kwargs).to(
            self.device
        )
        self.value_optimizer = optimizer_factory(
            self.value_net.parameters(), **self.optimizer_kwargs
        )

        self.cat_policy_old = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(
            self.device
        )
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

        self.MseLoss = nn.MSELoss()  # TODO: turn into argument

        self.memory = ReplayBuffer(max_replay_size=self.n_steps, rng=self.rng)
        self.memory.setup_entry("states", dtype=np.float32)
        self.memory.setup_entry("actions", dtype=int)
        self.memory.setup_entry("rewards", dtype=np.float32)
        self.memory.setup_entry("dones", dtype=bool)
        self.memory.setup_entry("action_logprobs", dtype=np.float32)

        self.total_timesteps = 0
        self.total_episodes = 0

    def policy(self, observation):
        state = observation
        assert self.cat_policy is not None
        state = torch.from_numpy(state).float().to(self.device)
        action_dist = self.cat_policy_old(state)
        action = action_dist.sample().item()
        return action

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters
        ----------
        budget: int
            number of episodes. Each episode runs for self.horizon unless it
            enconters a terminal state in which case it stops early.
        """
        del kwargs
        timesteps_counter = 0
        episode_rewards = 0.0
        episode_timesteps = 0
        state = self.env.reset()

        while timesteps_counter < budget:

            # running policy_old
            state = torch.from_numpy(state).float().to(self.device)
            action, action_logprob = self._select_action(state)
            next_state, reward, done, _ = self.env.step(action.item())

            episode_rewards += reward

            self.memory.append(
                {
                    "states": state,
                    "actions": action,
                    "rewards": reward,
                    "dones": done,
                    "action_logprobs": action_logprob,
                }
            )

            # counters and next obs
            self.total_timesteps += 1
            timesteps_counter += 1
            episode_timesteps += 1
            state = next_state

            if self.total_timesteps % self.n_steps == 0:
                self._update()

            # update state

            if done:
                self.total_episodes += 1
                self.memory.end_episode()
                if self.writer:
                    self.writer.add_scalar(
                        "episode_rewards", episode_rewards, self.total_timesteps
                    )
                    self.writer.add_scalar(
                        "total_episodes", self.total_episodes, self.total_timesteps
                    )
                episode_rewards = 0.0
                episode_timesteps = 0
                state = self.env.reset()

    def _select_action(self, state):
        action_dist = self.cat_policy_old(state)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)
        return action, action_logprob

    def _update(self):

        memory_data = self.memory.data

        # convert list to tensor
        full_old_states = torch.stack(memory_data["states"]).to(self.device).detach()
        full_old_actions = torch.stack(memory_data["actions"]).to(self.device).detach()
        full_old_logprobs = (
            torch.stack(memory_data["action_logprobs"]).to(self.device).detach()
        )

        state_values = self.value_net(full_old_states).detach()
        state_values = torch.squeeze(state_values).tolist()

        returns, advantages = self._compute_returns_avantages(
            memory_data["rewards"], memory_data["dones"], state_values
        )

        full_old_returns = returns.to(self.device).detach()
        full_old_advantages = advantages.to(self.device).detach()

        # optimize policy for K epochs

        # n_samples = full_old_actions.size(0)
        assert (
            self.n_steps >= self.batch_size
        ), "n_samples must be greater than batch_size"
        n_batches = self.n_steps // self.batch_size

        for _ in range(self.k_epochs):

            # shuffle samples
            rd_indices = self.rng.choice(self.n_steps, size=self.n_steps, replace=False)
            shuffled_states = full_old_states[rd_indices]
            shuffled_actions = full_old_actions[rd_indices]
            shuffled_logprobs = full_old_logprobs[rd_indices]
            shuffled_returns = full_old_returns[rd_indices]
            shuffled_advantages = full_old_advantages[rd_indices]

            for k in range(n_batches):

                # sample batch
                batch_idx = np.arange(
                    k * self.batch_size, min((k + 1) * self.batch_size, self.n_steps)
                )
                old_states = shuffled_states[batch_idx]
                old_actions = shuffled_actions[batch_idx]
                old_logprobs = shuffled_logprobs[batch_idx]
                old_returns = shuffled_returns[batch_idx]
                old_advantages = shuffled_advantages[batch_idx]

                # evaluate old actions and values
                action_dist = self.cat_policy(old_states)
                logprobs = action_dist.log_prob(old_actions)
                state_values = torch.squeeze(self.value_net(old_states))
                dist_entropy = action_dist.entropy()

                # find ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs)

                # TODO: add this option
                # normalizing the rewards
                # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

                # normalize the advantages
                old_advantages = old_advantages.view(
                    -1,
                )

                if self.normalize_advantages:
                    old_advantages = (old_advantages - old_advantages.mean()) / (
                        old_advantages.std() + 1e-10
                    )

                # compute surrogate loss
                surr1 = ratios * old_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * old_advantages
                )
                surr_loss = torch.min(surr1, surr2)

                # compute value function loss
                loss_vf = self.vf_coef * self.MseLoss(state_values, old_returns)

                # compute entropy loss
                loss_entropy = self.entr_coef * dist_entropy

                # compute total loss
                loss = -surr_loss + loss_vf - loss_entropy

                # take gradient step
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                loss.mean().backward()

                self.policy_optimizer.step()
                self.value_optimizer.step()

        # log
        if self.writer:
            self.writer.add_scalar(
                "fit/surrogate_loss",
                surr_loss.mean().cpu().detach().numpy(),
                self.total_episodes,
            )
            self.writer.add_scalar(
                "fit/entropy_loss",
                dist_entropy.mean().cpu().detach().numpy(),
                self.total_episodes,
            )

        # copy new weights into old policy
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

    def _compute_returns_avantages(self, rewards, is_terminals, state_values):

        length_rollout = len(rewards)
        returns = torch.zeros(length_rollout).to(self.device)
        advantages = torch.zeros(length_rollout).to(self.device)

        if not self.use_gae:
            for t in reversed(range(length_rollout)):
                if t == length_rollout - 1:
                    returns[t] = (
                        rewards[t]
                        + self.gamma * (1 - is_terminals[t]) * state_values[-1]
                    )
                else:
                    returns[t] = (
                        rewards[t] + self.gamma * (1 - is_terminals[t]) * returns[t + 1]
                    )

                advantages[t] = returns[t] - state_values[t]
        else:
            last_adv = 0
            for t in reversed(range(length_rollout)):
                if t == length_rollout - 1:
                    returns[t] = (
                        rewards[t]
                        + self.gamma * (1 - is_terminals[t]) * state_values[-1]
                    )
                    td_error = returns[t] - state_values[t]
                else:
                    returns[t] = (
                        rewards[t] + self.gamma * (1 - is_terminals[t]) * returns[t + 1]
                    )
                    td_error = (
                        rewards[t]
                        + self.gamma * (1 - is_terminals[t]) * state_values[t + 1]
                        - state_values[t]
                    )

                last_adv = (
                    self.gae_lambda * self.gamma * (1 - is_terminals[t]) * last_adv
                    + td_error
                )
                advantages[t] = last_adv

        return returns, advantages

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        batch_size = trial.suggest_categorical("batch_size", [1, 4, 8, 16, 32])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)

        entr_coef = trial.suggest_loguniform("entr_coef", 1e-8, 0.1)

        eps_clip = trial.suggest_categorical("eps_clip", [0.1, 0.2, 0.3])

        k_epochs = trial.suggest_categorical("k_epochs", [1, 5, 10, 20])

        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "entr_coef": entr_coef,
            "eps_clip": eps_clip,
            "k_epochs": k_epochs,
        }


# if __name__ == "__main__":
#     env = (gym_make, dict(id="Acrobot-v1"))
#     # env = gym_make(id="Acrobot-v1")
#     ppo = PPOAgent(env)
#     ppo.fit(100000)
