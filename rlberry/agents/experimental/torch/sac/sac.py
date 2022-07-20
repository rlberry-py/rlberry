from .utils import ReplayBuffer, get_qref, get_vref, alpha_sync

import torch
import torch.nn as nn
from torch.nn.functional import one_hot

import gym.spaces as spaces

from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.torch.utils.training import optimizer_factory
from rlberry.agents.torch.utils.models import default_policy_net_fn
from rlberry.agents.torch.utils.models import default_value_net_fn
from rlberry.agents.torch.utils.models import default_twinq_net_fn
from rlberry.utils.torch import choose_device
from rlberry.wrappers.uncertainty_estimator_wrapper import UncertaintyEstimatorWrapper

import rlberry

logger = rlberry.logger


class SACAgent(AgentWithSimplePolicy):
    """
    Experimental Soft Actor Critic Agent (WIP).

    SAC, or SOFT Actor Critic, an offpolicy actor-critic deep RL algorithm
    based on the maximum entropy reinforcement learning framework. In this
    framework, the actor aims to maximize expected reward while also
    maximizing entropy.

    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    batch_size : int
        Number of episodes to wait before updating the policy.
    gamma : double
        Discount factor in [0, 1].
    entr_coef : double
        Entropy coefficient.
    learning_rate : double
        Learning rate.
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    k_epochs : int
        Number of epochs per update.
    policy_net_fn : function(env, **kwargs)
        Function that returns an instance of a policy network (pytorch).
        If None, a default net is used.
    value_net_fn : function(env, **kwargs)
        Function that returns an instance of a value network (pytorch).
        If None, a default net is used.
    twinq_net_fn : function(env, **kwargs)
        Function that returns a tuple composed of two Q networks (pytorch).
        If None, a default net function is used.
    policy_net_kwargs : dict
        kwargs for policy_net_fn
    value_net_kwargs : dict
        kwargs for value_net_fn
    twinq_net_kwargs : dict
        kwargs for twinq_net_fn
    use_bonus : bool, default = False
        If true, compute an 'exploration_bonus' and add it to the reward.
        See also UncertaintyEstimatorWrapper.
    uncertainty_estimator_kwargs : dict
        Arguments for the UncertaintyEstimatorWrapper
    device : str
        Device to put the tensors on

    References
    ----------
    Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications."
    arXiv preprint arXiv:1812.05905 (2018).
    """

    name = "SAC"

    def __init__(
        self,
        env,
        batch_size=8,
        gamma=0.99,
        entr_coef=0.01,
        learning_rate=0.01,
        buffer_capacity: int = 30000,
        optimizer_type="ADAM",
        k_epochs=5,
        policy_net_fn=None,
        value_net_fn=None,
        twinq_net_fn=None,
        policy_net_kwargs=None,
        value_net_kwargs=None,
        twinq_net_kwargs=None,
        use_bonus=False,
        uncertainty_estimator_kwargs=None,
        device="cuda:best",
        **kwargs
    ):

        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.use_bonus = use_bonus
        if self.use_bonus:
            self.env = UncertaintyEstimatorWrapper(
                self.env, **uncertainty_estimator_kwargs
            )

        self.batch_size = batch_size
        self.gamma = gamma
        self.entr_coef = entr_coef
        self.learning_rate = learning_rate
        self.buffer_capacity = buffer_capacity
        self.k_epochs = k_epochs
        self.device = choose_device(device)

        self.policy_net_kwargs = policy_net_kwargs or {}
        self.value_net_kwargs = value_net_kwargs or {}
        self.twinq_net_kwargs = twinq_net_kwargs or {}

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        #
        self.policy_net_fn = policy_net_fn or default_policy_net_fn
        self.value_net_fn = value_net_fn or default_value_net_fn
        self.twinq_net_fn = twinq_net_fn or default_twinq_net_fn

        self.optimizer_kwargs = {"optimizer_type": optimizer_type, "lr": learning_rate}

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # categorical policy function
        self.cat_policy = None

        # initialize
        self.reset()

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.rng)

    def reset(self, **kwargs):
        # actor
        self.cat_policy = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(
            self.device
        )
        self.policy_optimizer = optimizer_factory(
            self.cat_policy.parameters(), **self.optimizer_kwargs
        )
        self.cat_policy_old = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(
            self.device
        )
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

        # critic
        self.value_net = self.value_net_fn(self.env, **self.value_net_kwargs).to(
            self.device
        )
        self.target_value_net = self.value_net_fn(self.env, **self.value_net_kwargs).to(
            self.device
        )
        self.value_optimizer = optimizer_factory(
            self.value_net.parameters(), **self.optimizer_kwargs
        )
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        # twinq networks
        twinq_net = self.twinq_net_fn(self.env, **self.twinq_net_kwargs)
        self.q1, self.q2 = twinq_net
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.q1_optimizer = optimizer_factory(
            self.q1.parameters(), **self.optimizer_kwargs
        )
        self.q2_optimizer = optimizer_factory(
            self.q2.parameters(), **self.optimizer_kwargs
        )

        # loss function
        self.MseLoss = nn.MSELoss()

        # initialize episode counter
        self.episode = 0

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
        n_episodes_to_run = budget
        count = 0
        while count < n_episodes_to_run:
            self._run_episode()
            count += 1

    def _get_batch(self, device="cpu"):
        (
            batch_state,
            batch_next_state,
            batch_action,
            batch_action_log_prob,
            batch_reward,
            batch_done,
        ) = self.replay_buffer.sample(self.batch_size)

        # convert to torch tensors
        batch_state_tensor = torch.FloatTensor(batch_state).to(self.device)
        batch_next_state_tensor = torch.FloatTensor(batch_next_state).to(self.device)
        batch_action_tensor = torch.LongTensor(batch_action).to(self.device)
        batch_action_log_prob_tensor = torch.FloatTensor(batch_action_log_prob).to(
            self.device
        )
        batch_reward_tensor = (
            torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        )
        batch_done_tensor = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)

        return (
            batch_state_tensor,
            batch_next_state_tensor,
            batch_action_tensor,
            batch_action_log_prob_tensor,
            batch_reward_tensor,
            batch_done_tensor,
        )

    def _select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action_dist = self.cat_policy_old(state)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)

        return action.item(), action_logprob.item()

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        done = False

        while not done:
            # running policy_old
            action, action_logprob = self._select_action(state)
            next_state, reward, done, info = self.env.step(action)
            episode_rewards += reward

            # check whether to use bonus
            bonus = 0.0
            if self.use_bonus:
                if info is not None and "exploration_bonus" in info:
                    bonus = info["exploration_bonus"]

            # save in batch
            self.replay_buffer.push(
                (state, next_state, action, action_logprob, reward + bonus, done)
            )

            # update state
            state = next_state

        # update; TODO this condition "self.episode % self.batch_size == 0:" seems to be  completely random to me
        # implement self.episode -> self.steps
        self.episode += 1
        if self.episode % self.batch_size == 0:
            self._update()

        # add rewards to writer
        if self.writer is not None:
            self.writer.add_scalar("episode_rewards", episode_rewards, self.episode)

        return episode_rewards

    def _update(self):
        # optimize for K epochs
        for _ in range(self.k_epochs):
            # sample batch
            batch = self._get_batch(self.device)
            states, _, actions, _, _, _ = batch

            # compute target values
            qref = get_qref(batch, self.target_value_net, self.gamma, self.device)
            vref = get_vref(
                self.env,
                batch,
                (self.q1, self.q2),
                self.cat_policy,
                self.entr_coef,
                self.device,
            )

            # Critic
            self.value_optimizer.zero_grad()
            val_v = self.value_net(states)
            v_loss_v = self.MseLoss(val_v.squeeze(), vref)
            v_loss_v.backward()
            self.value_optimizer.step()
            if self.writer is not None:
                self.writer.add_scalar("loss_v", float(v_loss_v.detach()), self.episode)

            # TwinQ
            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()
            actions_one_hot = one_hot(actions, self.env.action_space.n)
            q1_v, q2_v = self.q1(torch.cat([states, actions_one_hot], dim=1)), self.q2(
                torch.cat([states, actions_one_hot], dim=1)
            )
            q1_loss_v = self.MseLoss(q1_v.squeeze(), qref.detach())
            q2_loss_v = self.MseLoss(q2_v.squeeze(), qref.detach())
            q1_loss_v.backward()
            q2_loss_v.backward()
            self.q1_optimizer.step()
            self.q2_optimizer.step()
            if self.writer is not None:
                self.writer.add_scalar(
                    "loss_q1", float(q1_loss_v.detach()), self.episode
                )
                self.writer.add_scalar(
                    "loss_q2", float(q2_loss_v.detach()), self.episode
                )

            # Actor
            self.policy_optimizer.zero_grad()
            action_dist = self.cat_policy(states)
            acts_v = action_dist.sample()
            acts_v_one_hot = one_hot(acts_v, self.env.action_space.n)
            q_out_v1 = self.q1(torch.cat([states, acts_v_one_hot], dim=1))
            q_out_v2 = self.q2(torch.cat([states, acts_v_one_hot], dim=1))
            q_out_v = torch.min(q_out_v1, q_out_v2)
            act_loss = (
                -q_out_v.mean() + self.entr_coef * action_dist.log_prob(acts_v).mean()
            )
            act_loss.backward()
            self.policy_optimizer.step()
            if self.writer is not None:
                self.writer.add_scalar(
                    "loss_act", float(act_loss.detach()), self.episode
                )

        # copy new weights into old policy
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

        # update target_value_net
        alpha_sync(self.value_net, self.target_value_net, 1 - 1e-3)

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        batch_size = trial.suggest_categorical("batch_size", [1, 4, 8, 16, 32])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
        entr_coef = trial.suggest_loguniform("entr_coef", 1e-8, 0.1)
        k_epochs = trial.suggest_categorical("k_epochs", [1, 5, 10, 20])

        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "entr_coef": entr_coef,
            "k_epochs": k_epochs,
        }
