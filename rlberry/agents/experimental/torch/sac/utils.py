import torch
from torch.nn.functional import one_hot
import gymnasium.spaces as spaces
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, rng):
        """
        Parameters
        ----------
        capacity : int
        Maximum number of transitions
        rng :
        instance of numpy's default_rng
        """
        self.capacity = capacity
        self.rng = rng  # random number generator
        self.memory = []
        self.position = 0

    def push(self, sample):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = self.rng.choice(len(self.memory), size=batch_size)
        samples = [self.memory[idx] for idx in indices]
        return map(np.asarray, zip(*samples))

    def __len__(self):
        return len(self.memory)


@torch.no_grad()
def get_qref(batch, target_val_net, gamma, device="cpu"):
    _, next_states, _, _, rewards, dones = batch
    val_next_states = target_val_net(next_states)

    batch_target_val = rewards + (1 - dones) * gamma * val_next_states
    return batch_target_val


@torch.no_grad()
def get_vref(env, batch, twinq_net, policy_net, ent_alpha: float, device="cpu"):
    assert isinstance(twinq_net, tuple)
    assert isinstance(env.action_space, spaces.Discrete)
    num_actions = env.action_space.n

    states, _, _, _, _, _ = batch
    q1, q2 = twinq_net
    # references for the critic network
    act_dist = policy_net(states)
    cur_actions = act_dist.sample()
    actions_one_hot = one_hot(cur_actions, num_actions)
    q_input = torch.cat([states, actions_one_hot], dim=1)

    q1_v, q2_v = q1(q_input), q2(q_input)
    # element-wise minimum
    vref = torch.min(q1_v, q2_v).squeeze() - ent_alpha * act_dist.log_prob(cur_actions)
    return vref


def alpha_sync(model, target_model, alpha):
    assert isinstance(alpha, float)
    assert 0.0 < alpha <= 1.0
    state = model.state_dict()
    tgt_state = target_model.state_dict()
    for k, v in state.items():
        tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
    return target_model.load_state_dict(tgt_state)
