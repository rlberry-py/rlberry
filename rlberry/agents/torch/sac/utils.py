import numpy as np
import torch
import torch.distributions as distr
from rlberry.agents.utils.memories import Memory
from torch.nn.functional import one_hot

import gym.spaces as spaces

#implement next state function?
#implement is_terminal function?




def unpack_batch(batch, device = "cpu"):
    assert isinstance(batch.rewards, list)
    assert isinstance(batch.states, list)
    assert isinstance(batch.actions, list)
    assert isinstance(batch.logprobs, list)
    assert isinstance(batch.is_terminals, list) 
    actions = torch.stack(batch.actions).to(device).detach()
    states = torch.stack(batch.states).to(device).detach()
    rewards = torch.stack(batch.rewards).to(device).detach()
    logprobs = torch.stack(batch.logprobs).to(device).detach()
    is_terminals = torch.stack(batch.is_terminals).to(device).detach()
    return states, actions, logprobs, rewards, is_terminals


#we assume that batch consists of lists of tensors

@torch.no_grad()
def get_qref(batch, target_val_net, gamma, device = "cpu"):
    #extract data
    old_states, _, _, qref, is_terminal = unpack_batch(batch, device)

    #get next_states
    non_terminal = torch.logical_not(is_terminal)
    next_states_idx = (torch.nonzero(non_terminal) + 1).view(-1)
    next_states = old_states[next_states_idx].to(device).detach()


    values = target_val_net(next_states)[:,0]
    qref[non_terminal] += gamma * values
    return qref


@torch.no_grad()
def get_vref(env, batch, twinq_net, policy_net, ent_alpha: float,
                     device="cpu"):

    assert isinstance(twinq_net, tuple)
    assert isinstance(env.action_space, spaces.Discrete)
    num_actions = env.action_space.n

    states, _, _, _, _ = unpack_batch(batch, device)
    q1, q2 = twinq_net
    # references for the critic network
    act_dist = policy_net(states)
    # act_dist = distr.Normal(mu_v, torch.exp(policy_net.logstd))
    cur_actions = act_dist.sample()
    actions_one_hot = one_hot(cur_actions, num_actions)
    q_input = torch.cat([states, actions_one_hot], dim=1)

    q1_v, q2_v = q1(q_input), q2(q_input)
    # element-wise minimum
    vref = torch.min(q1_v, q2_v).squeeze() - \
                 ent_alpha * act_dist.log_prob(cur_actions).sum(dim=1)
    return vref


def alpha_sync(model, target_model, alpha):
    assert isinstance(alpha, float)
    assert 0.0 < alpha <= 1.0
    state = model.state_dict()
    tgt_state = target_model.state_dict()
    for k, v in state.items():
        tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
    return target_model.load_state_dict(tgt_state)
