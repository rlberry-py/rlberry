import numpy as np
import torch
import torch.distributions as distr
from rlberry.agents.utils.memories import Memory


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
def get_vref(batch, val_net, twinq_net, policy_net,
                     gamma: float, ent_alpha: float,
                     device="cpu"):

    states, _, _, _, _ = unpack_batch(batch, device)

    # references for the critic network
    action_dist = policy_net(states)
    # act_dist = distr.Normal(mu_v, torch.exp(policy_net.logstd))
    cur_actions = act_dist.sample()
    q1_v, q2_v = twinq_net(states, cur_actions)
    # element-wise minimum
    vref = torch.min(q1_v, q2_v).squeeze() - \
                 ent_alpha * action_dist.log_prob(cur_actions).sum(dim=1)
    return vref
