import torch
from torch.nn.functional import one_hot
import gym.spaces as spaces


def unpack_batch(batch, device="cpu"):
    assert isinstance(batch.rewards, list)
    assert isinstance(batch.states, list)
    assert isinstance(batch.actions, list)
    assert isinstance(batch.logprobs, list)
    assert isinstance(batch.is_terminals, list)
    actions = torch.stack(batch.actions).to(device).detach()
    states = torch.stack(batch.states).to(device).detach()
    rewards = torch.tensor(batch.rewards).to(device).detach()
    logprobs = torch.stack(batch.logprobs).to(device).detach()
    is_terminals = torch.tensor(batch.is_terminals).to(device).detach()
    return states, actions, logprobs, rewards, is_terminals

@torch.no_grad()
def get_qref(batch, target_val_net, gamma, device="cpu"):
    # TODO I am not sure if this is maybe slightly different from original SAC where you take just non terminal states?
    _, batch_next_state, _, _, batch_reward, batch_done = batch
    target_next_batch = target_val_net(batch_next_state)
    
    batch_target_val = (
        batch_reward
        + (1 - batch_done)
        * gamma
        * torch.max(target_next_batch, dim=1, keepdim=True)[0]
    )
    return batch_target_val

@torch.no_grad()
def get_vref(env, batch, twinq_net, policy_net, ent_alpha: float, device="cpu"):
    
    assert isinstance(twinq_net, tuple)
    assert isinstance(env.action_space, spaces.Discrete)
    num_actions = env.action_space.n

    states, _, _, _, _, _= batch
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
