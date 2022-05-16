import torch


def _normalize(x, epsilon=1e-5):
    return (x - x.mean()) / (x.std() + epsilon)


def stable_kl_div(old_probs, new_probs, epsilon=1e-12):
    old_probs = old_probs.probs + epsilon
    new_probs = new_probs.probs + epsilon
    kl = new_probs * torch.log(new_probs) - new_probs * torch.log(old_probs)
    return kl
