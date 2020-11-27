import torch
from torch.nn import functional as F


def loss_function_factory(loss_function):
    if loss_function == "l2":
        return F.mse_loss
    elif loss_function == "l1":
        return F.l1_loss
    elif loss_function == "smooth_l1":
        return F.smooth_l1_loss
    elif loss_function == "bce":
        return F.binary_cross_entropy
    else:
        raise ValueError("Unknown loss function : {}".format(loss_function))


def optimizer_factory(optimizer_type, params, lr=None, weight_decay=None, k=None, **kwargs):
    if optimizer_type == "ADAM":
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "RMS_PROP":
        return torch.optim.RMSprop(params=params, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optimizer_type))
