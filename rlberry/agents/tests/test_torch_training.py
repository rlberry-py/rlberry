from rlberry.agents.utils.torch_training import *


# def loss_function_factory(loss_function):
#     if loss_function == "l2":
#         return F.mse_loss
#     elif loss_function == "l1":
#         return F.l1_loss
#     elif loss_function == "smooth_l1":
#         return F.smooth_l1_loss
#     elif loss_function == "bce":
#         return F.binary_cross_entropy
#     else:
#         raise ValueError("Unknown loss function : {}".format(loss_function))
#

print(loss_function_factory("l2"))
