# Define import flags
TORCH_INSTALLED = True
try:
    import torch
except ModuleNotFoundError:
    TORCH_INSTALLED = False

TENSORBOARD_INSTALLED = True
try:
    import torch.utils.tensorboard
except ModuleNotFoundError:
    TENSORBOARD_INSTALLED = False

NUMBA_INSTALLED = True
try:
    import numba
except ModuleNotFoundError:
    NUMBA_INSTALLED = False
