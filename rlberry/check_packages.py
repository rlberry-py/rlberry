# Define import flags

TORCH_INSTALLED = True
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    TORCH_INSTALLED = False  # pragma: no cover

TENSORBOARD_INSTALLED = True
try:
    import torch.utils.tensorboard
except ModuleNotFoundError:  # pragma: no cover
    TENSORBOARD_INSTALLED = False  # pragma: no cover

NUMBA_INSTALLED = True
try:
    import numba
except ModuleNotFoundError:  # pragma: no cover
    NUMBA_INSTALLED = False  # pragma: no cover
