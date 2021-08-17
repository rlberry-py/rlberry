# Define import flags
TORCH_INSTALLED = True
try:
    import torch
except ModuleNotFoundError:
    TORCH_INSTALLED = False

NUMBA_INSTALLED = True
try:
    import numba
except ModuleNotFoundError:
    NUMBA_INSTALLED = False
