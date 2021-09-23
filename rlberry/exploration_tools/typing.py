import numpy as np

_TORCH_INSTALLED = True
try:
    import torch
except ImportError:
    _TORCH_INSTALLED = False


def _get_type(arg):
    if _TORCH_INSTALLED and isinstance(arg, torch.Tensor):
        return 'torch'
    elif isinstance(arg, np.ndarray):
        return 'numpy'
    else:
        return type(arg)


def process_type(arg, expected_type):
    """
    Utility function to preprocess numpy/torch arguments,
    according to a expected type.

    For instance, if arg is numpy and expected_type is torch,
    converts arg to torch.tensor.

    Parameters
    ----------
    expected_type: {'torch', 'numpy'}
        Desired type for output.
    """
    if arg is None:
        return None

    if expected_type == 'torch':
        assert _TORCH_INSTALLED, "expected_type is 'torch', but torch is not installed!"
        if isinstance(arg, torch.Tensor):
            return arg
        elif isinstance(arg, np.ndarray):
            return torch.from_numpy(arg)
        elif np.issubdtype(type(arg), np.number):
            return torch.tensor(arg)
        else:
            return arg
    elif expected_type == 'numpy':
        if isinstance(arg, np.ndarray):
            return arg
        elif _TORCH_INSTALLED and isinstance(arg, torch.Tensor):
            return arg.detach().cpu().numpy()
        else:
            return arg
    else:
        return arg


def preprocess_args(expected_type):
    """
    Utility decorator for methods to preprocess numpy/torch arguments,
    according to an expected type.

    Output type = input type of the first argument.

    For instance, if function args are numpy and expected_type is torch,
    converts function args to torch.tensor.

    Parameters
    ----------
    expected_type: {'torch', 'numpy'}
        Desired type for output.
    """

    def decorator(func):
        def inner(self, *args, **kwargs):
            processed_args = ()
            for ii, arg in enumerate(args):
                processed_args += (process_type(arg, expected_type),)
            output = func(self, *processed_args, **kwargs)
            # Process output according to first argument
            ouput_expected_type = _get_type(args[0])
            processed_output = process_type(output, ouput_expected_type)
            return processed_output

        return inner

    return decorator
