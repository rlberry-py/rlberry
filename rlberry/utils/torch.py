import os
import re
import shutil
from subprocess import check_output, run, PIPE
import numpy as np
import torch
import logging

import rlberry

logger = rlberry.logger


def get_gpu_memory_map():
    result = check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
    )
    return [int(x) for x in result.split()]


def least_used_device():
    """Get the  GPU device with most available memory."""
    if not torch.cuda.is_available():
        raise RuntimeError("cuda unavailable")

    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "nvidia-smi unavailable: \
cannot select device with most least memory used."
        )

    memory_map = get_gpu_memory_map()
    device_id = np.argmin(memory_map)
    logger.debug(
        f"Choosing GPU device: {device_id}, " f"memory used: {memory_map[device_id]}"
    )
    return torch.device("cuda:{}".format(device_id))


def choose_device(preferred_device, default_device="cpu"):
    """Choose torch device, use default if choice is not available.

    Parameters
    ----------
    preferred_device: str
        Torch device to be used (if available), e.g. "cpu", "cuda:0", "cuda:best".
        If "cuda:best", returns the least used device in the machine.
    default_device: str, default = "cpu"
        Default device if preferred_device is not available.
    """
    if preferred_device == "cuda:best":
        try:
            preferred_device = least_used_device()
        except RuntimeError:
            logger.debug(
                f"Could not find least used device (nvidia-smi might be missing), use cuda:0 instead"
            )
            if torch.cuda.is_available():
                return choose_device("cuda:0")
            else:
                return choose_device("cpu")
    try:
        torch.zeros((1,), device=preferred_device)  # Test availability
    except (RuntimeError, AssertionError) as e:
        logger.debug(
            f"Preferred device {preferred_device} unavailable ({e})."
            f"Switching to default {default_device}"
        )
        return default_device
    return preferred_device


def get_memory(pid=None):
    if not pid:
        pid = os.getpid()
    command = "nvidia-smi"
    result = run(
        command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True
    ).stdout
    m = re.findall(
        "\| *[0-9] *" + str(pid) + " *C *.*python.*? +([0-9]+).*\|",
        result,
        re.MULTILINE,
    )
    return [int(mem) for mem in m]
