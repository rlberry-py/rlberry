from typing import Tuple
from copy import deepcopy
from rlberry.seeding import safe_reseed


import rlberry

logger = rlberry.logger


def process_env(env, seeder, copy_env=True):
    if isinstance(env, Tuple):
        constructor = env[0]
        if constructor is None:
            return None
        kwargs = env[1] or {}
        processed_env = constructor(**kwargs)
    else:
        if env is None:
            return None
        if copy_env:
            try:
                processed_env = deepcopy(env)
            except Exception as ex:
                raise RuntimeError("[Agent] Not possible to deepcopy env: " + str(ex))
        else:
            processed_env = env
    reseeded = safe_reseed(processed_env, seeder)
    if not reseeded:
        logger.warning("[Agent] Not possible to reseed environment.")
    return processed_env
