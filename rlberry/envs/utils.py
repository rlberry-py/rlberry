from typing import Tuple
from copy import deepcopy
from rlberry.seeding import safe_reseed
import logging

logger = logging.getLogger(__name__)


def process_env(env, seeder, copy_env=True):
    if isinstance(env, Tuple):
        constructor = env[0]
        kwargs = env[1] or {}
        processed_env = constructor(**kwargs)
    else:
        if copy_env:
            try:
                processed_env = deepcopy(env)
            except Exception as ex:
                logger.warning("[Agent] Not possible to deepcopy env: " + str(ex))
        else:
            processed_env = env
    reseeded = safe_reseed(processed_env, seeder)
    if not reseeded:
        logger.warning("[Agent] Not possible to reseed environment.")
    return processed_env
