import numpy as np
from numpy.random import SeedSequence, default_rng

# Define global seed sequence
_GLOBAL_SEED     = 42
_GLOBAL_SEED_SEQ = SeedSequence(_GLOBAL_SEED)

def set_global_seed(seed):
    """
    rlberry has a global seed from which we can obtain different random number
    generators with (close to) independent outcomes.

    If the global seed is altered by the user, it has to be done only once,
    after importing the rlberry
    """
    _GLOBAL_SEED = seed
    _GLOBAL_SEED_SEQ = SeedSequence(_GLOBAL_SEED)

def get_rng():
    """
    Get a random number generator (rng), from the global seed sequence.
    """
    # Spawn off 1 child SeedSequence
    child_seed = _GLOBAL_SEED_SEQ.spawn(1)
    return default_rng(child_seed[0])
