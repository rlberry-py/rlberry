from numpy.random import SeedSequence, default_rng

# Define global seed sequence
_GLOBAL_SEED = 42
_GLOBAL_SEED_SEQ = SeedSequence(_GLOBAL_SEED)


def set_global_seed(seed):
    """
    rlberry has a global seed from which we can obtain different random number
    generators with (close to) independent outcomes.

    Important:

    If the global seed is altered by the user, it should be done only once,
    after importing rlberry for the first time. This is to ensure that all
    random number generators are children of the same SeedSequence.
    """
    global _GLOBAL_SEED, _GLOBAL_SEED_SEQ
    _GLOBAL_SEED = seed
    _GLOBAL_SEED_SEQ = SeedSequence(_GLOBAL_SEED)


def get_rng():
    """
    Get a random number generator (rng), from the global seed sequence.

    Returns
    -------
    rng : numpy.random._generator.Generator
        random number generator
    """
    # Spawn off 1 child SeedSequence
    child_seed = _GLOBAL_SEED_SEQ.spawn(1)
    rng = default_rng(child_seed[0])
    return rng
