from numpy.random import SeedSequence, default_rng

# Define global seed sequence
_GLOBAL_SEED = None
_GLOBAL_SEED_SEQ = None

# Global rng
_GLOBAL_RNG = None


#
# Seeding other libraries
#

_TORCH_INSTALLED = True

try:
    import torch
except Exception:
    _TORCH_INSTALLED = False


#
# Seeding functions
#

def set_global_seed(seed=42):
    """
    rlberry has a global seed from which we can obtain different random number
    generators with (close to) independent outcomes.

    Important:

    If the global seed is altered by the user, it should be done only once,
    after importing rlberry for the first time. This is to ensure that all
    random number generators are children of the same SeedSequence.

    To do:
        Check (torch seeding):
        https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    """
    global _GLOBAL_SEED, _GLOBAL_SEED_SEQ, _GLOBAL_RNG
    _GLOBAL_SEED = seed
    _GLOBAL_SEED_SEQ = SeedSequence(_GLOBAL_SEED)
    _GLOBAL_RNG = get_rng()

    # get state (for seeding)
    # rng_libs = get_rng()
    # state = rng_libs.__getstate__()['state']['state']

    # seed torch
    if _TORCH_INSTALLED:
        torch.manual_seed(seed)


def generate_uniform_seed():
    """
    Return a seed value using a global random number generator.
    """
    return _GLOBAL_RNG.integers(2**32).item()


def safe_reseed(object):
    """
    Calls object.reseed() method if available;
    If a object.seed() method is available, call object.seed(seed_val), where seed_val is returned by
    generate_uniform_seed().
    Otherwise, does nothing.

    Returns
    -------
    True if reseeding was done, False otherwise.
    """
    try:
        object.reseed()
        return True
    except AttributeError:
        seed_val = generate_uniform_seed()
        try:
            object.seed(seed_val)
            return True
        except AttributeError:
            return False


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
