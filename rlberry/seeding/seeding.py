import numpy as np
from numpy.random import SeedSequence, default_rng

# Define global seed sequence
_GLOBAL_ENTROPY = None
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

def set_global_seed(entropy=42):
    """
    rlberry has a global seed from which we can obtain different random number
    generators with (close to) independent outcomes.

    Important:

    In each new process/thread, set_global_seed must be called with a new seed.

    To do:
        Check (torch seeding):
        https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668

    Parameters
    ---------
    entropy: int, SeedSequence optional
        The entropy for creating the global SeedSequence, or a SeedSequence
    """
    global _GLOBAL_ENTROPY, _GLOBAL_SEED_SEQ, _GLOBAL_RNG

    if isinstance(entropy, SeedSequence):
        seedseq = entropy
        _GLOBAL_ENTROPY = seedseq.entropy
        _GLOBAL_SEED_SEQ = seedseq
    else:
        _GLOBAL_ENTROPY = entropy
        _GLOBAL_SEED_SEQ = SeedSequence(entropy)

    _GLOBAL_RNG = get_rng()

    # seed torch
    if _TORCH_INSTALLED:
        torch.manual_seed(_GLOBAL_SEED_SEQ.generate_state(1, dtype=np.uint32)[0])


def generate_uniform_seed():
    """
    Return a seed value using a global random number generator.
    """
    return _GLOBAL_RNG.integers(2**32).item()


def global_rng():
    """
    Returns the global random number generator.
    """
    return _GLOBAL_RNG


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


def spawn(n):
    """
    Spawn a list of SeedSequence from the global seed sequence.

    Parameters
    ----------
    n : int
        Number of seed sequences to spawn
    Returns
    -------
    seed_seq : list
        List of spawned seed sequences
    """
    return _GLOBAL_SEED_SEQ.spawn(n)


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
