from numpy.random import SeedSequence, default_rng


class Seeder:
    """
    Base class to define objects that use random number generators.

    See also:
    https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html

    Attributes
    ----------
    rng : numpy.random._generator.Generator
        random number generator provided by rlberry.seeding

    Methods
    -------
    reseed()
        get new random number generator
    """

    def __init__(self, seed_seq=None, spawn_seed_seq=True):
        """
        Parameters
        ----------
        seed_seq : np.random.SeedSequence, rlberry.seeding.Seeder or int, default : None
            Seed sequence from which to spawn the random number generator.
            If None, generate random seed.
            If int, use as entropy for SeedSequence.
            If seeder, use seeder.seed_seq
        spawn_seed_seq : bool, default : True
            If True, uses seed_seq to spawn a new seed sequence (strongly recommended) for the Seeder.
            If False, uses the input seed_seq to define the Seeder.
            Warning: Setting to false can lead to unexpected behavior. This argument is only used internally
            in rlberry, in Seeder.spawn(), to avoid unnecessary spawning.
        """
        super().__init__()
        if seed_seq is None:
            seed_seq = SeedSequence()
        elif isinstance(seed_seq, SeedSequence):
            seed_seq = seed_seq
        elif isinstance(seed_seq, Seeder):
            seed_seq = seed_seq.seed_seq
        else:  # integer
            seed_seq = SeedSequence(seed_seq)

        if spawn_seed_seq:
            seed_seq = seed_seq.spawn(1)[0]

        self.seed_seq = seed_seq
        self.rng = default_rng(self.seed_seq)

    def reseed(self, seed_seq=None):
        """
        Get new random number generator.

        Parameters
        ----------
        seed_seq : np.random.SeedSequence, rlberry.seeding.Seeder or int, default : None
            Seed sequence from which to spawn the random number generator.
            If None, generate random seed.
            If int, use as entropy for SeedSequence.
            If seeder, use seeder.seed_seq
        """
        # if None, new seed sequence
        if seed_seq is None:
            seed_seq = SeedSequence()
        # if SeedSequence, do nothing
        elif isinstance(seed_seq, SeedSequence):
            seed_seq = seed_seq
        # if Seeder, get Seeder.seed_seq
        elif isinstance(seed_seq, Seeder):
            seed_seq = seed_seq.seed_seq
        # if integer, new SeedSequence
        else:
            seed_seq = SeedSequence(seed_seq)

        # spawn
        seed_seq = seed_seq.spawn(1)[0]

        self.seed_seq = seed_seq
        self.rng = default_rng(self.seed_seq)

    def spawn(self, n=1, squeeze=True):
        """
        Spawn a list of Seeder from the seed sequence of the object (self.seed_seq).

        Parameters
        ----------
        n : int
            Number of seed sequences to spawn
        squeeze : bool
            If False, returns a list even if n=1. Otherwise,
            returns a Seeder if n=1.
        Returns
        -------
        seed_seq : list
            List of spawned seed sequences, or a single Seeder if n=1.
        """
        seed_seq_list = self.seed_seq.spawn(n)
        spawned_seeders = [Seeder(seq, spawn_seed_seq=False) for seq in seed_seq_list]
        if len(spawned_seeders) == 1 and squeeze:
            spawned_seeders = spawned_seeders[0]
        return spawned_seeders

    def generate_state(self, n):
        return self.seed_seq.generate_state(n)

    def __str__(self):
        return f"Seeder object with: {self.seed_seq.__str__()}"
