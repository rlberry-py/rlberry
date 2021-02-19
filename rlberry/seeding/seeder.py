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

    def __init__(self, seed_seq=None):
        """
        Parameters
        ----------
        seed_seq : np.random.SeedSequence, rlberry.seeding.Seeder or int, default : None
            Seed sequence from which to spawn the random number generator.
            If None, generate random seed.
            If int, use as entropy for SeedSequence.
            If seeder, use seeder.seed_seq
        """
        super().__init__()
        if seed_seq is None:
            seed_seq = SeedSequence()
        elif isinstance(seed_seq, SeedSequence):
            seed_seq = seed_seq.spawn(1)[0]
        elif isinstance(seed_seq, Seeder):
            seed_seq = seed_seq.seed_seq.spawn(1)[0]
        else:  # integer
            seed_seq = SeedSequence(seed_seq)
        self.seed_seq = seed_seq

        child_seed = self.seed_seq.spawn(1)
        self.rng = default_rng(child_seed[0])

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
        # new seed sequence if None
        if seed_seq is None:
            seed_seq = SeedSequence()
        # spawn if SeedSequence
        elif isinstance(seed_seq, SeedSequence):
            seed_seq = seed_seq.spawn(1)[0]
        # spawn if Seeder
        elif isinstance(seed_seq, Seeder):
            seed_seq = seed_seq.seed_seq.spawn(1)[0]
        # new SeedSequence if integer
        else:
            seed_seq = SeedSequence(seed_seq)
        self.seed_seq = seed_seq
        self.rng = default_rng(self.seed_seq)

    def spawn(self, n=1):
        """
        Spawn a list of Seeder from the seed sequence of the object (self.seed_seq).

        Parameters
        ----------
        n : int
            Number of seed sequences to spawn
        Returns
        -------
        seed_seq : list
            List of spawned seed sequences, or a single Seeder if n=1.
        """
        seed_seq_list = self.seed_seq.spawn(n)
        spawned_seeders = [Seeder(seq) for seq in seed_seq_list]
        if len(spawned_seeders) == 1:
            spawned_seeders = spawned_seeders[0]
        return spawned_seeders

    def generate_state(self, n):
        return self.seed_seq.generate_state(n)
