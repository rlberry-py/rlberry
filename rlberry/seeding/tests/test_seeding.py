from rlberry.seeding import Seeder


def test_seeder_basic():
    seeder1 = Seeder(43)
    data1 = seeder1.rng.integers(100, size=1000)

    seeder2 = Seeder(44)
    data2 = seeder2.rng.integers(100, size=1000)

    seeder3 = Seeder(44)
    data3 = seeder3.rng.integers(100, size=1000)

    assert (data1 != data2).sum() > 5
    assert (data2 != data3).sum() == 0
    assert seeder2.spawn(1).generate_state(1)[0] == seeder3.spawn(1).generate_state(1)[0]
    assert seeder1.spawn(1).generate_state(1)[0] != seeder3.spawn(1).generate_state(1)[0]


def test_seeder_initialized_from_seeder():
    """
    Check that Seeder(seed_seq) respawns seed_seq in the constructor.
    """
    seeder1 = Seeder(43)
    seeder_temp = Seeder(43)
    seeder2 = Seeder(seeder_temp)

    data1 = seeder1.rng.integers(100, size=1000)
    data2 = seeder2.rng.integers(100, size=1000)
    assert (data1 != data2).sum() > 5


def test_seeder_spawning():
    """
    Check that Seeder(seed_seq) respawns seed_seq in the constructor.
    """
    seeder1 = Seeder(43)
    seeder2 = seeder1.spawn()
    seeder3 = seeder2.spawn()

    print(seeder1)
    print(seeder2)
    print(seeder3)

    data1 = seeder1.rng.integers(100, size=1000)
    data2 = seeder2.rng.integers(100, size=1000)
    assert (data1 != data2).sum() > 5


def test_seeder_reseeding():
    """
    Check that reseeding with a Seeder instance works properly.
    """
    # seeders 1 and 2 are identical
    seeder1 = Seeder(43)
    seeder2 = Seeder(43)

    # reseed seeder 2 using seeder 1
    seeder2.reseed(seeder1)

    data1 = seeder1.rng.integers(100, size=1000)
    data2 = seeder2.rng.integers(100, size=1000)
    assert (data1 != data2).sum() > 5
