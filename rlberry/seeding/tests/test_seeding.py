from rlberry.seeding import Seeder


def test_seeder():
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
