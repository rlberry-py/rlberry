import rlberry.seeding as seeding


def test_seeding():
    seed = 123
    seeding.set_global_seed(seed)

    # check that reimports do not cause problems
    import rlberry
    import rlberry.seeding
    #

    assert seeding._GLOBAL_SEED_SEQ.entropy == seed

    _ = seeding.get_rng()
    assert seeding._GLOBAL_SEED_SEQ.n_children_spawned == 1

    # check that reimports do not cause problems
    import rlberry
    import rlberry.seeding
    assert seeding._GLOBAL_SEED_SEQ.entropy == seed
    #

    _ = seeding.get_rng()
    assert seeding._GLOBAL_SEED_SEQ.n_children_spawned == 2


def test_random_numbers():
    seed = 43
    seeding.set_global_seed(seed)
    rng1 = seeding.get_rng()
    data1 = rng1.integers(100, size=1000)

    seed = 44
    seeding.set_global_seed(seed)
    rng2 = seeding.get_rng()
    data2 = rng2.integers(100, size=1000)

    seed = 44
    seeding.set_global_seed(seed)
    rng3 = seeding.get_rng()
    data3 = rng3.integers(100, size=1000)

    assert (data1 != data2).sum() > 5
    assert (data2 != data3).sum() == 0
