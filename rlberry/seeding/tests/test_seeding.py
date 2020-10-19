import rlberry.seeding as seeding

def test_seeding():
    seed = 123
    seeding.set_global_seed(seed)
    assert seeding._GLOBAL_SEED_SEQ.entropy == seed

    rng1 = seeding.get_rng()
    assert seeding._GLOBAL_SEED_SEQ.n_children_spawned == 1

    rng2 = seeding.get_rng()
    assert seeding._GLOBAL_SEED_SEQ.n_children_spawned == 2
