from rlberry.seeding.seeder import Seeder
import concurrent.futures
import pytest
from joblib import Parallel, delayed


def get_random_number_setting_seed(seeder):
    return seeder.rng.integers(2**32)


def test_multithread_seeding():
    """
    Checks that different seeds are given to different threads
    """
    for ii in range(5):
        main_seeder = Seeder(123)
        for jj in range(10):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for seed in main_seeder.spawn(2):
                    futures.append(
                        executor.submit(get_random_number_setting_seed, seed)
                    )

                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(
                        future.result()
                    )
                assert results[0] != results[1], f"error in simulation {(ii, jj)}"


@pytest.mark.parametrize("backend", ['loky', 'threading', 'multiprocessing'])
def test_joblib_seeding_giving_seed(backend):
    """
    Solves the problem of test_joblib_seeding() by setting global seed in each of the
    subprocesses/threads
    """
    main_seeder = Seeder(123)
    workers_output = Parallel(n_jobs=4,
                              verbose=5,
                              backend=backend)(
            delayed(get_random_number_setting_seed)(seed) for seed in main_seeder.spawn(2))
    assert workers_output[0] != workers_output[1]
