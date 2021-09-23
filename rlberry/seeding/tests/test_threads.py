from rlberry.seeding.seeder import Seeder
import concurrent.futures
import pytest


def get_random_number_setting_seed(seeder):
    return seeder.rng.integers(2 ** 32)


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
