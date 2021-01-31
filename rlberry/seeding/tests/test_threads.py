import rlberry.seeding as seeding
import concurrent.futures
import pytest
from joblib import Parallel, delayed


def get_random_number(arg=None):
    return seeding.generate_uniform_seed()


def get_random_number_setting_seed(global_seed):
    seeding.set_global_seed(global_seed)
    return seeding.generate_uniform_seed()


@pytest.mark.parametrize("give_seed", [True, False])
def test_multithread_seeding(give_seed):
    """
    Checks that different seeds are given to different threads,
    even if we don't call set_global_seed in each thread.
    """
    seeding.set_global_seed(123)
    for _ in range(10):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for seed in seeding.spawn(2):
                if give_seed:
                    futures.append(
                        executor.submit(get_random_number_setting_seed, seed)
                    )
                else:
                    futures.append(
                        executor.submit(get_random_number)
                    )

            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(
                    future.result()
                )

            assert results[0] != results[1]


# @pytest.mark.parametrize("backend", ['loky', 'threading', 'multiprocessing'])
# def test_joblib_seeding(backend):
#     """
#     Checks that different seeds are given to different joblib workers.

#     'loky': global seed is reset to default (42); outputs are the same for all workers.
#     'threading': global seed is the same (123) for both workers; outputs are different,
#     so the seeding module is shared.
#     'multiprocessing': global seed is the same (123) for both workers; outputs are equal,
#      so the seeding module is copied.

#     Conclusion: threading is ok for automatic seeding; the other backends are not!
#     """
#     seeding.set_global_seed(123)
#     workers_output = Parallel(n_jobs=4,
#                               verbose=5,
#                               backend=backend)(
#             delayed(get_random_number)(None) for _ in range(2))
#     print(workers_output)


@pytest.mark.parametrize("backend", ['loky', 'threading', 'multiprocessing'])
def test_joblib_seeding_giving_seed(backend):
    """
    Solves the problem of test_joblib_seeding() by setting global seed in each of the
    subprocesses/threads
    """
    seeding.set_global_seed(123)
    workers_output = Parallel(n_jobs=4,
                              verbose=5,
                              backend=backend)(
            delayed(get_random_number_setting_seed)(seed) for seed in seeding.spawn(2))
    assert workers_output[0] != workers_output[1]
