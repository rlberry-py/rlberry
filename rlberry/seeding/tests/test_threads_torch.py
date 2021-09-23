from rlberry.seeding.seeder import Seeder
from rlberry.seeding import set_external_seed
import concurrent.futures

_TORCH_INSTALLED = True
try:
    import torch
except Exception:
    _TORCH_INSTALLED = False


def get_torch_random_number_setting_seed(seeder):
    set_external_seed(seeder)
    return torch.randint(2 ** 32, (1,))[0].item()


def test_torch_multithread_seeding():
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
                        executor.submit(get_torch_random_number_setting_seed, seed)
                    )

                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(
                        future.result()
                    )
                assert results[0] != results[1], f"error in simulation {(ii, jj)}"
