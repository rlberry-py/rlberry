import concurrent.futures
import functools
import multiprocessing
from typing import Optional


def fit_stats(stats, save):
    stats.fit()
    if save:
        stats.save()
    return stats


class MultipleManagers:
    """
    Class to fit multiple ExperimentManager instances in parallel with multiple threads.

    Parameters
    ----------
    max_workers: int, default=None
        max number of workers (ExperimentManager instances) fitted at the same time.
    parallelization: {'thread', 'process'}, default: 'process'
        Whether to parallelize  agent training using threads or processes.
    mp_context: {'spawn', 'fork', 'forkserver'}, default: 'spawn'.
        Context for python multiprocessing module.
        Warning: If you're using JAX or PyTorch, it only works with 'spawn'.
                 If running code on a notebook or interpreter, use 'fork'.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        parallelization: str = "process",
        mp_context="spawn",
    ) -> None:
        super().__init__()
        self.instances = []
        self.max_workers = max_workers
        self.parallelization = parallelization
        self.mp_context = mp_context

    def append(self, agent_manager):
        """
        Append new ExperimentManager instance.

        Parameters
        ----------
        agent_manager : ExperimentManager
        """
        self.instances.append(agent_manager)

    def run(self, save=True):
        """
        Fit ExperimentManager instances in parallel.

        Parameters
        ----------
        save: bool, default: True
            If true, save ExperimentManager intances immediately after fitting.
            ExperimentManager.save() is called.
        """
        if self.parallelization == "thread":
            executor_class = concurrent.futures.ThreadPoolExecutor
        elif self.parallelization == "process":
            executor_class = functools.partial(
                concurrent.futures.ProcessPoolExecutor,
                mp_context=multiprocessing.get_context(self.mp_context),
            )
        else:
            raise ValueError(
                f"Invalid backend for parallelization: {self.parallelization}"
            )

        with executor_class(max_workers=self.max_workers) as executor:
            futures = []
            for inst in self.instances:
                futures.append(executor.submit(fit_stats, inst, save=save))

            fitted_instances = []
            for future in concurrent.futures.as_completed(futures):
                fitted_instances.append(future.result())

            self.instances = fitted_instances

    def save(self):
        """
        Pickle ExperimentManager instances and saves fit statistics in .csv files.
        The output folder is defined in each of the ExperimentManager instances.
        """
        for stats in self.instances:
            stats.save()

    @property
    def managers(self):
        return self.instances
