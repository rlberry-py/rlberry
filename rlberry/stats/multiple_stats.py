import concurrent.futures


def fit_stats(stats, save):
    stats.fit()
    if save:
        stats.save()
    return stats


class MultipleStats:
    """
    Class to fit multiple AgentStats instances in parallel with multiple threads.
    """

    def __init__(self) -> None:
        super().__init__()
        self.instances = []

    def append(self, agent_stats):
        """
        Append new AgentStats instance.

        Parameters
        ----------
        agent_stats : AgentStats
        """
        self.instances.append(agent_stats)

    def run(self, save=False):
        """
        Fit AgentStats instances in parallel.

        Parameters
        ----------
        save: bool, default: False
            If true, save AgentStats intances immediately after fitting.
            AgentStats.save() is called.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for inst in self.instances:
                futures.append(
                    executor.submit(fit_stats, inst, save=save)
                )

            fitted_instances = []
            for future in concurrent.futures.as_completed(futures):
                fitted_instances.append(
                    future.result()
                )

            self.instances = fitted_instances

    def save(self):
        """
        Pickle AgentStats instances and saves fit statistics in .csv files.
        The output folder is defined in each of the AgentStats instances.
        """
        for stats in self.instances:
            stats.save()

    @property
    def allstats(self):
        return self.instances
