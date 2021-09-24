import concurrent.futures


def fit_stats(stats, save):
    stats.fit()
    if save:
        stats.save()
    return stats


class MultipleManagers:
    """
    Class to fit multiple AgentManager instances in parallel with multiple threads.
    """

    def __init__(self) -> None:
        super().__init__()
        self.instances = []

    def append(self, agent_manager):
        """
        Append new AgentManager instance.

        Parameters
        ----------
        agent_manager : AgentManager
        """
        self.instances.append(agent_manager)

    def run(self, save=False):
        """
        Fit AgentManager instances in parallel.

        Parameters
        ----------
        save: bool, default: False
            If true, save AgentManager intances immediately after fitting.
            AgentManager.save() is called.
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
        Pickle AgentManager instances and saves fit statistics in .csv files.
        The output folder is defined in each of the AgentManager instances.
        """
        for stats in self.instances:
            stats.save()

    @property
    def managers(self):
        return self.instances
