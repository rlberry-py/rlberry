from multiprocessing import Pool


def fit_stats(stats):
    stats.fit()
    return stats


class MultipleStats:
    """
    Class to fit multiple AgentStats instances in Parallel.
    """
    def __init__(self) -> None:
        super().__init__()
        self.instances = []

    def append(self, agent_stats):
        self.instances.append(agent_stats)

    def run(self, n_processes=4):
        with Pool(n_processes) as p:
            self.instances = p.map(fit_stats, self.instances)

    def save(self):
        for stats in self.instances:
            stats.save_results()
            stats.save()

    @property
    def allstats(self):
        return self.instances
