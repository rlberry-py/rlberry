"""
.. _checkpointing_example:

=============
Checkpointing
=============

This is a minimal example of how to create checkpoints while training
your agents, and how to restore from a previous checkpoint.
"""
from rlberry.agents import Agent
from rlberry.manager import AgentManager
from rlberry.manager import plot_writer_data


class MyAgent(Agent):
    name = "my-agent"

    def __init__(self, **kwargs):
        Agent.__init__(self, **kwargs)
        self.data = [0.0, 1.0]
        self.checkpoint_file = None
        self.total_timesteps = 0

    def fit(self, budget: int, **kwargs):
        """This agent is solving a simple difference equation,
        just to illustrate checkpoints :D"""
        del kwargs

        # check if there is a checkpoint to be loaded
        if self.checkpoint_file is not None:
            loaded_checkpoint = MyAgent.load(self.checkpoint_file, **self.get_params())
            self.__dict__.update(loaded_checkpoint.__dict__)
            print(f" \n --> MyAgent loaded from checkpoint: {self.checkpoint_file} \n")

        # run training loop
        for _ in range(budget):
            self.total_timesteps += 1
            yt = (2 * self.data[-1] - self.data[-2]) / (1 + 0.01**2)
            yt += 0.1 * self.rng.normal()
            self.data.append(yt)
            if self.writer:
                self.writer.add_scalar("y(t)", yt, self.total_timesteps)

            # Checkpoint every 500 timesteps
            if self.total_timesteps % 500 == 0:
                self.checkpoint_file = self.save(self.output_dir / "checkpoint.pickle")
                print(
                    f"checkpoint at {self.checkpoint_file} (timestep = {self.total_timesteps})"
                )

    def eval(self, **kwargs):
        del kwargs
        return self.data[-1]


if __name__ == "__main__":
    manager = AgentManager(
        MyAgent,
        fit_budget=-1,
        n_fit=2,
        seed=123,
    )
    # Save manager **before** fit for several timesteps! So that we can see why checkpoints are useful:
    # even if AgentManager is interrupted, it can be loaded and continue training
    # from last checkpoint.
    # But first, we need an initial call to AgentManager.fit() (for zero or a small number of timesteps),
    # so that AgentManager can instantiate MyAgent and it can start checkpointing itself.
    # This is because the __init__ method of MyAgent is only executed
    # after the first call to AgentManager.fit().
    manager.fit(0)
    manager_file = manager.save()
    print(f"\n Saved manager at {manager_file}.\n")

    # Fit for a 1000 timesteps
    manager.fit(1000)

    # Delete manager. This simulates a situation where we couldn't call
    # manager.save() after calling manager.fit(), e.g. when your process is interrupted
    # before .fit() returns.
    del manager

    # Load manager and continue training from last checkpoint
    print(f"\n Loading manager from {manager_file}.\n")
    loaded_manager = AgentManager.load(manager_file)
    # Fit for 500 more timesteps
    loaded_manager.fit(500)

    # The plot shows a total of 1500 timesteps!
    plot_writer_data(loaded_manager, tag="y(t)", show=True)
