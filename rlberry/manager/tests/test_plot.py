import tempfile
import os
import numpy as np

from rlberry.wrappers import WriterWrapper
from rlberry.envs import GridWorld
from rlberry.manager import plot_writer_data, AgentManager
from rlberry.agents import UCBVIAgent


class VIAgent(UCBVIAgent):
    name = "UCBVIAgent"

    def __init__(self, env, **kwargs):
        UCBVIAgent.__init__(self, env, horizon=50, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


def test_plot_writer_data():
    with tempfile.TemporaryDirectory() as tmpdirname:
        env_ctor = GridWorld
        env_kwargs = dict(nrows=2, ncols=2, reward_at={(1, 1): 0.1, (2, 2): 1.0})

        agent = AgentManager(
            VIAgent,
            (env_ctor, env_kwargs),
            fit_budget=10,
            n_fit=3,
            output_dir=tmpdirname + "/rlberry_data",
        )
        agent.fit()

        def compute_reward(rewards):
            return np.cumsum(rewards)

        os.system("ls " + tmpdirname + "/rlberry_data/manager_data")

        # Plot of the cumulative reward.
        output = plot_writer_data(
            agent,
            tag="reward",
            preprocess_func=compute_reward,
            title="Cumulative Reward",
            show=False,
            savefig_fname=tmpdirname + "/test.png",
        )
        assert (
            os.path.getsize(tmpdirname + "/test.png") > 1000
        )  # check that the file is not empty
        assert len(output) > 1
