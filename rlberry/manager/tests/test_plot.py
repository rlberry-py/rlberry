import tempfile
import os
import numpy as np
from pathlib import Path


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
        ), "plot_writer_data output an empty image"
        assert len(output) > 1

        output2 = plot_writer_data(
            tmpdirname + "/rlberry_data",
            tag="reward",
            preprocess_func=compute_reward,
            title="Cumulative Reward",
            show=False,
            savefig_fname=tmpdirname + "/test.png",
        )
        assert np.all(output.shape == output2.shape)

        list_dirs = list(Path(tmpdirname + "/rlberry_data/manager_data").iterdir())
        list_dirs = [str(dir) for dir in list_dirs]

        output3 = plot_writer_data(
            list_dirs,
            tag="reward",
            preprocess_func=compute_reward,
            title="Cumulative Reward",
            show=False,
            savefig_fname=tmpdirname + "/test.png",
        )

        assert np.all(output.shape == output3.shape)
