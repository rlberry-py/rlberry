from rlberry.manager import tensorboard_folder_to_dataframe
from stable_baselines3 import PPO, A2C
import tempfile
import os
import pandas as pd


def test_tensorboard_folder_to_dataframe():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # create data to test
        path_ppo = str(tmpdirname + "/ppo_cartpole_tensorboard/")
        path_a2c = str(tmpdirname + "/a2c_cartpole_tensorboard/")
        model = PPO("MlpPolicy", "CartPole-v1", tensorboard_log=path_ppo)
        model2 = A2C("MlpPolicy", "CartPole-v1", tensorboard_log=path_a2c)
        model.learn(total_timesteps=5_000, tb_log_name="ppo")
        model2.learn(total_timesteps=5_000, tb_log_name="A2C")

        assert os.path.exists(path_ppo)
        assert os.path.exists(path_a2c)

        data_in_dataframe = tensorboard_folder_to_dataframe(tmpdirname)

        assert isinstance(data_in_dataframe, dict)
        assert "rollout/ep_rew_mean" in data_in_dataframe
        a_dict = data_in_dataframe["rollout/ep_rew_mean"]

        assert isinstance(a_dict, pd.DataFrame)
        assert "name" in a_dict.columns
        assert "n_simu" in a_dict.columns
        assert "x" in a_dict.columns
        assert "y" in a_dict.columns
