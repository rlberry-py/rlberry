from rlberry.manager import tensorboard_to_dataframe
from stable_baselines3 import PPO, A2C
import tempfile
import os
import pandas as pd


def test_tensorboard_to_dataframe():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # create data to test
        path_ppo = str(tmpdirname + "/ppo_cartpole_tensorboard/")
        path_a2c = str(tmpdirname + "/a2c_cartpole_tensorboard/")
        model = PPO("MlpPolicy", "CartPole-v1", tensorboard_log=path_ppo)
        model2 = A2C("MlpPolicy", "CartPole-v1", tensorboard_log=path_a2c)
        model2_seed2 = A2C("MlpPolicy", "CartPole-v1", tensorboard_log=path_a2c)
        model.learn(total_timesteps=5_000, tb_log_name="ppo")
        model2.learn(total_timesteps=5_000, tb_log_name="A2C")
        model2_seed2.learn(total_timesteps=5_000, tb_log_name="A2C")

        assert os.path.exists(path_ppo)
        assert os.path.exists(path_a2c)

        # check with parent folder
        data_in_dataframe = tensorboard_to_dataframe(tmpdirname)

        assert isinstance(data_in_dataframe, dict)
        assert "rollout/ep_rew_mean" in data_in_dataframe
        a_dict = data_in_dataframe["rollout/ep_rew_mean"]

        assert isinstance(a_dict, pd.DataFrame)
        assert "name" in a_dict.columns
        assert "n_simu" in a_dict.columns
        assert "x" in a_dict.columns
        assert "y" in a_dict.columns

        # check with list of folder
        folder_ppo_1 = str(path_ppo + "ppo_1/")
        folder_A2C_1 = str(path_a2c + "A2C_1/")
        folder_A2C_2 = str(path_a2c + "A2C_2/")

        path_event_ppo_1 = str(folder_ppo_1 + os.listdir(folder_ppo_1)[0])
        path_event_A2C_1 = str(folder_A2C_1 + os.listdir(folder_A2C_1)[0])
        path_event_A2C_2 = str(folder_A2C_2 + os.listdir(folder_A2C_2)[0])

        input_dict = {
            "ppo_cartpole_tensorboard": [path_event_ppo_1],
            "a2c_cartpole_tensorboard": [path_event_A2C_1, path_event_A2C_2],
        }

        data_in_dataframe2 = tensorboard_to_dataframe(input_dict)
        assert isinstance(data_in_dataframe2, dict)
        assert "rollout/ep_rew_mean" in data_in_dataframe2
        a_dict2 = data_in_dataframe2["rollout/ep_rew_mean"]

        assert isinstance(a_dict2, pd.DataFrame)
        assert "name" in a_dict2.columns
        assert "n_simu" in a_dict2.columns
        assert "x" in a_dict2.columns
        assert "y" in a_dict2.columns

        # check both strategies give the same result
        assert set(a_dict.keys()) == set(a_dict2.keys())
        for key in a_dict:
            if (
                key != "n_simu"
            ):  # seed will be different because one come from the folder name, and the other come for the index in the list
                print(key)
                assert a_dict[key].equals(a_dict2[key])
