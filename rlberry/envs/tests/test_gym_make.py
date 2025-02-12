from rlberry.envs.gym_make import atari_make
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def test_atari_make():
    wrappers_dict = dict(terminal_on_life_loss=True, frame_skip=8)
    env = atari_make(
        "ALE/Freeway-v5", render_mode="rgb_array", atari_SB3_wrappers_dict=wrappers_dict
    )
    assert "EpisodicLifeEnv" in str(env)
    assert "MaxAndSkipEnv" in str(env)
    assert "ClipRewardEnv" in str(env)
    assert env.render_mode == "rgb_array"

    wrappers_dict2 = dict(terminal_on_life_loss=False, frame_skip=0)
    env2 = atari_make(
        "ALE/Breakout-v5", render_mode="human", atari_SB3_wrappers_dict=wrappers_dict2
    )
    assert "EpisodicLifeEnv" not in str(env2)
    assert "MaxAndSkipEnv" not in str(env2)
    assert "ClipRewardEnv" in str(env2)
    assert env2.render_mode == "human"


def test_rendering_with_atari_make():
    from rlberry.manager import ExperimentManager

    from gymnasium.wrappers.rendering import RecordVideo
    import os
    from rlberry.envs.gym_make import atari_make

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdirname:
        from stable_baselines3 import (
            PPO,
        )  # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

        from rlberry.agents.stable_baselines import StableBaselinesAgent

        tuned_xp = ExperimentManager(
            StableBaselinesAgent,  # The Agent class.
            (
                atari_make,
                dict(id="ALE/Breakout-v5"),
            ),  # The Environment to solve.
            init_kwargs=dict(  # Where to put the agent's hyperparameters
                algo_cls=PPO,
                policy="MlpPolicy",
            ),
            fit_budget=1000,  # The number of interactions between the agent and the environment during training.
            eval_kwargs=dict(
                eval_horizon=500
            ),  # The number of interactions between the agent and the environment during evaluations.
            n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
            agent_name="PPO_tuned",  # The agent's name.
            output_dir=str(tmpdirname) + "/PPO_for_breakout",
        )

        tuned_xp.fit()

        env = atari_make("ALE/Breakout-v5", render_mode="rgb_array")
        env = RecordVideo(env, str(tmpdirname) + "/_video/temp")

        if "render_modes" in env.metadata:
            env.metadata["render.modes"] = env.metadata[
                "render_modes"
            ]  # bug with some 'gym' version

        observation, info = env.reset()
        for tt in range(3000):
            action = tuned_xp.get_agent_instances()[0].policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                break

        env.close()

        assert os.path.exists(str(tmpdirname) + "/_video/temp/rl-video-episode-0.mp4")
