from rlberry.envs.gym_make import atari_make


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
    from rlberry.agents.torch import PPOAgent
    from gymnasium.wrappers.record_video import RecordVideo
    import os
    from rlberry.envs.gym_make import atari_make
    from rlberry.agents.torch.utils.training import model_factory_from_env
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdirname:
        policy_mlp_configs = {
            "type": "MultiLayerPerceptron",  # A network architecture
            "layer_sizes": [16],  # Network dimensions
            "reshape": False,
            "is_policy": True,  # The network should output a distribution
            # over actions
        }

        critic_mlp_configs = {
            "type": "MultiLayerPerceptron",
            "layer_sizes": [16],
            "reshape": False,
            "out_size": 1,  # The critic network is an approximator of
            # a value function V: States -> |R
        }

        policy_configs = {
            "type": "ConvolutionalNetwork",  # A network architecture
            "activation": "RELU",
            "in_channels": 4,
            "in_height": 84,
            "in_width": 84,
            "head_mlp_kwargs": policy_mlp_configs,
            "transpose_obs": False,
            "is_policy": True,  # The network should output a distribution
        }

        critic_configs = {
            "type": "ConvolutionalNetwork",
            "layer_sizes": "RELU",
            "in_channels": 4,
            "in_height": 84,
            "in_width": 84,
            "head_mlp_kwargs": critic_mlp_configs,
            "transpose_obs": False,
            "out_size": 1,
        }

        tuned_agent = ExperimentManager(
            PPOAgent,  # The Agent class.
            (
                atari_make,
                dict(id="ALE/Breakout-v5"),
            ),  # The Environment to solve.
            init_kwargs=dict(  # Where to put the agent's hyperparameters
                batch_size=16,
                optimizer_type="ADAM",  # What optimizer to use for policy gradient descent steps.
                learning_rate=2.5e-4,  # Size of the policy gradient descent steps.
                policy_net_fn=model_factory_from_env,  # A policy network constructor
                policy_net_kwargs=policy_configs,  # Policy network's architecure
                value_net_fn=model_factory_from_env,  # A Critic network constructor
                value_net_kwargs=critic_configs,  # Critic network's architecure.
                n_envs=8,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.1,
                k_epochs=4,
                n_steps=32,
            ),
            fit_budget=1000,  # The number of interactions between the agent and the environment during training.
            eval_kwargs=dict(
                eval_horizon=500
            ),  # The number of interactions between the agent and the environment during evaluations.
            n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
            agent_name="PPO_tuned",  # The agent's name.
            output_dir=str(tmpdirname) + "/PPO_for_breakout",
        )

        tuned_agent.fit()

        env = atari_make("ALE/Breakout-v5", render_mode="rgb_array")
        env = RecordVideo(env, str(tmpdirname) + "/_video/temp")

        if "render_modes" in env.metadata:
            env.metadata["render.modes"] = env.metadata[
                "render_modes"
            ]  # bug with some 'gym' version

        observation, info = env.reset()
        for tt in range(3000):
            action = tuned_agent.get_agent_instances()[0].policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                break

        env.close()

        assert os.path.exists(str(tmpdirname) + "/_video/temp/rl-video-episode-0.mp4")
