"""
===============================================
A demo of ATARI Atlantis environment with vectorized PPOAgent
===============================================
Illustration of the training and video rendering of PPO Agent in
ATARI Atlantis environment.

Agent is slightly tuned, but not optimal. This is just for illustration purpose.

.. video:: ../../example_plot_atari_atlantis_vectorized_ppo.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/example_plot_atari_atlantis_vectorized_ppo.jpg'


import os
import shutil
from datetime import datetime

from gymnasium.wrappers.record_video import RecordVideo

from rlberry.agents.torch import PPOAgent
from rlberry.agents.torch.utils.training import model_factory_from_env
from rlberry.envs.gym_make import atari_make
from rlberry.manager import ExperimentManager

initial_time = datetime.now()
print("-------- init agent --------")


policy_mlp_configs = {
    "type": "MultiLayerPerceptron",  # A network architecture
    "layer_sizes": [512],  # Network dimensions
    "reshape": False,
    "is_policy": True,  # The network should output a distribution
    # over actions
}

critic_mlp_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": [512],
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
        dict(id="ALE/Atlantis-v5"),
    ),  # The Environment to solve.
    init_kwargs=dict(  # Where to put the agent's hyperparameters
        batch_size=64,
        optimizer_type="ADAM",  # What optimizer to use for policy gradient descent steps.
        learning_rate=1e-4,  # Size of the policy gradient descent steps.
        policy_net_fn=model_factory_from_env,  # A policy network constructor
        policy_net_kwargs=policy_configs,  # Policy network's architecure
        value_net_fn=model_factory_from_env,  # A Critic network constructor
        value_net_kwargs=critic_configs,  # Critic network's architecure.
        n_envs=7,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        k_epochs=10,
        n_steps=1024,
    ),
    fit_budget=10_000_000,  # The number of interactions between the agent and the environment during training.
    eval_kwargs=dict(
        eval_horizon=500
    ),  # The number of interactions between the agent and the environment during evaluations.
    n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
    agent_name="PPO_tuned",  # The agent's name.
    output_dir="PPO_for_atlantis",
)

print("-------- init agent : done!--------")
print("-------- train agent --------")

tuned_agent.fit()

print("-------- train agent : done!--------")

final_train_time = datetime.now()

print("-------- test agent with video--------")

env = atari_make("ALE/Atlantis-v5", render_mode="rgb_array")
env = RecordVideo(env, "_video/temp")

if "render_modes" in env.metadata:
    env.metadata["render.modes"] = env.metadata[
        "render_modes"
    ]  # bug with some 'gym' version

observation, info = env.reset()
for tt in range(30000):
    action = tuned_agent.get_agent_instances()[0].policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        break

env.close()

print("-------- test agent with video : done!--------")
final_test_time = datetime.now()
tuned_agent.save()

# need to move the final result inside the folder used for documentation
os.rename(
    "_video/temp/rl-video-episode-0.mp4",
    "_video/example_plot_atari_atlantis_vectorized_ppo.mp4",
)
shutil.rmtree("_video/temp/")


print("Done!!!")
print("-------------")
print("begin run at :" + str(initial_time))
print("end training at :" + str(final_train_time))
print("end run at :" + str(final_test_time))
print("-------------")
