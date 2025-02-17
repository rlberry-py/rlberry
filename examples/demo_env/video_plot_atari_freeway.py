"""
===============================================
A demo of ATARI Freeway environment with DQNAgent
===============================================
Illustration of the training and video rendering of DQN Agent in
ATARI Freeway environment.

Agent is slightly tuned, but not optimal. This is just for illustration purpose.

.. video:: ../../video_plot_atari_freeway.mp4
   :width: 600

"""

# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_atari_freeway.jpg'


from rlberry.manager import ExperimentManager
from datetime import datetime
from rlberry_research.agents.torch.dqn.dqn import DQNAgent
from gymnasium.wrappers.record_video import RecordVideo
import shutil
import os
from rlberry.envs.gym_make import atari_make


initial_time = datetime.now()
print("-------- init agent --------")

mlp_configs = {
    "type": "MultiLayerPerceptron",  # A network architecture
    "layer_sizes": [512],  # Network dimensions
    "reshape": False,
    "is_policy": False,  # The network should output a distribution
    # over actions
}

cnn_configs = {
    "type": "ConvolutionalNetwork",  # A network architecture
    "activation": "RELU",
    "in_channels": 4,
    "in_height": 84,
    "in_width": 84,
    "head_mlp_kwargs": mlp_configs,
    "transpose_obs": False,
    "is_policy": False,  # The network should output a distribution
}

tuned_xp = ExperimentManager(
    DQNAgent,  # The Agent class.
    (
        atari_make,
        dict(
            id="ALE/Freeway-v5",
        ),
    ),  # The Environment to solve.
    init_kwargs=dict(  # Where to put the agent's hyperparameters
        q_net_constructor="rlberry_research.agents.torch.utils.training.model_factory_from_env",
        q_net_kwargs=cnn_configs,
        max_replay_size=50000,
        batch_size=32,
        learning_starts=25000,
        gradient_steps=1,
        epsilon_final=0.01,
        learning_rate=1e-4,  # Size of the policy gradient descent steps.
        chunk_size=1,
    ),
    fit_budget=90000,  # The number of interactions between the agent and the environment during training.
    eval_kwargs=dict(
        eval_horizon=500
    ),  # The number of interactions between the agent and the environment during evaluations.
    n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
    agent_name="DQN_tuned",  # The agent's name.
    output_dir="DQN_for_freeway",
)

print("-------- init agent : done!--------")
print("-------- train agent --------")

tuned_xp.fit()

print("-------- train agent : done!--------")

final_train_time = datetime.now()

print("-------- test agent with video--------")

env = atari_make("ALE/Freeway-v5", render_mode="rgb_array")
env = RecordVideo(env, "_video/temp")

if "render_modes" in env.metadata:
    env.metadata["render.modes"] = env.metadata[
        "render_modes"
    ]  # bug with some 'gym' version

observation, info = env.reset()
for tt in range(30000):
    action = tuned_xp.get_agent_instances()[0].policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        break

env.close()

print("-------- test agent with video : done!--------")
final_test_time = datetime.now()
tuned_xp.save()

# need to move the final result inside the folder used for documentation
os.rename("_video/temp/rl-video-episode-0.mp4", "_video/video_plot_atari_freeway.mp4")
shutil.rmtree("_video/temp/")


print("Done!!!")
print("-------------")
print("begin run at :" + str(initial_time))
print("end training at :" + str(final_train_time))
print("end run at :" + str(final_test_time))
print("-------------")
