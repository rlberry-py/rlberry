(agent_page)=

# How to use an Agent
In Reinforcement learning, the Agent is the entity to train to solve an environment. It's able to interact with the environment: observe, take actions, and learn through trial and error.
In rlberry, you can use existing Agent, or create your own custom Agent. You can find the API [here](/api) and [here](rlberry.agents.Agent) .


## Use rlberry Agent
An agent needs an environment to train. We'll use the same environment as in the [environment](environment_page) section of the user guide.
("Chain" environment from "[rlberry-research](https://github.com/rlberry-py/rlberry-research)")

### without agent
```python
from rlberry_research.envs.finite import Chain

env = Chain(10, 0.1)
env.enable_rendering()
for tt in range(50):
    env.step(env.action_space.sample())
env.render(loop=False)

# env.save_video is only available for rlberry envs and custom env (with 'RenderInterface' as parent class)
video = env.save_video("_agent_page_chain1.mp4")
env.close()
```
</br>

<video controls="controls" style="max-width: 600px;">
   <source src="../../user_guide_video/_agent_page_chain1.mp4" type="video/mp4">
</video>

If we use random actions on this environment, we don't have good results (the cross don't go to the right)

### With agent

With the same environment, we will use an Agent to choose the actions instead of random actions.
For this example, you can use "ValueIterationAgent" Agent from "[rlberry-scool](https://github.com/rlberry-py/rlberry-scool)"

```python
from rlberry_research.envs.finite import Chain
from rlberry_scool.agents.dynprog import ValueIterationAgent

env = Chain(10, 0.1)  # same env
agent = ValueIterationAgent(env, gamma=0.95)  # creation of the agent
info = agent.fit()  # Agent's training   (ValueIteration don't use budget)
print(info)

# test the trained agent
env.enable_rendering()
observation, info = env.reset()
for tt in range(50):
    action = agent.policy(
        observation
    )  # use the agent's policy to choose the next action
    observation, reward, terminated, truncated, info = env.step(action)  # do the action
    done = terminated or truncated
    if done:
        break  # stop if the environement is done
env.render(loop=False)

# env.save_video is only available for rlberry envs and custom env (with 'RenderInterface' as parent class)
video = env.save_video("_agent_page_chain2.mp4")
env.close()
```

```none
{'n_iterations': 269, 'precision': 1e-06}
  pg.display.set_mode(display, DOUBLEBUF | OPENGL)
  _ = pg.display.set_mode(display, DOUBLEBUF | OPENGL)
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)
  configuration: --prefix=/usr --extra-version=0ubuntu0.22.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 70.100 / 56. 70.100
  libavcodec     58.134.100 / 58.134.100
  libavformat    58. 76.100 / 58. 76.100
  libavdevice    58. 13.100 / 58. 13.100
  libavfilter     7.110.100 /  7.110.100
  libswscale      5.  9.100 /  5.  9.100
  libswresample   3.  9.100 /  3.  9.100
  libpostproc    55.  9.100 / 55.  9.100
Input #0, rawvideo, from 'pipe:':
  Duration: N/A, start: 0.000000, bitrate: 38400 kb/s
  Stream #0:0: Video: rawvideo (RGB[24] / 0x18424752), rgb24, 800x80, 38400 kb/s, 25 tbr, 25 tbn, 25 tbc
Stream mapping:
  Stream #0:0 -> #0:0 (rawvideo (native) -> h264 (libx264))
[libx264 @ 0x5570932967c0] using cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2 AVX512
[libx264 @ 0x5570932967c0] profile High, level 1.3, 4:2:0, 8-bit
[libx264 @ 0x5570932967c0] 264 - core 163 r3060 5db6aa6 - H.264/MPEG-4 AVC codec - Copyleft 2003-2021 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=2 lookahead_threads=1 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
Output #0, mp4, to '_agent_page_chain.mp4':
  Metadata:
    encoder         : Lavf58.76.100
  Stream #0:0: Video: h264 (avc1 / 0x31637661), yuv420p(tv, progressive), 800x80, q=2-31, 25 fps, 12800 tbn
    Metadata:
      encoder         : Lavc58.134.100 libx264
    Side data:
      cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A
frame=   51 fps=0.0 q=-1.0 Lsize=      12kB time=00:00:01.92 bitrate=  51.9kbits/s speed=48.8x
video:11kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 12.817029%
[libx264 @ 0x5570932967c0] frame I:1     Avg QP:29.06  size:  6089
[libx264 @ 0x5570932967c0] frame P:18    Avg QP:18.13  size:   172
[libx264 @ 0x5570932967c0] frame B:32    Avg QP:13.93  size:    37
[libx264 @ 0x5570932967c0] consecutive B-frames: 15.7%  0.0%  5.9% 78.4%
[libx264 @ 0x5570932967c0] mb I  I16..4: 46.4%  0.0% 53.6%
[libx264 @ 0x5570932967c0] mb P  I16..4:  5.9%  0.8%  1.3%  P16..4:  0.4%  0.0%  0.0%  0.0%  0.0%    skip:91.6%
[libx264 @ 0x5570932967c0] mb B  I16..4:  0.1%  0.0%  0.2%  B16..8:  1.6%  0.0%  0.0%  direct: 0.0%  skip:98.1%  L0:58.9% L1:41.1% BI: 0.0%
[libx264 @ 0x5570932967c0] 8x8 transform intra:6.3% inter:14.3%
[libx264 @ 0x5570932967c0] coded y,uvDC,uvAC intra: 46.1% 37.1% 35.7% inter: 0.0% 0.0% 0.0%
[libx264 @ 0x5570932967c0] i16 v,h,dc,p: 55%  7% 38%  1%
[libx264 @ 0x5570932967c0] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu:  0%  0% 100%  0%  0%  0%  0%  0%  0%
[libx264 @ 0x5570932967c0] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 33% 14% 41%  0%  4%  3%  2%  1%  1%
[libx264 @ 0x5570932967c0] i8c dc,h,v,p: 87%  5%  7%  1%
[libx264 @ 0x5570932967c0] Weighted P-Frames: Y:5.6% UV:5.6%
[libx264 @ 0x5570932967c0] ref P L0: 10.5%  5.3% 73.7% 10.5%
[libx264 @ 0x5570932967c0] ref B L0: 59.2% 27.6% 13.2%
[libx264 @ 0x5570932967c0] ref B L1: 96.2%  3.8%
[libx264 @ 0x5570932967c0] kb/s:40.59
```
</br>

<video controls="controls" style="max-width: 600px;">
   <source src="../../user_guide_video/_agent_page_chain2.mp4" type="video/mp4">
</video>

The agent has learned how to obtain good results (the cross go to the right).






## Use StableBaselines3 as rlberry Agent
With rlberry, you can use an algorithm from [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html) and wrap it in rlberry Agent. To do that, you need to use [StableBaselinesAgent](rlberry.agents.stable_baselines.StableBaselinesAgent).


```python
from rlberry.envs import gym_make
from gymnasium.wrappers.record_video import RecordVideo
from stable_baselines3 import PPO
from rlberry.agents.stable_baselines import StableBaselinesAgent

env = gym_make("CartPole-v1", render_mode="rgb_array")
agent = StableBaselinesAgent(
    env, PPO, "MlpPolicy", verbose=1
)  # wrap StableBaseline3's PPO inside rlberry Agent
info = agent.fit(10000)  # Agent's training
print(info)

env = RecordVideo(
    env, video_folder="./", name_prefix="CartPole"
)  # wrap the env to save the video output
observation, info = env.reset()  # initialize the environment
for tt in range(3000):
    action = agent.policy(
        observation
    )  # use the agent's policy to choose the next action
    observation, reward, terminated, truncated, info = env.step(action)  # do the action
    done = terminated or truncated
    if done:
        break  # stop if the environement is done
env.close()
```

```none
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 22       |
|    ep_rew_mean     | 22       |
| time/              |          |
|    fps             | 2490     |
|    iterations      | 1        |
|    time_elapsed    | 0        |
|    total_timesteps | 2048     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 28.1        |
|    ep_rew_mean          | 28.1        |
| time/                   |             |
|    fps                  | 1842        |
|    iterations           | 2           |
|    time_elapsed         | 2           |
|    total_timesteps      | 4096        |
| train/                  |             |
|    approx_kl            | 0.009214947 |
|    clip_fraction        | 0.102       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.686      |
|    explained_variance   | -0.00179    |
|    learning_rate        | 0.0003      |
|    loss                 | 8.42        |
|    n_updates            | 10          |
|    policy_gradient_loss | -0.0158     |
|    value_loss           | 51.5        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 40          |
|    ep_rew_mean          | 40          |
| time/                   |             |
|    fps                  | 1708        |
|    iterations           | 3           |
|    time_elapsed         | 3           |
|    total_timesteps      | 6144        |
| train/                  |             |
|    approx_kl            | 0.009872524 |
|    clip_fraction        | 0.0705      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.666      |
|    explained_variance   | 0.119       |
|    learning_rate        | 0.0003      |
|    loss                 | 16          |
|    n_updates            | 20          |
|    policy_gradient_loss | -0.0195     |
|    value_loss           | 38.7        |
-----------------------------------------
[INFO] 16:36: [[worker: -1]] | max_global_step = 6144 | time/iterations = 2 | rollout/ep_rew_mean = 28.13 | rollout/ep_len_mean = 28.13 | time/fps = 1842 | time/time_elapsed = 2 | time/total_timesteps = 4096 | train/learning_rate = 0.0003 | train/entropy_loss = -0.6860913151875139 | train/policy_gradient_loss = -0.015838009686558508 | train/value_loss = 51.528612112998964 | train/approx_kl = 0.009214947000145912 | train/clip_fraction = 0.10205078125 | train/loss = 8.420166969299316 | train/explained_variance = -0.001785874366760254 | train/n_updates = 10 | train/clip_range = 0.2 |
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 50.2         |
|    ep_rew_mean          | 50.2         |
| time/                   |              |
|    fps                  | 1674         |
|    iterations           | 4            |
|    time_elapsed         | 4            |
|    total_timesteps      | 8192         |
| train/                  |              |
|    approx_kl            | 0.0076105352 |
|    clip_fraction        | 0.068        |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.634       |
|    explained_variance   | 0.246        |
|    learning_rate        | 0.0003       |
|    loss                 | 29.6         |
|    n_updates            | 30           |
|    policy_gradient_loss | -0.0151      |
|    value_loss           | 57.3         |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 66          |
|    ep_rew_mean          | 66          |
| time/                   |             |
|    fps                  | 1655        |
|    iterations           | 5           |
|    time_elapsed         | 6           |
|    total_timesteps      | 10240       |
| train/                  |             |
|    approx_kl            | 0.006019583 |
|    clip_fraction        | 0.0597      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.606      |
|    explained_variance   | 0.238       |
|    learning_rate        | 0.0003      |
|    loss                 | 31.1        |
|    n_updates            | 40          |
|    policy_gradient_loss | -0.0147     |
|    value_loss           | 72.3        |
-----------------------------------------
None

Moviepy - Building video <yourPath>/CartPole-episode-0.mp4.
Moviepy - Writing video <yourPath>/CartPole-episode-0.mp4

Moviepy - Done !
Moviepy - video ready <yourPath>/CartPole-episode-0.mp4
```

</br>

<video controls="controls" style="max-width: 600px;">
   <source src="../../user_guide_video/_agent_page_CartPole.mp4" type="video/mp4">
</video>



## Create your own Agent
<span>&#9888;</span> **warning :** For advanced users only <span>&#9888;</span>

rlberry requires you to use a **very simple interface** to write agents, with basically
two methods to implement: `fit()` and `eval()`.

You can find more information on this interface [here(Agent)](rlberry.agents.agent.Agent)   (or [here(AgentWithSimplePolicy)](rlberry.agents.agent.AgentWithSimplePolicy))

The example below shows how to create an agent.


```python
import numpy as np
from rlberry.agents import AgentWithSimplePolicy


class MyAgentQLearning(AgentWithSimplePolicy):
    name = "QLearning"
    # create an agent with q-table

    def __init__(
        self,
        env,
        exploration_rate=0.01,
        learning_rate=0.8,
        discount_factor=0.95,
        **kwargs
    ):  # it's important to put **kwargs to ensure compatibility with the base class
        # self.env is initialized in the base class
        super().__init__(env=env, **kwargs)

        state_space_size = env.observation_space.n
        action_space_size = env.action_space.n

        self.exploration_rate = exploration_rate  # percentage to select random action
        self.q_table = np.zeros(
            (state_space_size, action_space_size)
        )  # q_table to store result and choose actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor  # gamma

    def fit(self, budget, **kwargs):
        """
        The parameter budget can represent the number of steps, the number of episodes etc,
        depending on the agent.
        * Interact with the environment (self.env);
        * Train the agent
        * Return useful information
        """
        n_episodes = budget
        rewards = np.zeros(n_episodes)

        for ep in range(n_episodes):
            observation, info = self.env.reset()
            done = False
            while not done:
                action = self.policy(observation)
                next_step, reward, terminated, truncated, info = self.env.step(action)
                # update the q_table
                self.q_table[observation, action] = (
                    1 - self.learning_rate
                ) * self.q_table[observation, action] + self.learning_rate * (
                    reward + self.discount_factor * np.max(self.q_table[next_step, :])
                )
                observation = next_step
                done = terminated or truncated
                rewards[ep] += reward

        info = {"episode_rewards": rewards}
        return info

    def eval(self, **kwargs):
        """
        Returns a value corresponding to the evaluation of the agent on the
        evaluation environment.

        For instance, it can be a Monte-Carlo evaluation of the policy learned in fit().
        """

        return super().eval()  # use the eval() from AgentWithSimplePolicy

    def policy(self, observation, explo=True):
        state = observation
        if explo and np.random.rand() < self.exploration_rate:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(self.q_table[state, :])  # Exploit

        return action
```


<span>&#9888;</span> **warning :**  It's important that your agent accepts optional `**kwargs` and pass it to the base class as `Agent.__init__(self, env, **kwargs)`. <span>&#9888;</span>

You can use it like this :

```python
from gymnasium.wrappers.record_video import RecordVideo
from rlberry.envs import gym_make

env = gym_make(
    "FrozenLake-v1", render_mode="rgb_array", is_slippery=False
)  # remove the slippery from the env
agent = MyAgentQLearning(
    env, exploration_rate=0.25, learning_rate=0.8, discount_factor=0.95
)
info = agent.fit(100000)  # Agent's training
print("----------")
print(agent.q_table)  # display the q_table content
print("----------")

env = RecordVideo(
    env, video_folder="./", name_prefix="FrozenLake_no_slippery"
)  # wrap the env to save the video output
observation, info = env.reset()  # initialize the environment
for tt in range(3000):
    action = agent.policy(
        observation, explo=False
    )  # use the agent's policy to choose the next action (without exploration)
    observation, reward, terminated, truncated, info = env.step(action)  # do the action
    done = terminated or truncated
    if done:
        break  # stop if the environement is done
env.close()
```


```none
----------
[[0.73509189 0.77378094 0.77378094 0.73509189]
 [0.73509189 0.         0.81450625 0.77378094]
 [0.77378094 0.857375   0.77378094 0.81450625]
 [0.81450625 0.         0.77378094 0.77378094]
 [0.77378094 0.81450625 0.         0.73509189]
 [0.         0.         0.         0.        ]
 [0.         0.9025     0.         0.81450625]
 [0.         0.         0.         0.        ]
 [0.81450625 0.         0.857375   0.77378094]
 [0.81450625 0.9025     0.9025     0.        ]
 [0.857375   0.95       0.         0.857375  ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.9025     0.95       0.857375  ]
 [0.9025     0.95       1.         0.9025    ]
 [0.         0.         0.         0.        ]]
----------

Moviepy - Building video <yourPath>/FrozenLake_no_slippery-episode-0.mp4.
Moviepy - Writing video <yourPath>/FrozenLake_no_slippery-episode-0.mp4

Moviepy - Done !
Moviepy - video ready <yourPath>/FrozenLake_no_slippery-episode-0.mp4
0.7

```


<video controls="controls" style="max-width: 600px;">
   <source src="../../user_guide_video/_agent_page_frozenLake.mp4" type="video/mp4">
</video>




## Use experimentManager

This is one of the core element in rlberry. The ExperimentManager allows you to easily make an experiment between an Agent and an Environment. It is used to train, optimize hyperparameters, evaluate and gather statistics about an agent.
You can find the guide for ExperimentManager [here](experimentManager_page).
