(environment_page)=

# How to use an environment

This is the world with which the agent interacts. The agent can observe this environment, and can perform actions to modify it (but cannot change its rules). With rlberry, you can use an existing environment, or create your own custom environment.


## Use rlberry environment
You can find some environments in our other projects "[rlberry-research](https://github.com/rlberry-py/rlberry-research)" and "[rlberry-scool](https://github.com/rlberry-py/rlberry-scool)".
For this example, you can use "Chain" environment from "[rlberry-research](https://github.com/rlberry-py/rlberry-research)"
```python
from rlberry_research.envs.finite import Chain

env = Chain(10, 0.1)
env.enable_rendering()
for tt in range(20):
    # Force to go right every 4 steps to have a better video render.
    if tt % 4 == 0:
        env.step(1)
    else:
        env.step(env.action_space.sample())
env.render(loop=False)

# env.save_video is only available for rlberry envs and custom env (with 'RenderInterface' as parent class)
video = env.save_video("_env_page_chain.mp4")
env.close()
```

```none
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
[libx264 @ 0x5644b31f07c0] using cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2 AVX512
[libx264 @ 0x5644b31f07c0] profile High, level 1.3, 4:2:0, 8-bit
[libx264 @ 0x5644b31f07c0] 264 - core 163 r3060 5db6aa6 - H.264/MPEG-4 AVC codec - Copyleft 2003-2021 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=2 lookahead_threads=1 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
Output #0, mp4, to '_env_page_chain.mp4':
  Metadata:
    encoder         : Lavf58.76.100
  Stream #0:0: Video: h264 (avc1 / 0x31637661), yuv420p(tv, progressive), 800x80, q=2-31, 25 fps, 12800 tbn
    Metadata:
      encoder         : Lavc58.134.100 libx264
    Side data:
      cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A
frame=   21 fps=0.0 q=-1.0 Lsize=      11kB time=00:00:00.72 bitrate= 128.4kbits/s speed=29.6x
video:10kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 10.128633%
[libx264 @ 0x5644b31f07c0] frame I:1     Avg QP:29.04  size:  6175
[libx264 @ 0x5644b31f07c0] frame P:12    Avg QP:24.07  size:   220
[libx264 @ 0x5644b31f07c0] frame B:8     Avg QP:22.19  size:   124
[libx264 @ 0x5644b31f07c0] consecutive B-frames: 33.3% 38.1% 28.6%  0.0%
[libx264 @ 0x5644b31f07c0] mb I  I16..4: 56.0%  0.0% 44.0%
[libx264 @ 0x5644b31f07c0] mb P  I16..4:  8.4%  1.6%  1.7%  P16..4:  1.1%  0.0%  0.0%  0.0%  0.0%    skip:87.3%
[libx264 @ 0x5644b31f07c0] mb B  I16..4:  0.2%  0.5%  0.9%  B16..8:  4.1%  0.0%  0.0%  direct: 0.1%  skip:94.3%  L0:48.8% L1:51.2% BI: 0.0%
[libx264 @ 0x5644b31f07c0] 8x8 transform intra:9.0% inter:23.5%
[libx264 @ 0x5644b31f07c0] coded y,uvDC,uvAC intra: 45.6% 37.1% 36.1% inter: 0.2% 0.0% 0.0%
[libx264 @ 0x5644b31f07c0] i16 v,h,dc,p: 51% 13% 35%  1%
[libx264 @ 0x5644b31f07c0] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu:  0%  0% 100%  0%  0%  0%  0%  0%  0%
[libx264 @ 0x5644b31f07c0] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 33% 16% 40%  1%  4%  2%  2%  1%  0%
[libx264 @ 0x5644b31f07c0] i8c dc,h,v,p: 91%  5%  4%  0%
[libx264 @ 0x5644b31f07c0] Weighted P-Frames: Y:8.3% UV:8.3%
[libx264 @ 0x5644b31f07c0] ref P L0: 21.9%  0.0% 65.6% 12.5%
[libx264 @ 0x5644b31f07c0] ref B L0: 52.5% 47.5%
[libx264 @ 0x5644b31f07c0] kb/s:93.39
```
</br>

<video controls="controls" style="max-width: 600px;">
   <source src="../../../../_video/user_guide_video/_env_page_chain.mp4" type="video/mp4">
</video>


## Use Gymnasium environment
Gymnasium can give you some classic environment. You can use theme with [gym_make](rlberry.envs.gym_make). More information [here](https://gymnasium.farama.org/environments/classic_control/).

```python
from rlberry.envs import gym_make
from gymnasium.wrappers.record_video import RecordVideo

# If you want an output video of your Gymnasium env, you have to :
# - add a 'render_mode' parameter at your gym_make
# - add a 'RecordVideo' wrapper around the gymnasium environment.
env = gym_make("MountainCar-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./", name_prefix="MountainCar")
# else, this line is enough:
# env = gym_make("MountainCar-v0")

observation, info = env.reset()
for tt in range(50):
    env.step(env.action_space.sample())
env.close()
```

```none
[your path]/.conda/lib/python3.10/site-packages/gymnasium/wrappers/record_video.py:94: UserWarning: WARN: Overwriting existing videos at [your path] folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)
  logger.warn(
[your path]/.conda/lib/python3.10/site-packages/gymnasium/core.py:311: UserWarning: WARN: env.is_vector_env to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.is_vector_env` for environment variables or `env.get_wrapper_attr('is_vector_env')` that will search the reminding wrappers.
  logger.warn(
Moviepy - Building video [your path]/MountainCar-episode-0.mp4.
Moviepy - Writing video [your path]/MountainCar-episode-0.mp4

Moviepy - Done !
Moviepy - video ready [your path]/MountainCar-episode-0.mp4
```

</br>

<video controls="controls" style="max-width: 600px;">
   <source src="../../../../_video/user_guide_video/_env_page_MountainCar.mp4" type="video/mp4">
</video>



## Use Atari environment
A set of Atari 2600 environment simulated through Stella and the Arcade Learning Environment. More information [here](https://gymnasium.farama.org/environments/atari/).

The function "[atari_make()](rlberry.envs.atari_make)" add wrappers on gym.make, to make it easier to use on Atari games.

```python
from rlberry.envs import atari_make
from gymnasium.wrappers.record_video import RecordVideo


# If you want an output video of your Atari env, you have to :
# - add a 'render_mode' parameter at your atari_make
# - add a 'RecordVideo' wrapper around the gymnasium environment.
env = atari_make("ALE/Breakout-v5", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./", name_prefix="Breakout")
# else, this line is enough:
# env = atari_make("ALE/Breakout-v5")

observation, info = env.reset()
for tt in range(50):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    # if the environment is terminated or truncated (no more life), it need to be reset
    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

```none
A.L.E: Arcade Learning Environment (version 0.8.1+53f58b7)
[Powered by Stella]
[your path]/.conda/lib/python3.10/site-packages/gymnasium/core.py:311: UserWarning: WARN: env.is_vector_env to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.is_vector_env` for environment variables or `env.get_wrapper_attr('is_vector_env')` that will search the reminding wrappers.
  logger.warn(
[your path]/.conda/lib/python3.10/site-packages/gymnasium/utils/passive_env_checker.py:335: UserWarning: WARN: No render fps was declared in the environment (env.metadata['render_fps'] is None or not defined), rendering may occur at inconsistent fps.
  logger.warn(
Moviepy - Building video [your path]/Breakout-episode-0.mp4.
Moviepy - Writing video [your path]/Breakout-episode-0.mp4

Moviepy - Done !
Moviepy - video ready [your path]/Breakout-episode-0.mp4
```

</br>

<video controls="controls" style="max-width: 600px;">
   <source src="../../../../_video/user_guide_video/_env_page_Breakout.mp4" type="video/mp4">
</video>


## Create your own environment
<span>&#9888;</span> **warning :** For advanced users only <span>&#9888;</span>

You need to create a new class that inherits from [gymnasium.Env](https://gymnasium.farama.org/api/env/) or one of it child class like [Model](rlberry.envs.interface.Model) (and RenderInterface/one of it child class, if you want an environment with rendering).

Then you need to make the specific functions that respect gymnasium template (as step, reset, ...). More information [here](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)

You can find examples in our other github project "[rlberry-research](https://github.com/rlberry-py/rlberry-research)" with [Acrobot](https://github.com/rlberry-py/rlberry-research/blob/main/rlberry_research/envs/classic_control/acrobot.py), [MountainCar](https://github.com/rlberry-py/rlberry-research/blob/main/rlberry_research/envs/classic_control/mountain_car.py) or [Chain](https://github.com/rlberry-py/rlberry-research/blob/main/rlberry_research/envs/finite/chain.py) (and their parent classes).
