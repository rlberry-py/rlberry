"""
Based on https://github.com/openai/gym and then modified for our purpose.
"""

from typing import Union, Optional
import inspect

import gymnasium as gym
import numpy as np
from gymnasium import logger
from gymnasium import spaces


def _is_numpy_array_space(space: spaces.Space) -> bool:
    """
    Returns False if provided space is not representable as a single numpy array
    (e.g. Dict and Tuple spaces return False)
    """
    return not isinstance(space, (spaces.Dict, spaces.Tuple))


def _check_image_input(observation_space: spaces.Box, key: str = "") -> None:
    """
    Check that the input adheres to general standards
    when the observation is apparently an image.
    """

    if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
        logger.warn(
            f"It seems that your observation space {key} is an image but the "
            "upper and lower bounds are not in [0, 255]. "
            "Generally, CNN policies assume observations are within that range, "
            "so you may encounter an issue if the observation values are not."
        )


def _check_nan(env: gym.Env, check_inf: bool = True) -> None:
    """Check for NaN and Inf."""
    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, _, _, _ = env.step(action)

        if np.any(np.isnan(observation)):
            logger.warn("Encountered NaN value in observations.")
        if np.any(np.isnan(reward)):
            logger.warn("Encountered NaN value in rewards.")
        if check_inf and np.any(np.isinf(observation)):
            logger.warn("Encountered inf value in observations.")
        if check_inf and np.any(np.isinf(reward)):
            logger.warn("Encountered inf value in rewards.")


def _check_obs(
    obs: Union[tuple, dict, np.ndarray, int],
    observation_space: spaces.Space,
    method_name: str,
) -> None:
    """
    Check that the observation returned by the environment
    correspond to the declared one.
    """
    if not isinstance(observation_space, spaces.Tuple):
        assert not isinstance(
            obs, tuple
        ), f"The observation returned by the `{method_name}()` method should be a single value, not a tuple"

    if isinstance(observation_space, spaces.Discrete):
        assert np.isscalar(
            obs
        ), f"The observation returned by `{method_name}()` method must be a scalar"
    elif _is_numpy_array_space(observation_space):
        assert isinstance(
            obs, np.ndarray
        ), f"The observation returned by `{method_name}()` method must be a numpy array"

    assert observation_space.contains(
        obs
    ), f"The observation returned by the `{method_name}()` method does not match the given observation space"


def _check_box_obs(observation_space: spaces.Box, key: str = "") -> None:
    """
    Check that the observation space is correctly formatted
    when dealing with a ``Box()`` space. In particular, it checks:
    - that the dimensions are big enough when it is an image, and that the type matches
    - that the observation has an expected shape (warn the user if not)
    """
    # If image, check the low and high values, the type and the number of channels
    # and the shape (minimal value)
    if len(observation_space.shape) == 3:
        _check_image_input(observation_space)

    if np.any(np.equal(observation_space.low, observation_space.high)):
        logger.warn("Agent's maximum and minimum observation space values are equal")
    if np.any(np.greater(observation_space.low, observation_space.high)):
        assert False, "Agent's minimum observation value is greater than it's maximum"
    if observation_space.low.shape != observation_space.shape:
        assert (
            False
        ), "Agent's observation_space.low and observation_space have different shapes"
    if observation_space.high.shape != observation_space.shape:
        assert (
            False
        ), "Agent's observation_space.high and observation_space have different shapes"


def _check_box_action(action_space: spaces.Box):
    if np.any(np.equal(action_space.low, -np.inf)):
        logger.warn(
            "Agent's minimum action space value is -infinity. This is probably too low."
        )
    if np.any(np.equal(action_space.high, np.inf)):
        logger.warn(
            "Agent's maxmimum action space value is infinity. This is probably too high"
        )
    if np.any(np.equal(action_space.low, action_space.high)):
        logger.warn("Agent's maximum and minimum action space values are equal")
    if np.any(np.greater(action_space.low, action_space.high)):
        assert False, "Agent's minimum action value is greater than it's maximum"
    if action_space.low.shape != action_space.shape:
        assert False, "Agent's action_space.low and action_space have different shapes"
    if action_space.high.shape != action_space.shape:
        assert False, "Agent's action_space.high and action_space have different shapes"


def _check_returned_values(
    env: gym.Env, observation_space: spaces.Space, action_space: spaces.Space
) -> None:
    """
    Check the returned values by the env when calling `.reset()` or `.step()` methods.
    """
    # because env inherits from gymnasium.Env, we assume that `reset()` and `step()` methods exists
    observation, info = env.reset()

    if isinstance(observation_space, spaces.Dict):
        assert isinstance(
            observation, dict
        ), "The observation returned by `reset()` must be a dictionary"
        for key in observation_space.spaces.keys():
            try:
                _check_obs(observation[key], observation_space.spaces[key], "reset")
            except AssertionError as e:
                raise AssertionError(f"Error while checking key={key}: " + str(e))
    else:
        _check_obs(observation, observation_space, "reset")

    # Sample a random action
    action = action_space.sample()
    data = env.step(action)

    assert (
        len(data) == 5
    ), "The `step()` method must return four values: observation, reward, terminated, truncated, info "

    # Unpack
    observation, reward, terminated, truncated, info = data

    if isinstance(observation_space, spaces.Dict):
        assert isinstance(
            observation, dict
        ), "The observation returned by `step()` must be a dictionary"
        for key in observation_space.spaces.keys():
            try:
                _check_obs(observation[key], observation_space.spaces[key], "step")
            except AssertionError as e:
                raise AssertionError(f"Error while checking key={key}: " + str(e))

    else:
        _check_obs(observation, observation_space, "step")

    # We also allow int because the reward will be cast to float
    assert isinstance(
        reward, (float, int, np.float32)
    ), "The reward returned by `step()` must be a float"
    assert isinstance(
        terminated, bool
    ), "The `terminated` returned by `step()` must be a boolean"
    assert isinstance(
        truncated, bool
    ), "The `truncated` returned by `step()` must be a boolean"
    done = terminated or truncated
    assert isinstance(done, bool), "The `done` signal must be a boolean"
    assert isinstance(
        info, dict
    ), "The `info` returned by `step()` must be a python dictionary"


# Check render cannot be covered by CI
def _check_render(
    env: gym.Env, warn: bool = True, headless: bool = False
) -> None:  # pragma: no cover
    """
    Check the declared render modes and the `render()`/`close()`
    method of the environment.
    :param env: The environment to check
    :param warn: Whether to output additional warnings
    :param headless: Whether to disable render modes
        that require a graphical interface. False by default.
    """
    render_modes = env.metadata.get("render.modes")
    if render_modes is None:
        if warn:
            logger.warn(
                "No render modes was declared in the environment "
                " (env.metadata['render.modes'] is None or not defined), "
                "you may have trouble when calling `.render()`"
            )

    else:
        # Don't check render mode that require a
        # graphical interface (useful for CI)
        if headless and "human" in render_modes:
            render_modes.remove("human")
        # Check all declared render modes
        for render_mode in render_modes:
            env.render(mode=render_mode)
        env.close()


def _check_reset_seed(env: gym.Env, seed: Optional[int] = None) -> None:
    """
    Check that the environment can be reset with a random seed.
    """
    signature = inspect.signature(env.reset)

    seed_param = signature.parameters.get("seed")
    # Check the default value is None
    if seed_param is not None and seed_param.default is not None:
        logger.warn(
            "The default seed argument in reset should be `None`, "
            "otherwise the environment will by default always be deterministic"
        )


def check_gym_env(
    env: gym.Env, warn: bool = True, skip_render_check: bool = True
) -> None:
    """
    Check that an environment follows Gymnasium API.
    This is particularly useful when using a custom environment.
    Please take a look at https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py
    for more information about the API.
    It also optionally check that the environment is compatible with Stable-Baselines.
    :param env: The Gym environment that will be checked
    :param warn: Whether to output additional warnings
        mainly related to the interaction with Stable Baselines
    :param skip_render_check: Whether to skip the checks for the render method.
        True by default (useful for the CI)
    """
    # ============= Check the spaces (observation and action) ================
    # Define aliases for convenience
    observation_space = env.observation_space
    action_space = env.action_space
    try:
        env.step(env.action_space.sample())

    except AssertionError as e:
        assert str(e) == "Cannot call env.step() before calling reset()"

    # Warn the user if needed.
    # A warning means that the environment may run but not work properly with popular RL libraries.
    if warn:
        obs_spaces = (
            observation_space.spaces
            if isinstance(observation_space, spaces.Dict)
            else {"": observation_space}
        )
        for key, space in obs_spaces.items():
            if isinstance(space, spaces.Box):
                _check_box_obs(space, key)

        # Check for the action space, it may lead to hard-to-debug issues
        if isinstance(action_space, spaces.Box):
            _check_box_action(action_space)

    # ============ Check the returned values ===============
    _check_returned_values(env, observation_space, action_space)

    # ==== Check the render method and the declared render modes ====
    if not skip_render_check:
        _check_render(env, warn=warn)  # pragma: no cover

    # The check only works with numpy arrays
    if _is_numpy_array_space(observation_space) and _is_numpy_array_space(action_space):
        _check_nan(env)

    # ==== Check the reset method ====
    _check_reset_seed(env)
    _check_reset_seed(env, seed=0)
