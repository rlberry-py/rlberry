"""
================================================
Using multiple virtual environments with rlberry
================================================

This example illustrate how to use the "with_venv" decorator
in order to automatically construct and use virtual environments
for RL experimentation with several separated environments.

The decorator `with_venv` is used to generate scripts at compile time
and then are run via `run_venv_xp`.
Remark: the functions 'run_sb' and 'run_mushroom' are not directly called
and are only there to give the script's text.
"""


from rlberry.manager import with_venv, run_venv_xp


# Decorator with_venv will create a script to be run in the virtual environment with
# the libraries in the import_libs list. Here we want to create a virtual environment
# containing mushroom_rl library and run an example script taken from mushroom_rl doc.
@with_venv(import_libs=["numpy", "mushroom_rl"], venv_dir_name="rlberry_venvs")
def run_mushroom():
    """
    Simple script to solve a simple chain with Q-Learning.

    """
    import numpy as np

    from mushroom_rl.algorithms.value import QLearning
    from mushroom_rl.core import Core, Logger
    from mushroom_rl.environments import generate_simple_chain
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.utils.parameters import Parameter
    from mushroom_rl.utils.dataset import compute_J

    np.random.seed()

    logger = Logger(QLearning.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + QLearning.__name__)

    # MDP
    mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=0.8, rew=1, gamma=0.9)

    # Policy
    epsilon = Parameter(value=0.15)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(value=0.2)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = QLearning(mdp.info, pi, **algorithm_params)

    # Core
    core = Core(agent, mdp)

    # Initial policy Evaluation
    dataset = core.evaluate(n_steps=1000)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    logger.info(f"J start: {J}")

    # Train
    core.learn(n_steps=10000, n_steps_per_fit=1)

    # Final Policy Evaluation
    dataset = core.evaluate(n_steps=1000)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    logger.info(f"J final: {J}")


# Here we want to create a virtual environment containing stable-baselines3 library
# and run an example script taken from stable-baselines3 doc.
@with_venv(
    import_libs=["stable-baselines3"], venv_dir_name="rlberry_venvs", python_ver="3.9"
)
def run_sb():
    import gymnasium as gym

    from stable_baselines3 import A2C

    env = gym.make("CartPole-v1")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_500)

    vec_env = model.get_env()
    obs = vec_env.reset()
    cum_reward = 0
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        cum_reward += reward
    print(cum_reward)


if __name__ == "__main__":
    # Collect all the scripts from the directory rlberry_venvs and tun them.
    run_venv_xp(venv_dir_name="rlberry_venvs")
