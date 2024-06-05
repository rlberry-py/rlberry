from rlberry.manager import with_venv, run_venv_xp


@with_venv(import_libs=["numpy", "mushroom_rl"])
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


@with_venv(import_libs=["stable-baselines3"], python_ver="3.9")
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
    run_venv_xp()
