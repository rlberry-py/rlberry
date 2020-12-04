from rlberry.envs import gym_make
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from rlberry.agents.dqn import DQNAgent
from rlberry.utils.logging import configure_logging

configure_logging(level="INFO")

env = gym_make("CartPole-v0")
agent = DQNAgent(env, n_episodes=50, epsilon_decay=1000)
agent.set_writer(SummaryWriter())

print(f"Running DQN on {env}")
print(f"Visualize with tensorboard by \
running:\n$tensorboard --logdir {Path(agent.writer.log_dir).parent}")

agent.fit()

for episode in range(3):
    done = False
    state = env.reset()
    while not done:
        action = agent.policy(state)
        state, reward, done, _ = env.step(action)
        env.render()
env.close()
