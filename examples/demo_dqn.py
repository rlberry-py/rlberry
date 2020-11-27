import gym
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from rlberry.agents.dqn.pytorch import DQNAgent

env = gym.make("CartPole-v0")
config = {
    "n_episodes": 100,
    "exploration": {"tau": 1000},
}
agent = DQNAgent(env, config=config)
agent.set_writer(SummaryWriter())
print(f"Running DQN on {env}")
print(f"Visualize with tensorboard by running:\n$ tensorboard --logdir {Path(agent.writer.log_dir).parent}")
agent.fit()

for episode in range(3):
    done = False
    state = env.reset()
    while not done:
        action = agent.policy(state)
        state, reward, done, _ = env.step(action)
        env.render()
env.close()
