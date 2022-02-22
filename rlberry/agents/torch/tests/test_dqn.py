import pytest
from rlberry.envs import gym_make
from rlberry.agents.torch.dqn import DQNAgent
from rlberry.seeding import Seeder


@pytest.mark.parametrize(
    "use_double_dqn, use_prioritized_replay", [(False, False), (True, True)]
)
def test_dqn_agent(use_double_dqn, use_prioritized_replay):
    env = gym_make("CartPole-v0")
    agent = DQNAgent(
        env,
        learning_starts=5,
        eval_interval=75,
        train_interval=2,
        gradient_steps=-1,
        use_double_dqn=use_double_dqn,
        use_prioritized_replay=use_prioritized_replay,
    )
    agent.fit(budget=500)
