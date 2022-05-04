import pytest
from rlberry.envs import gym_make
from rlberry.agents.torch.dqn import DQNAgent
from rlberry.agents.torch.utils.training import model_factory


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

    model_configs = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": (5, 5),
        "reshape": False,
    }

    def mlp(env, **kwargs):
        """
        Returns a default Q value network.
        """
        kwargs["in_size"] = env.observation_space.shape[0]
        return model_factory(**kwargs)

    new_agent = DQNAgent(env, q_net_constructor=mlp, q_net_kwargs=model_configs)
    new_agent.fit(budget=2000)