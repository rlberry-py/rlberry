import pytest
from rlberry.envs import gym_make
from rlberry.agents.torch.dqn import MunchausenDQNAgent
from rlberry.agents.torch.utils.training import model_factory


@pytest.mark.parametrize("use_prioritized_replay", [(False), (True)])
def test_mdqn_agent(use_prioritized_replay):
    env = gym_make("CartPole-v1")
    agent = MunchausenDQNAgent(
        env,
        learning_starts=5,
        batch_size=5,
        eval_interval=2,
        train_interval=2,
        gradient_steps=-1,
        use_prioritized_replay=use_prioritized_replay,
    )
    agent.fit(budget=50)

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
        kwargs["out_size"] = env.action_space.n
        return model_factory(**kwargs)

    new_agent = MunchausenDQNAgent(
        env, q_net_constructor=mlp, q_net_kwargs=model_configs, learning_starts=100
    )
    new_agent.fit(budget=200)
    observation, info = env.reset()
    new_agent.policy(observation)
