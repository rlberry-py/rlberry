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
        kwargs["out_size"] = env.action_space.n
        return model_factory(**kwargs)

    new_agent = DQNAgent(
        env, q_net_constructor=mlp, q_net_kwargs=model_configs, learning_starts=100
    )
    new_agent.fit(budget=2000)


import os
from rlberry.manager.agent_manager import AgentManager
import shutil
import pathlib


def test_dqn_classic_env():
    env = gym_make("CartPole-v0")
    agent = DQNAgent(
        env,
        learning_starts=5,
        eval_interval=75,
        train_interval=2,
        gradient_steps=-1,
        use_double_dqn=True,
        use_prioritized_replay=True,
    )
    agent.fit(budget=200)

    saving_path = "rlberry/agents/torch/tests/agent_test_dqn_classic_env.pickle"

    # VRemove previous save
    if os.path.exists(saving_path):
        os.remove(saving_path)
    assert not os.path.exists(saving_path)

    # test the save function
    agent.save(saving_path)
    assert os.path.exists(saving_path)

    # test the loading function
    test_load_env = gym_make("CartPole-v0")
    loaded_agent = DQNAgent.load(saving_path, **dict(env=test_load_env))
    assert loaded_agent

    # test the agent
    state = test_load_env.reset()
    for tt in range(50):
        action = loaded_agent.policy(state)
        next_state, reward, done, _ = test_load_env.step(action)
        if done:
            next_state = test_load_env.reset()
        state = next_state

    os.remove(saving_path)


def test_dqn_agent_manager_classic_env():
    saving_path = "rlberry/agents/torch/tests/agentmanager_test_dqn_classic_env"

    # Remove previous save
    if os.path.exists(saving_path):
        shutil.rmtree(saving_path)
    assert not os.path.exists(saving_path)

    test_agent_manager = AgentManager(
        DQNAgent,  # The Agent class.
        (
            gym_make,
            dict(
                id="CartPole-v0",
            ),
        ),  # The Environment to solve.
        init_kwargs=dict(  # Where to put the agent's hyperparameters
            learning_starts=5,
            eval_interval=75,
            train_interval=2,
            gradient_steps=-1,
            use_double_dqn=True,
            use_prioritized_replay=True,
            chunk_size=1,
        ),
        fit_budget=200,  # The number of interactions between the agent and the environment during training.
        eval_kwargs=dict(
            eval_horizon=50
        ),  # The number of interactions between the agent and the environment during evaluations.
        n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
        agent_name="test_dqn_classic_env",  # The agent's name.
        output_dir=saving_path,
    )

    test_agent_manager.fit(budget=200)

    # test the save function
    test_agent_manager.save()
    assert os.path.exists(saving_path)

    # test the loading function
    test_load_env = gym_make("CartPole-v0")
    path_to_load = next(pathlib.Path(saving_path).glob("**/*.pickle"))
    loaded_agent_manager = AgentManager.load(path_to_load)
    assert loaded_agent_manager

    # test the agent
    state = test_load_env.reset()
    for tt in range(50):
        action = loaded_agent_manager.get_agent_instances()[0].policy(state)
        next_s, _, done, test = test_load_env.step(action)
        if done:
            break
        state = next_s

    shutil.rmtree(saving_path)
