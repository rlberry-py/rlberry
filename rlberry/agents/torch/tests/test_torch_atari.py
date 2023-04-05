from rlberry.manager.agent_manager import AgentManager
from rlberry.agents.torch.dqn.dqn import DQNAgent
from rlberry.envs.gym_make import atari_make



def test_forward_dqn():
    mlp_configs = {
        "type": "MultiLayerPerceptron",  # A network architecture
        "layer_sizes": [512],  # Network dimensions
        "reshape": False,
        "is_policy": False,  # The network should output a distribution
        # over actions
    }

    cnn_configs = {
        "type": "ConvolutionalNetwork",  # A network architecture
        "activation": "RELU",
        "in_channels": 4,
        "in_height": 84,
        "in_width": 84,
        "head_mlp_kwargs": mlp_configs,
        "transpose_obs": False,
        "is_policy": False,  # The network should output a distribution
    }

    tuned_agent = AgentManager(
        DQNAgent,  # The Agent class.
        (
            atari_make,
            # uncomment when rlberry will manage vectorized env
            # dict(id="ALE/Breakout-v5", n_envs=3),
            dict(id="ALE/Breakout-v5", n_envs=1),
        ),  # The Environment to solve.
        init_kwargs=dict(  # Where to put the agent's hyperparameters
            q_net_constructor="rlberry.agents.torch.utils.training.model_factory_from_env",
            q_net_kwargs=cnn_configs,
            max_replay_size=100,
            batch_size=32,
            learning_starts=100,
            gradient_steps=1,
            epsilon_final=0.01,
            learning_rate=1e-4,  # Size of the policy gradient descent steps.
            chunk_size=5,
        ),
        fit_budget=200,  # The number of interactions between the agent and the environment during training.
        eval_kwargs=dict(
            eval_horizon=10
        ),  # The number of interactions between the agent and the environment during evaluations.
        n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
        agent_name="DQN_test",  # The agent's name.
    )

    tuned_agent.fit()



def test_forward_empty_input_dim():
    mlp_configs = {
        "type": "MultiLayerPerceptron",  # A network architecture
        "layer_sizes": [512],  # Network dimensions
        "reshape": False,
        "is_policy": False,  # The network should output a distribution
        # over actions
    }

    cnn_configs = {
        "type": "ConvolutionalNetwork",  # A network architecture
        "activation": "RELU",
        "head_mlp_kwargs": mlp_configs,
        "transpose_obs": False,
        "is_policy": False,  # The network should output a distribution
    }

    tuned_agent = AgentManager(
        DQNAgent,  # The Agent class.
        (
            atari_make,
            # uncomment when rlberry will manage vectorized env
            # dict(id="ALE/Breakout-v5", n_envs=3),
            dict(id="ALE/Breakout-v5", n_envs=1),
        ),  # The Environment to solve.
        init_kwargs=dict(  # Where to put the agent's hyperparameters
            q_net_constructor="rlberry.agents.torch.utils.training.model_factory_from_env",
            q_net_kwargs=cnn_configs,
            max_replay_size=100,
            batch_size=32,
            learning_starts=100,
            gradient_steps=1,
            epsilon_final=0.01,
            learning_rate=1e-4,  # Size of the policy gradient descent steps.
            chunk_size=5,
        ),
        fit_budget=10,  # The number of interactions between the agent and the environment during training.
        eval_kwargs=dict(
            eval_horizon=10
        ),  # The number of interactions between the agent and the environment during evaluations.
        n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
        agent_name="DQN_test",  # The agent's name.
    )

    tuned_agent.fit()
