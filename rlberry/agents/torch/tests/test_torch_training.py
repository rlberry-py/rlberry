import torch

import os
from rlberry.agents.torch.utils.training import (
    loss_function_factory,
    optimizer_factory,
    model_factory,
    model_factory_from_env,
)
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.agents.torch.utils.models import (
    default_policy_net_fn,
    Net,
    MultiLayerPerceptron,
)
from rlberry.agents.torch.dqn import DQNAgent


# loss_function_factory
assert isinstance(loss_function_factory("l2"), torch.nn.MSELoss)
assert isinstance(loss_function_factory("l1"), torch.nn.L1Loss)
assert isinstance(loss_function_factory("smooth_l1"), torch.nn.SmoothL1Loss)
assert isinstance(loss_function_factory("bce"), torch.nn.BCELoss)

# optimizer_factory
env = get_benchmark_env(level=1)
assert (
    optimizer_factory(default_policy_net_fn(env).parameters(), "ADAM").defaults["lr"]
    == 0.001
)
assert optimizer_factory(default_policy_net_fn(env).parameters(), "ADAM").defaults[
    "betas"
] == (0.9, 0.999)
assert (
    optimizer_factory(default_policy_net_fn(env).parameters(), "RMS_PROP").defaults[
        "lr"
    ]
    == 0.01
)
assert (
    optimizer_factory(default_policy_net_fn(env).parameters(), "RMS_PROP").defaults[
        "alpha"
    ]
    == 0.99
)


# test model_factory

obs_shape = env.observation_space.shape
n_act = env.action_space.n

test_net = Net(obs_size=obs_shape[0], hidden_size=10, n_actions=n_act)

test_net2 = MultiLayerPerceptron(in_size=obs_shape[0], layer_sizes=[10], out_size=1)


test_net3 = MultiLayerPerceptron(
    in_size=obs_shape[0], layer_sizes=[10], out_size=n_act, is_policy=True
)


model_factory(net=test_net)
model_factory_from_env(env, net=test_net)
model_factory_from_env(env, net=test_net2, out_size=1)
model_factory_from_env(env, net=test_net3, is_policy=True)


# test loading pretrained nn
dqn_agent = DQNAgent(
    env, q_net_constructor=model_factory_from_env, q_net_kwargs=dict(net=test_net)
)

dqn_agent.fit(50)

torch.save(dqn_agent._qnet_online, "test_dqn.pickle")


parameters_to_save = dqn_agent._qnet_online.state_dict()
torch.save(parameters_to_save, "test_dqn.pt")


model_factory(filename="test_dqn.pickle")
model_factory(net=test_net, filename="test_dqn.pt")


dqn_agent = DQNAgent(
    env,
    q_net_constructor=model_factory_from_env,
    q_net_kwargs=dict(filename="test_dqn.pickle"),
)

dqn_agent = DQNAgent(
    env,
    q_net_constructor=model_factory_from_env,
    q_net_kwargs=dict(net=test_net, filename="test_dqn.pt"),
)

assert dqn_agent._qnet_online.state_dict().keys() == parameters_to_save.keys()

for k in parameters_to_save.keys():
    assert (dqn_agent._qnet_online.state_dict()[k] == parameters_to_save[k]).all()

os.remove("test_dqn.pickle")
os.remove("test_dqn.pt")

print("done")
