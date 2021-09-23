from rlberry.agents.torch.utils.training import loss_function_factory, optimizer_factory
from torch.nn import functional as F
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.agents.torch.utils.models import default_policy_net_fn

# loss_function_factory
assert isinstance(loss_function_factory("l2").__name__, type(F.mse_loss.__name__))
assert isinstance(loss_function_factory("l1").__name__, type(F.l1_loss.__name__))
assert isinstance(loss_function_factory("smooth_l1").__name__, type(F.smooth_l1_loss.__name__))
assert isinstance(loss_function_factory("bce").__name__, type(F.binary_cross_entropy.__name__))

# optimizer_factory
env = get_benchmark_env(level=1)
assert optimizer_factory(default_policy_net_fn(env).parameters(), "ADAM").defaults["lr"] == 0.001
assert optimizer_factory(default_policy_net_fn(env).parameters(), "ADAM").defaults["betas"] == (0.9, 0.999)
assert optimizer_factory(default_policy_net_fn(env).parameters(), "RMS_PROP").defaults["lr"] == 0.01
assert optimizer_factory(default_policy_net_fn(env).parameters(), "RMS_PROP").defaults["alpha"] == 0.99
