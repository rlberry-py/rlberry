from rlberry.agents.torch.utils.training import model_factory


def default_q_net_fn(env, **kwargs):
    """
    Returns a default Q value network.

    Parameters
    ----------
    env : gymnasium.Env
        An environment.
    **kwargs : Keyword Arguments
        Extra arguments used to call :func:`~rlberry.agents.torch.utils.training.model_factory`.
    """
    model_config = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": (256, 256),
        "reshape": True,
        "in_size": env.observation_space.shape[0] + env.action_space.shape[0],
        "out_size": 1,
    }
    if kwargs:
        for k, v in kwargs.items():
            model_config[k] = v
    return model_factory(**model_config)


def default_policy_net_fn(env, **kwargs):
    """
    Returns a default Q value network.

    Parameters
    ----------
    env : gymnasium.Env
        An environment.
    **kwargs : Keyword Arguments
        Extra arguments used to call :func:`~rlberry.agents.torch.utils.training.model_factory`.
    """
    model_config = {
        "type": "MultiLayerPerceptron",
        "in_size": env.observation_space.shape[0],
        "layer_sizes": [256, 256],
        "out_size": env.action_space.shape[0],
        "reshape": True,
        "is_policy": True,
        "ctns_actions": True,
        "squashed_policy": True,
    }
    if kwargs:
        for k, v in kwargs.items():
            model_config[k] = v
    return model_factory(**model_config)
