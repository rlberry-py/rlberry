def get_base_env(env):
    """Traverse the wrappers to find the base environment."""
    while hasattr(env, "env"):
        env = env.env
    return env
