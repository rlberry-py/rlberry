from rlberry.envs import Chain
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.manager import AgentManager
import numpy as np
from rlberry.seeding import set_external_seed
import tempfile
import os

SEED = 42


def _make_env(env):
    """
    help function to construct env from parameter env.
    """
    if isinstance(env, str):
        if env == "continuous_state":
            env_ctor = PBall2D
            env_kwargs = {}
        elif env == "discrete_state":
            env_ctor = Chain
            env_kwargs = {}
        else:
            raise ValueError("The env given in parameter is not implemented")
    elif isinstance(env, tuple):
        env_ctor = env[0]
        env_kwargs = env[1]
    else:
        raise ValueError("The env given in parameter is not implemented")
    return env_ctor, env_kwargs


def _fit_agent_manager(Agent, env="continuous_state"):
    """
    Check that the agent is compatible with :class:`~rlberry.manager.AgentManager`.

    Parameters
    ----------

    Agent: rlberry agent module
        Agent class to test.

    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    """
    train_env = _make_env(env)
    try:
        agent = AgentManager(Agent, train_env, fit_budget=5, n_fit=1, seed=SEED)
        agent.fit()
    except Exception as exc:
        raise RuntimeError("Agent not compatible with Agent Manager") from exc

    return agent


def check_agent_manager(Agent, env="continuous_state"):
    """
    Check that the agent is compatible with :class:`~rlberry.manager.AgentManager`.

    Parameters
    ----------

    Agent: rlberry agent module
        Agent class to test.

    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    """
    _ = _fit_agent_manager(Agent, env)


def check_agents_almost_equal(agent1, agent2, compare_using="policy", n_checks=5):
    """
    Check that two agents with a fixed global seed are almost equal in the
    sense that their agents give the same actions n_checks times on a given state.

    WARNING: we set a global seed in this check. Ideally an agent should be
    reproducible when rlberry seed is fixed but this is hard to do, in particular
    when using torch.
    Parameters
    ----------

    agent1: rlberry Agent instance
        first agent to be compared
    agent2: rlberry Agent instance
        second agent to be compared
    compare_using: str, default = "policy"
        method that we use to compare the agents. Can be "policy" or "eval" or
        the name of any method implemented by an agent that we can call
        and that return a real number.
    n_checks : int, default=5
        number of checks to do.
    """
    result = True
    set_external_seed(SEED)
    results1 = []
    if compare_using == "policy":
        state = agent1.env.reset()
    for f in range(5):
        # do several tests if there is some randomness.
        if compare_using == "policy":
            results1.append(agent1.policy(state))
        else:
            method_to_call = getattr(agent1, compare_using)
            results1.append(method_to_call())
    set_external_seed(SEED)
    for f in range(5):
        # do several tests if there is some randomness.
        if compare_using == "policy":
            result2 = agent2.policy(state)
        else:
            method_to_call = getattr(agent2, compare_using)
            result2 = method_to_call()
        if not np.all(results1[f] == result2):
            result = False
    return result


def check_fit_additive(Agent, env="continuous_state"):
    """
    check that fitting two times with 10 fit budget is the same as fitting
    one time with 20 fit budget.

    Parameters
    ----------

    Agent: rlberry agent module
        Agent class to test.

    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    """
    train_env = _make_env(env)
    agent1 = AgentManager(Agent, train_env, fit_budget=5, n_fit=1, seed=SEED)
    agent1.fit(3)
    agent1.fit(3)

    agent2 = AgentManager(Agent, train_env, fit_budget=5, n_fit=1, seed=SEED)
    agent2.fit(6)

    result = check_agents_almost_equal(
        agent1.agent_handlers[0], agent2.agent_handlers[0]
    )

    assert (
        result
    ), "Error: fitting the agent two times for 10 steps is not equivalent to fitting it one time for 20 steps."


def check_save_load(Agent, env="continuous_state"):
    """
    Check that the agent save a non-empty file and can load.


    Parameters
    ----------

    Agent: rlberry agent module
        Agent class to test.

    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    """
    train_env = _make_env(env)
    env = train_env[0](**train_env[1])
    with tempfile.TemporaryDirectory() as tmpdirname:
        agent = AgentManager(
            Agent,
            train_env,
            fit_budget=5,
            n_fit=1,
            seed=SEED,
            output_dir=tmpdirname,
        )
        agent.fit(3)
        assert (
            os.path.getsize(str(agent.output_dir_) + "/agent_handlers/idx_0.pickle") > 1
        ), "the saved file is empty"
        try:
            agent.load(str(agent.output_dir_) + "/agent_handlers/idx_0.pickle")
        except:
            raise RuntimeError("failed to load the agent file")


def check_seeding_agent(Agent, env=None, continuous_state=False):
    """
    Check that the agent is reproducible.

    Parameters
    ----------

    Agent: rlberry agent module
        Agent class to test.

    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    """
    agent1 = _fit_agent_manager(Agent, env)
    agent2 = _fit_agent_manager(Agent, env)

    result = check_agents_almost_equal(
        agent1.agent_handlers[0], agent2.agent_handlers[0]
    )

    assert result, "Agent not reproducible (same seed give different results)"


def check_rl_agent(Agent, env="continuous_state"):
    """
    Check agent manager compatibility  and check reproducibility/seeding.
    Raises an exception if a check fails.

    Parameters
    ----------

    Agent: rlberry agent module
        Agent class to test.

    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.

    Examples
    --------
    >>> from rlberry.agents import UCBVIAgent
    >>> from rlberry.utils import check_rl_agent
    >>> check_rl_agent(UCBVIAgent) # which does not return an error.
    """
    check_agent_manager(Agent, env)  # check manager compatible.
    check_seeding_agent(Agent, env)  # check reproducibility
    check_fit_additive(Agent, env)
    check_save_load(Agent, env)
