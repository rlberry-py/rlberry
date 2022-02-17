from rlberry.envs import Chain
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.manager import AgentManager
import numpy as np
from rlberry.seeding import set_external_seed
import tempfile
import os

SEED = 42


def check_agent_manager(Agent, env=None, continuous_state=False):
    """
    Check that the agent is compatible with :class:`~rlberry.manager.AgentManager`.

    Parameters
    ----------

    Agent: rlberry agent module
        Agent class to test.

    env: tuple (env_ctor, env_kwargs) or None, default=None
        constructor and arguments of the env on which to test. If None,
        the parameter continuous-state is used to provide a known working
        environment for the test.

    continuous_state: bool, default=False
        whether to test on a discrete state environment or a continuous one.
        Used only if env is None.
    """
    if env is None and continuous_state:
        env_ctor = PBall2D
        env_kwargs = {}
    elif env is None:
        env_ctor = Chain
        env_kwargs = {}
    else:
        env_ctor = env[0]
        env_kwargs = env[1]
    try:
        agent = AgentManager(
            Agent, (env_ctor, env_kwargs), fit_budget=5, n_fit=1, seed=SEED
        )
        agent.fit()
    except Exception as exc:
        raise RuntimeError("Agent not compatible with Agent Manager") from exc

    return agent


def check_agent_manager_almost_equal(state, agent1, agent2, n_checks=5):
    """
    Check that two agent managers with a fixed seed are almost equal in the
    sense that they give the same actions n_checks times on a given state.


    Parameters
    ----------

    state: state
        environment state
    agent1: rlberry AgentManager instance
        first agent manager to be compared
    agent2: rlberry AgentManager instance
        second agent manager to be compared
    n_checks : int, default=5
        number of checks to do.
    """
    result = True
    set_external_seed(SEED)
    actions1 = []
    for f in range(5):
        # test on 5 actions
        actions1.append(agent1.agent_handlers[0].policy(state))
    set_external_seed(SEED)
    for f in range(5):
        # test on 5 actions
        action2 = agent2.agent_handlers[0].policy(state)
        if not np.all(actions1[f] == action2):
            result = False
    return result


def check_fit_additive(Agent, env=None, continuous_state=False):
    """
    check that fitting two times with 10 fit budget is the same as fitting
    one time with 20 fit budget.

    Parameters
    ----------

    Agent: rlberry agent module
        Agent class to test.

    env: tuple (env_ctor, env_kwargs) or None, default=None
        constructor and arguments of the env on which to test. If None,
        the parameter continuous-state is used to provide a known working
        environment for the test.

    continuous_state: bool, default=False
        whether to test on a discrete state environment or a continuous one.
        Used only if env is None.
    """
    if env is None and continuous_state:
        env_ctor = PBall2D
        env_kwargs = {}
    elif env is None:
        env_ctor = Chain
        env_kwargs = {}
    else:
        env_ctor = env[0]
        env_kwargs = env[1]
    agent1 = AgentManager(
        Agent, (env_ctor, env_kwargs), fit_budget=5, n_fit=1, seed=SEED
    )
    agent1.fit(10)
    agent1.fit(10)

    agent2 = AgentManager(
        Agent, (env_ctor, env_kwargs), fit_budget=5, n_fit=1, seed=SEED
    )
    agent2.fit(20)

    env = env_ctor()
    state = env.reset()
    result = check_agent_manager_almost_equal(state, agent1, agent2)

    assert (
        result
    ), "Error: fitting the agent two times for 10 steps is not equivalent to fitting it one time for 20 steps."


def check_save_load(Agent, env=None, continuous_state=False):
    """
    Check that the agent save a non-empty file and can load.


    Parameters
    ----------

    Agent: rlberry agent module
        Agent class to test.

    env: tuple (env_ctor, env_kwargs) or None, default=None
        constructor and arguments of the env on which to test. If None,
        the parameter continuous-state is used to provide a known working
        environment for the test.

    continuous_state: bool, default=False
        whether to test on a discrete state environment or a continuous one.
        Used only if env is None.
    """
    if env is None and continuous_state:
        env_ctor = PBall2D
        env_kwargs = {}
    elif env is None:
        env_ctor = Chain
        env_kwargs = {}
    else:
        env_ctor = env[0]
        env_kwargs = env[1]
    env = env_ctor(**env_kwargs)
    with tempfile.TemporaryDirectory() as tmpdirname:
        if continuous_state:
            env_ctor = PBall2D
        else:
            env_ctor = Chain
        env_kwargs = {}
        agent = AgentManager(
            Agent,
            (env_ctor, env_kwargs),
            fit_budget=5,
            n_fit=1,
            seed=SEED,
            output_dir=tmpdirname,
        )
        agent.fit(10)
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

    env: tuple (env_ctor, env_kwargs) or None, default=None
        constructor and arguments of the env on which to test. If None,
        the parameter continuous-state is used to provide a known working
        environment for the test.

    continuous_state: bool, default=False
        whether to test on a discrete state environment or a continuous one.
        Used only if env is None.
    """
    agent1 = check_agent_manager(Agent, env, continuous_state)
    agent2 = check_agent_manager(Agent, env, continuous_state)

    if env is None and continuous_state:
        env_ctor = PBall2D
        env_kwargs = {}
    elif env is None:
        env_ctor = Chain
        env_kwargs = {}
    else:
        env_ctor = env[0]
        env_kwargs = env[1]
    env = env_ctor(**env_kwargs)
    state = env.reset()
    result = check_agent_manager_almost_equal(state, agent1, agent2)

    assert result, "Agent not reproducible (same seed give different results)"


def check_rl_agent(Agent, env=None, continuous_state=False):
    """
    Check agent manager compatibility  and check reproducibility/seeding.
    Will raise an exception if there is something wrong with the agent.

    Parameters
    ----------

    Agent: rlberry agent module
        Agent class to test.

    env: tuple (env_ctor, env_kwargs) or None, default=None
        constructor and arguments of the env on which to test. If None,
        the parameter continuous-state is used to provide a known working
        environment for the test.

    continuous_state: bool, default=False
        whether to test on a discrete state environment or a continuous one.
        Used only if env is None.

    Examples
    --------
    >>> from rlberry.agents import UCBVIAgent
    >>> from rlberry.utils import check_rl_agent
    >>> check_rl_agent(UCBVIAgent) # which does not return an error.
    """
    check_agent_manager(Agent, env, continuous_state)  # check manager compatible.
    check_seeding_agent(Agent, env, continuous_state)  # check reproducibility
    check_fit_additive(Agent, env, continuous_state)
    check_save_load(Agent, env, continuous_state)
