from rlberry.envs import Chain
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.manager import AgentManager
import numpy as np
from rlberry.seeding import set_external_seed
import tempfile
import os
from rlberry.envs.gym_make import gym_make
import pathlib


SEED = 42


def _make_env(env):
    """
    Help function to construct env from parameter env.
    """
    if isinstance(env, str):
        if env == "continuous_state":
            env_ctor = PBall2D
            env_kwargs = {}
        elif env == "discrete_state":
            env_ctor = Chain
            env_kwargs = {}
        elif env == "vectorized_env_continuous":
            env_ctor = gym_make
            env_kwargs = dict(id="CartPole-v0")
        else:
            raise ValueError("The env given in parameter is not implemented")
    elif isinstance(env, tuple):
        env_ctor = env[0]
        env_kwargs = env[1]
    else:
        raise ValueError("The env given in parameter is not implemented")
    return env_ctor, env_kwargs


def _fit_agent_manager(agent, env="continuous_state", init_kwargs=None):
    """
    Check that the agent is compatible with :class:`~rlberry.manager.AgentManager`.

    Parameters
    ----------
    agent: rlberry agent module
        Agent class to test.
    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    init_kwargs : dict
        Arguments required by the agent's constructor.
    """
    if init_kwargs is None:
        init_kwargs = {}

    train_env = _make_env(env)
    try:
        agent = AgentManager(
            agent, train_env, fit_budget=5, n_fit=1, seed=SEED, init_kwargs=init_kwargs
        )
        agent.fit()
    except Exception as exc:
        raise RuntimeError("Agent not compatible with Agent Manager") from exc

    return agent


def check_agent_manager(agent, env="continuous_state", init_kwargs=None):
    """
    Check that the agent is compatible with :class:`~rlberry.manager.AgentManager`.

    Parameters
    ----------
    agent: rlberry agent module
        Agent class to test.
    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    init_kwargs : dict
        Arguments required by the agent's constructor.
    """
    _ = _fit_agent_manager(agent, env, init_kwargs=init_kwargs)


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
        state, info = agent1.env.reset()
    for f in range(n_checks):
        # do several tests if there is some randomness.
        if compare_using == "policy":
            results1.append(agent1.policy(state))
        else:
            method_to_call = getattr(agent1, compare_using)
            results1.append(method_to_call())
    set_external_seed(SEED)
    for f in range(n_checks):
        # do several tests if there is some randomness.
        if compare_using == "policy":
            result2 = agent2.policy(state)
        else:
            method_to_call = getattr(agent2, compare_using)
            result2 = method_to_call()
        if not np.all(results1[f] == result2):
            result = False
    return result


def check_fit_additive(agent, env="continuous_state", init_kwargs=None):
    """
    Check that fitting two times with 10 fit budget is the same as fitting
    one time with 20 fit budget.

    Parameters
    ----------
    agent: rlberry agent module
        Agent class to test.
    env: tuple (env_ctor, env_kwargs) or str in ["continuous_state", "discrete_state"], default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in ["continuous_state", "discrete_state"], we use a default Benchmark environment.
    init_kwargs : dict
        Arguments required by the agent's constructor.
    """
    if init_kwargs is None:
        init_kwargs = {}
    init_kwargs["seeder"] = SEED
    train_env = _make_env(env)

    set_external_seed(SEED)
    agent1 = agent(train_env, **init_kwargs)
    agent1.fit(10)
    agent1.fit(10)

    set_external_seed(SEED)
    agent2 = agent(train_env, **init_kwargs)
    agent2.fit(20)

    result = check_agents_almost_equal(agent1, agent2)

    assert (
        result
    ), "Error: fitting the agent two times for 10 steps is not equivalent to fitting it one time for 20 steps."


def check_save_load(agent, env="continuous_state", init_kwargs=None):
    """
    Check that the agent save a non-empty file and can load.

    Parameters
    ----------
    agent: rlberry agent module
        Agent class to test.
    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    init_kwargs : dict
        Arguments required by the agent's constructor.
    """
    if init_kwargs is None:
        init_kwargs = {}

    train_env_tuple = _make_env(env)
    with tempfile.TemporaryDirectory() as tmpdirname:
        manager = AgentManager(
            agent,
            train_env_tuple,
            fit_budget=5,
            n_fit=1,
            seed=SEED,
            init_kwargs=init_kwargs,
            output_dir=tmpdirname,
        )
        train_env = manager.train_env
        test_env = train_env_tuple[0](**train_env_tuple[1])

        manager.fit(3)

        # test individual agents save and load
        assert (
            os.path.getsize(str(manager.output_dir_) + "/agent_handlers/idx_0.pickle")
            > 1
        ), "The saved file is empty."
        try:
            manager.load(str(manager.output_dir_) + "/agent_handlers/idx_0.pickle")
        except Exception:
            raise RuntimeError("Failed to load the agent file.")

        # test agentManager save and load
        manager.save()
        assert os.path.exists(tmpdirname)

        path_to_load = next(pathlib.Path(tmpdirname).glob("**/*.pickle"))
        loaded_agent_manager = AgentManager.load(path_to_load)
        assert loaded_agent_manager

        # test with firest agent of the manager
        observation = test_env.reset()
        for tt in range(50):
            action = loaded_agent_manager.get_agent_instances()[0].policy(observation)
            next_observation, reward, done, info = test_env.step(action)
            if done:
                next_observation = test_env.reset()
            observation = next_observation


def check_seeding_agent(agent, env=None, continuous_state=False, init_kwargs=None):
    """
    Check that the agent is reproducible.

    Parameters
    ----------
    agent: rlberry agent module
        Agent class to test.
    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    init_kwargs : dict
        Arguments required by the agent's constructor.
    """
    agent1 = _fit_agent_manager(agent, env, init_kwargs=init_kwargs)
    agent2 = _fit_agent_manager(agent, env, init_kwargs=init_kwargs)

    result = check_agents_almost_equal(
        agent1.agent_handlers[0], agent2.agent_handlers[0]
    )

    assert result, "Agent not reproducible (same seed give different results)"


def check_multi_fit(agent, env="continuous_state", init_kwargs=None):
    """
    Check that fitting two times with budget greater than n_step (buffer size) is working.

    Parameters
    ----------
    agent: rlberry agent module
        Agent class to test.
    env: tuple (env_ctor, env_kwargs) or str in ["continuous_state", "discrete_state"], default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in ["continuous_state", "discrete_state"], we use a default Benchmark environment.
    init_kwargs : dict
        Arguments required by the agent's constructor.
    """
    if init_kwargs is None:
        init_kwargs = {}

    init_kwargs["seeder"] = SEED
    train_env_d = _make_env(env)
    train_env = train_env_d[0](**train_env_d[1])

    test_load_env_d = _make_env(env)
    test_load_env = test_load_env_d[0](**test_load_env_d[1])

    agent1 = agent(train_env, **init_kwargs)

    try:  # stableBaselines don't have get_params()
        if "n_steps" in agent1.get_params():
            agent1.n_steps = 30
    except:
        pass

    agent1.fit(100)
    agent1.fit(100)

    # test
    state = test_load_env.reset()
    for tt in range(50):
        action = agent1.policy(state)
        next_s, _, done, test = test_load_env.step(action)
        if done:
            break
        state = next_s


def check_vectorized_env_agent(
    agent, env="vectorized_env_continuous", agent_init_kwargs=None
):
    """
    Check that (multi-)fitting vectorized_env is working.

    Parameters
    ----------
    agent: rlberry agent module
        Agent class to test.
    env: tuple (env_ctor, env_kwargs) or str "vectorized_env_continuous (default="vectorized_env_continuous")
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state","vectorized"}, we use a default Benchmark environment.
    agent_init_kwargs : dict
        Arguments required by the agent's constructor.
    """

    if agent_init_kwargs is None:
        agent_init_kwargs = dict(
            learning_rate=1e-4, optimizer_type="ADAM", n_envs=3, n_steps=30
        )

    if "n_steps" not in agent_init_kwargs or agent_init_kwargs["n_envs"] is None:
        agent_init_kwargs["n_envs"] = 3
    if "n_steps" not in agent_init_kwargs or agent_init_kwargs["n_steps"] is None:
        agent_init_kwargs["n_steps"] = 30

    agent_init_kwargs["seeder"] = SEED

    env_d = _make_env(env)
    train_env = env_d[0](**env_d[1])
    test_env = env_d[0](**env_d[1])

    agent1 = agent(train_env, **agent_init_kwargs)
    agent1.fit(100)
    agent1.fit(100)

    # test the agent
    state = test_env.reset()
    for tt in range(50):
        action = agent1.policy(state)
        next_s, _, done, test = test_env.step(action)
        if done:
            break
        state = next_s


def check_rl_agent(agent, env="continuous_state", init_kwargs=None):
    """
    Check agent manager compatibility  and check reproducibility/seeding.
    Raises an exception if a check fails.

    Parameters
    ----------
    agent: rlberry agent module
        Agent class to test.
    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    init_kwargs : dict
        Arguments required by the agent's constructor.

    Examples
    --------
    >>> from rlberry.agents import UCBVIAgent
    >>> from rlberry.utils import check_rl_agent
    >>> check_rl_agent(UCBVIAgent) # which does not return an error.
    """
    check_agent_manager(
        agent, env, init_kwargs=init_kwargs
    )  # check manager compatible.
    check_seeding_agent(agent, env, init_kwargs=init_kwargs)  # check reproducibility
    check_fit_additive(agent, env, init_kwargs=init_kwargs)
    check_save_load(agent, env, init_kwargs=init_kwargs)
    check_multi_fit(agent, env, init_kwargs=init_kwargs)


def check_rlberry_agent(agent, env="continuous_state", init_kwargs=None):
    """
    Companion to check_rl_agent, contains additional tests. It is not mandatory
    for an agent to satisfy this check but satisfying this check give access to
    additional features in rlberry.

    Parameters
    ----------
    agent: rlberry agent module
        Agent class to test.
    env: tuple (env_ctor, env_kwargs) or str in {"continuous_state", "discrete_state"}, default="continuous_state"
        if tuple, env is the constructor and keywords of the env on which to test.
        if str in {"continuous_state", "discrete_state"}, we use a default Benchmark environment.
    init_kwargs : dict
        Arguments required by the agent's constructor.

    Examples
    --------
    >>> from rlberry.agents import UCBVIAgent
    >>> from rlberry.utils import check_rl_agent
    >>> check_rl_agent(UCBVIAgent) #
    """
    agent = _fit_agent_manager(agent, env, init_kwargs=init_kwargs).agent_handlers[0]
    try:
        params = agent.get_params()
    except Exception:
        raise RuntimeError("Fail to call get_params on the agent.")
