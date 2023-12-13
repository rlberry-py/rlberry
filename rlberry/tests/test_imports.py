def test_imports():
    try:
        import rlberry  # noqa
        from rlberry.manager import (  # noqa
            ExperimentManager,  # noqa
            evaluate_agents,  # noqa
            plot_writer_data,  # noqa
        )  # noqa
        from rlberry.agents import AgentWithSimplePolicy  # noqa
        from rlberry.wrappers import WriterWrapper  # noqa

    except Exception:
        raise RuntimeError("Fail to call get_params on the environment.")
        assert False
