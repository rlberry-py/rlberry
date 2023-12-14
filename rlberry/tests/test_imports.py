def test_imports():
    import rlberry  # noqa
    from rlberry.manager import (  # noqa
        ExperimentManager,  # noqa
        evaluate_agents,  # noqa
        plot_writer_data,  # noqa
    )  # noqa
    from rlberry.agents import AgentWithSimplePolicy  # noqa
    from rlberry.wrappers import WriterWrapper  # noqa
