.. _multiprocess:

Parallelization in rlberry
==========================

rlberry use python's standard multiprocessing library to execute the fit of agents in parallel on cpus. The parallelization is done via
:class:`~rlberry.manager.AgentManager` and via :class:`~rlberry.manager.MultipleManagers`.

If a user wants to use a third-party parallelization library like joblib, the user must be aware of where the seeding is done so as not to bias the results. rlberry automatically handles seeding when the native parallelization scheme are used.

Several multiprocessing scheme are implemented in rlberry.

Threading
---------

Thread multiprocessing "constructs higher-level threading interfaces on top of the lower level _thread module" (see the doc on `python's website <https://docs.python.org/fr/3/library/threading.html#module-threading>`_). This is the default scheme in rlberry, most of the time it will result in
having practically no parallelization except if the code executed in each thread (i.e. each fit) is executed without GIL (example: cython code or numpy code).

Process: spawn or forkserver
----------------------------

To have an efficient parallelization, it is often better to use processes (see the doc on `python's website <https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing>`_) using the parameter :code:`parallelization="process"` in :class:`~rlberry.manager.AgentManager` or :class:`~rlberry.manager.MultipleManagers`.

This implies that a new process will be launched for each fit of the AgentManager.

The advised method of parallelization is spawn (parameter :code:`mp_context="spawn"`), however spawn method has several drawbacks:

- The fit code needs to be encapsulated in a :code:`if __name__ == '__main__'` directive. Example :

.. code:: python

    from rlberry.agents.torch import A2CAgent
    from rlberry.manager import AgentManager
    from rlberry.envs.benchmarks.ball_exploration import PBall2D

    n_steps = 1e5
    batch_size = 256

    if __name__ == "__main__":
        manager = AgentManager(
            A2CAgent,
            (PBall2D, {}),
            init_kwargs=dict(batch_size=batch_size, gamma=0.99, learning_rate=0.001),
            n_fit=4,
            fit_budget=n_steps,
            parallelization="process",
            mp_context="spawn",
        )
        manager.fit()

- As a consequence, :code:`spawn` parallelization only works if called from the main script.
- :code:`spawn` does not work when called from a notebook. To work in a notebook, use :code:`fork` instead.
- :code:`forkserver` is an alternative to :code:`spawn` that performs sometimes faster than :code:`spawn`. :code:`forkserver` parallelization must also be encapsulated into a  :code:`if __name__ == '__main__'` directive and for now it is available only on Unix systems.


Process, fork
-------------

Forking multiprocessing is only possible on Unix systems (MacOS, Linux, ...).
It is available through the parameter :code:`mp_context="fork"` when :code:`parallelization="process"`.
Remark that there could be some logging error and hanging when using :code:`fork`. The usage of fork in rlberry is still experimental and may be unstable.
