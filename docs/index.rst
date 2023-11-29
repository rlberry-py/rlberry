.. image:: ../assets/logo_wide.svg
    :width: 50%
    :alt: rlberry logo
    :align: center

.. _rlberry: https://github.com/rlberry-py/rlberry
../

.. _index:

An RL Library for Research and Education
========================================


:ref:`new user guide<user_guide2>`

**Writing reinforcement learning algorithms is fun!** *But after the fun, we have
lots of boring things to implement*: run our agents in parallel, average and plot results,
optimize hyperparameters, compare to baselines, create tricky environments etc etc!

rlberry_ **is here to make your life easier** by doing all these things with a few lines of code,
so that you can spend most of your time developing agents. **Check our** :ref:`the quickstart<quick_start>`




In addition, rlberry_:

* Provides **implementations of several RL agents** for you to use as a starting point or as baselines;
* Provides a set of **benchmark environments**, very useful to debug and challenge your algorithms;
* Handles all random seeds for you, ensuring **reproducibility** of your results;
* Is **fully compatible with** several commonly used RL libraries like `Gymnasium <https://gymnasium.farama.org/>`_ and `Stable Baselines <https://stable-baselines.readthedocs.io/en/master/>`_ (see :ref:`userguide/agents<agent_page>`).



Seeding & Reproducibility
==========================

rlberry_ has a class :class:`~rlberry.seeding.seeder.Seeder` that conveniently wraps a `NumPy SeedSequence <https://numpy.org/doc/stable/reference/random/parallel.html>`_,
and allows us to create independent random number generators for different objects and threads, using a single
:class:`~rlberry.seeding.seeder.Seeder` instance. See :ref:`Seeding <seeding>`.



Contributing to rlberry
=======================

If you want to contribute to rlberry, check out :ref:`the contribution guidelines<contributing>`.



Documentation Contents
======================

.. toctree::
  :maxdepth: 3

  installation
  user_guide
  external

.. toctree::
  :maxdepth: 2

  api
  changelog
