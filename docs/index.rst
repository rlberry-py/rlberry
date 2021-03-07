.. rlberry documentation master file, created by
   sphinx-quickstart on Sat Jan 16 00:20:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ../assets/logo_wide.svg
    :width: 50%
    :alt: rlberry logo

.. _rlberry: https://github.com/rlberry-py/rlberry


An RL Library for Research and Education
========================================

.. image:: https://api.codacy.com/project/badge/Grade/27e91674d18a4ac49edf91c339af1502
    :alt: codacy

.. image:: https://codecov.io/gh/rlberry-py/rlberry/branch/main/graph/badge.svg?token=TIFP7RUD75
    :alt: codecov

.. image:: https://img.shields.io/pypi/pyversions/rlberry
    :alt: PyPI - Python Version

.. image:: https://img.shields.io/pypi/status/rlberry
    :alt: PyPI - Status

.. image:: https://github.com/rlberry-py/rlberry/workflows/test/badge.svg
    :alt: tests

**Writing reinforcement learning algorithms is fun!** *But after the fun, we have
lots of boring things to implement*: run our agents in parallel, average and plot results, 
optimize hyperparameters, compare to baselines, create tricky environments etc etc!

rlberry_ **is here to make your life easier** by doing all these things with a few lines of code,
so that you can spend most of your time developing agents!


In **a few minutes of reading**, you will learn how to:

- :ref:`Install rlberry <installation>`;
- :ref:`Create an agent <create_agent>`;
- :ref:`Evaluate an agent and optimize its hyperparameters <evaluate_agent>`;
- :ref:`Compare different agents <compare_agents>`;
- :ref:`Setup and run experiments using yaml config files <experiment_setup>`.

In addition, rlberry_: 

* Provides **implementations of several RL agents** for you to use as a starting point or as baselines;
* Provides a set of **benchmark environments**, very useful to debug and challenge your algorithms;
* Handles all random seeds for you, ensuring **reproducibility** of your results;
* Is **fully compatible with** several commonly used RL libraries like `OpenAI gym <https://gym.openai.com/>`_ and `Stable Baselines <https://stable-baselines.readthedocs.io/en/master/>`_.


Compatibility with External Libraries
=====================================

We provide examples to show you how to use rlberry_ with:

- :ref:`OpenAI Gym <gym>`;
- :ref:`Stable Baselines <stable_baselines>`.


Seeding & Reproducibility
==========================

rlberry_ has a class :class:`~rlberry.seeding.seeder.Seeder` that conveniently wraps a `NumPy SeedSequence <https://numpy.org/doc/stable/reference/random/parallel.html>`_,
and allows us to create independent random number generators for different objects and threads, using a single
:class:`~rlberry.seeding.seeder.Seeder` instance.

It works as follows:


.. code-block:: python

    from rlberry.seeding import Seeder

    seeder = Seeder(123)

    # Each Seeder instance has a random number generator (rng)
    # See https://numpy.org/doc/stable/reference/random/generator.html to check the
    # methods available in rng.
    seeder.rng.integers(5)
    seeder.rng.normal()
    print(type(seeder.rng))
    # etc

    # Environments and agents should be seeded using a single seeder,
    # to ensure that their random number generators are independent.
    from rlberry.envs import gym_make
    from rlberry.agents import RSUCBVIAgent
    env = gym_make('MountainCar-v0')
    env.reseed(seeder)

    agent = RSUCBVIAgent(env)
    agent.reseed(seeder)


    # Environments and Agents have their own seeder and rng.
    # When writing your own agents and inheriring from the Agent class,
    # you should use agent.rng whenever you need to generate random numbers;
    # the same applies to your environments.
    # This is necessary to ensure reproducibility.
    print("env seeder: ", env.seeder)
    print("random sample from env rng: ", env.rng.normal())
    print("agent seeder: ", agent.seeder)
    print("random sample from agent rng: ", agent.rng.normal())


    # A seeder can spawn other seeders that are independent from it.
    # This is useful to seed two different threads, using seeder1
    # in the first thread, and seeder2 in the second thread.
    seeder1, seeder2 = seeder.spawn(2)


    # You can also use a seeder to seed external libraries (such as torch)
    # using the function set_external_seed
    from rlberry.seeding import set_external_seed
    set_external_seed(seeder)


.. note:: 
    The class :class:`~rlberry.stats.agent_stats.AgentStats` provides a :code:`seed` parameter in its constructor,
    and handles automatically the seeding of all environments and agents used by it.

.. note:: 

   The :meth:`optimize_hyperparams` method of 
   :class:`~rlberry.stats.agent_stats.AgentStats` uses the `Optuna <https://optuna.org/>`_ 
   library for hyperparameter optimization and is **inherently non-deterministic**
   (see `Optuna FAQ <https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results>`_).



Citing rlberry_
===============

If you use rlberry in scientific publications, we would appreciate citations using the following Bibtex entry:

.. code-block:: bibtex

   @misc{rlberry,
   author = {Domingues, Omar Darwiche and ‪Flet-Berliac, Yannis and Leurent, Edouard and M{\'e}nard, Pierre and Shang, Xuedong and Valko, Michal},
   title = {{rlberry - A Reinforcement Learning Library for Research and Education}},
   year = {2021},
   publisher = {GitHub},
   journal = {GitHub repository},
   howpublished = {\url{https://github.com/rlberry-py/rlberry}},
   }


Documentation Contents
======================

.. toctree::
  :maxdepth: 2

  installation
  quickstart
  external
  source/modules


Indices and Tables
==================

* :ref:`modindex`
* :ref:`search`
