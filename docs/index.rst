.. image:: ../assets/logo_wide.svg
    :width: 50%
    :alt: rlberry logo

.. _rlberry: https://github.com/rlberry-py/rlberry


An RL Library for Research and Education
========================================


**Writing reinforcement learning algorithms is fun!** *But after the fun, we have
lots of boring things to implement*: run our agents in parallel, average and plot results,
optimize hyperparameters, compare to baselines, create tricky environments etc etc!

rlberry_ **is here to make your life easier** by doing all these things with a few lines of code,
so that you can spend most of your time developing agents. **Check our** :ref:`quick-tutorial` **section!**




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
:class:`~rlberry.seeding.seeder.Seeder` instance. See :ref:`Seeding <seeding>`.


Contributing to rlberry
=======================

If you want to contribute to rlberry, check out :ref:`the contribution guidelines<contributing>`.



Documentation Contents
======================

.. toctree::
  :maxdepth: 2
  :caption: Quick start

  installation
  quickstart
  external
  source/modules

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide
