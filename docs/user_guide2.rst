.. title:: User guide : contents

.. _user_guide2:

==========
User guide
==========

.. Introduction
.. ============
.. Welcome to rlberry. Use rlberry's ExperimentManager (add ref) to train, evaluate and compare rl agents. In addition to
.. the core ExperimentManager (add ref), rlberry provides the user with a set of bandit (add ref), tabular rl (add ref), and
.. deep rl agents (add ref) as well as a wrapper for stablebaselines3 (add link, and ref) agents.
.. Like other popular rl libraries, rlberry also provides basic tools for plotting, multiprocessing and logging (add refs).
.. In this user guide, we take you through the core features of rlberry and illustrate them with examples (add ref) and API documentation (add ref).

If you are new to rlberry, check the :ref:`Tutorials` below and the :ref:`the quickstart<quick_start>` documentation.
In the quick start, you will learn how to set up an experiment and evaluate the
efficiency of different agents.

For more information see :ref:`the gallery of examples<examples>`.


Tutorials
=========

The tutorials below will present to you the main functionalities of ``rlberry`` in a few minutes.

Installation
------------

.. toctree::
   :maxdepth: 2

   installation.rst


Quick start: setup an experiment and evaluate different agents
--------------------------------------------------------------

.. toctree::
   :maxdepth: 2

   basics/quick_start_rl/quickstart.md
   basics/DeepRLTutorial/TutorialDeepRL.md


Agents, hyperparameter optimization and experiment setup
---------------------------------------------------------

.. toctree::
   :maxdepth: 1

   basics/create_agent.rst
   basics/evaluate_agent.rst
   basics/experiment_setup.rst
   basics/seeding.rst
   basics/multiprocess.rst
   basics/comparison.md

We also provide examples to show how to use :ref:`torch checkpointing<checkpointing_example>`
in rlberry and :ref:`tensorboard<dqn_example>`

Compatibility with External Libraries
=====================================

We provide examples to show you how to use rlberry with:

- :ref:`Gymnasium <Gymnasium_ancor>`;
- :ref:`Stable Baselines <stable_baselines>`.


How to contribute?
==================

If you want to contribute to rlberry, check out :doc:`the contribution guidelines<contributing>`.
