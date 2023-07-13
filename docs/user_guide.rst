.. title:: User guide : contents

.. _user_guide:

==========
User guide
==========

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

   basics/quick_start_rl/quickstart.rst
   basics/DeepRLTutorial/TutorialDeepRL.rst


Agents, hyperparameter optimization and experiment setup
---------------------------------------------------------

.. toctree::
   :maxdepth: 1

   basics/create_agent.rst
   basics/evaluate_agent.rst
   basics/compare_agents.rst
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
