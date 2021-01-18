.. _rlberry: https://github.com/rlberry-py/rlberry
.. _`OpenAI Gym`: https://stable-baselines.readthedocs.io/en/master/

.. _gym:


Using rlberry_ and `OpenAI Gym`_
================================

If you want to use OpenAI Gym environments with rlberry_, simply do the following:

.. code-block:: python

   from rlberry.envs import gym_make  #  wraps gym.make

   # for example, let's take CartPole
   env = gym_make('CartPole-v1')

This way, :code:`env` **behaves exactly the same as the gym environment**, we simply replace the seeding
function by :meth:`env.reseed`, which ensures unified seeding and reproducibility when using rlberry.
