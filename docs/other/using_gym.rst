.. _rlberry: https://github.com/rlberry-py/rlberry
.. _Gymnasium: https://gymnasium.farama.org/


Using rlberry_ and Gymnasium_  
================================

If you want to use Gymnasium_ environments with rlberry_, simply do the following:

.. code-block:: python

   from rlberry.envs import gym_make  #  wraps gym.make

   # for example, let's take CartPole
   env = gym_make("CartPole-v1")

This way, :code:`env` **behaves exactly the same as the gym environment**, we simply replace the seeding
function by :meth:`env.reseed`, which ensures unified seeding and reproducibility when using rlberry.
