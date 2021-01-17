.. _rlberry: https://github.com/rlberry-py/rlberry

Create an agent
###############

rlberry_ requires you to use a **very simple interface** to write agents, with basically
two methods to implement: :code:`fit()` and :code:`policy()`.

The example below shows how to create an agent. 


.. code-block:: python

    import numpy as np
    from rlberry.agents import Agent

    class MyAgent(Agent):

        name = "MyAgent"

        def __init__(self,
                     env,
                     n_episodes=100,
                     param1=0.99,
                     param2=1e-5,
                     **kwargs):   # it's important to put **kwargs to ensure compatibility with the base class 
                Agent.__init__(self, env, **kwargs) # self.env is initialized in the base class

                self.n_episodes = n_episodes
                self.param1 = param1
                self.param2 = param2 

        def fit(self, **kwargs):  
            """
            * Interact with the environment (self.env); 
            * Train the agent
            * Return useful information
            """
            rewards = np.zeros(self.n_episodes)

            for ep in range(self.n_episodes):
                state = self.env.reset()
                done = False
                while not done:
                action = ...  
                next_state, reward, done, _ = self.env.step(action)
                rewards[ep] += reward

            info = {'episode_rewards': rewards}
            return info

        def policy(self, observation, **kwargs):
            """
            Given an observation, return an action.
            """
            return self.env.action_space.sample()


.. note:: It's important that your agent accepts optional `**kwargs` and pass it to the base class as :code:`Agent.__init__(self, env, **kwargs)`. 


.. seealso::
    Documentation of the :class:`~rlberry.agents.agent.Agent` class.
