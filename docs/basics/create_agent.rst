.. _rlberry: https://github.com/rlberry-py/rlberry

.. _create_agent:


Create an agent
===============

rlberry_ requires you to use a **very simple interface** to write agents, with basically
two methods to implement: :code:`fit()` and :code:`eval()`.

The example below shows how to create an agent. 


.. code-block:: python

    import numpy as np
    from rlberry.agents import Agent

    class MyAgent(Agent):

        name = "MyAgent"

        def __init__(self,
                     env,
                     param1=0.99,
                     param2=1e-5,
                     **kwargs):   # it's important to put **kwargs to ensure compatibility with the base class
                # self.env is initialized in the base class
                # An evaluation environment is also initialized: self.eval_env
                Agent.__init__(self, env, **kwargs)

                self.param1 = param1
                self.param2 = param2 

        def fit(self, budget, **kwargs):  
            """
            The parameter budget can represent the number of steps, the number of episodes etc,
            depending on the agent.
            * Interact with the environment (self.env); 
            * Train the agent
            * Return useful information
            """
            n_episodes = budget
            rewards = np.zeros(n_episodes)

            for ep in range(n_episodes):
                state, info = self.env.reset()
                done = False
                while not done:
                action = ...  
                observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                rewards[ep] += reward

            info = {'episode_rewards': rewards}
            return info

        def eval(self, **kwargs):
            """
            Returns a value corresponding to the evaluation of the agent on the 
            evaluation environment.

            For instance, it can be a Monte-Carlo evaluation of the policy learned in fit().
            """
            return 0.0


.. note:: It's important that your agent accepts optional `**kwargs` and pass it to the base class as :code:`Agent.__init__(self, env, **kwargs)`. 


.. seealso::
    Documentation of the classes :class:`~rlberry.agents.agent.Agent` 
    and :class:`~rlberry.agents.agent.AgentWithSimplePolicy`.
