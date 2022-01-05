import numpy as np
from rlberry.agents import AgentWithSimplePolicy

class IndexAgent(AgentWithSimplePolicy):
    """
    Agent for bandit environment using Index-based policy.

    Parameters
    -----------
    env : rlberry bandit environment

    index_function : callable, default = np.mean
            compute the index for an arm using the past rewards on this arm.
    """
    name = 'IndexAgent'
    def __init__(self, env, index_function = np.mean, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.n_arms = self.env.action_space.n
        self.index_function = index_function
    def fit(self, budget=None, **kwargs):
        n_episodes = budget
        rewards = np.zeros(n_episodes)
        actions = np.zeros(n_episodes)
        # Initialization : pull all the arms once
        for a in range(self.n_arms):
            action = a
            next_state, reward, done, _ = self.env.step(action)
            rewards[a] = reward
            actions[a] = action
            if a == n_episodes-1:
                break

        for ep in range(self.n_arms,n_episodes):
            indexes = self.get_indexes(rewards, actions)
            action = np.argmax(indexes)
            next_state, reward, done, _ = self.env.step(action)
            rewards[ep] = reward
            actions[ep] = action

        self.optimal_action = np.argmax(indexes)

        info = {'episode_rewards': rewards}
        return info

    def get_indexes(self, rewards, actions):
        indexes = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            indexes[a] = self.index_function(rewards[actions == a])
        return indexes


    def policy(self, observation):
        return self.optimal_action

    
