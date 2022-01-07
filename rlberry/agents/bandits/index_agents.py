import numpy as np
from rlberry.agents import AgentWithSimplePolicy

class IndexAgent(AgentWithSimplePolicy):
    """
    Agent for bandit environment using Index-based policy.

    Parameters
    -----------
    env : rlberry bandit environment

    index_function : callable, default = lambda rew, t : np.mean(rew)
        Compute the index for an arm using the past rewards on this arm and
        the current time t.

    phase : int, default=None
        used to compute "phased bandit" where the index is computed only every
        phase iterations. If None, the bandit is not phased.
    """
    name = 'IndexAgent'
    def __init__(self, env, index_function = lambda rew, t : np.mean(rew),
                 phased = None, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.n_arms = self.env.action_space.n
        self.index_function = index_function
        self.phased = phased
    def fit(self, budget=None, **kwargs):
        n_episodes = budget
        rewards = np.zeros(n_episodes)
        actions = np.ones(n_episodes)*np.nan
        # Initialization : pull all the arms once
        for a in range(self.n_arms):
            action = a
            next_state, reward, done, _ = self.env.step(action)
            rewards[a] = reward
            actions[a] = action
            self.writer.add_scalar('action',action, a)
            if a == n_episodes-1:
                break

        for ep in range(self.n_arms,n_episodes):
            if (phased is None) or (ep % phase == 0):
                indexes = self.get_indexes(rewards, actions, ep+1)
            action = np.argmax(indexes)
            next_state, reward, done, _ = self.env.step(action)
            rewards[ep] = reward
            actions[ep] = action
            self.writer.add_scalar('action',action, ep)

        self.optimal_action = np.argmax(indexes)

        info = {'episode_rewards': rewards}
        return info

    def get_indexes(self, rewards, actions, ep):
        indexes = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            indexes[a] = self.index_function(rewards[actions == a], ep)
        return indexes


    def policy(self, observation):
        return self.optimal_action
