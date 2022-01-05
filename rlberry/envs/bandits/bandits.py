import numpy as np
from scipy import stats
import logging

from rlberry.envs.interface import Model
import rlberry.spaces as spaces


logger = logging.getLogger(__name__)


class Bandit(Model):
    """
    Base class for a multi-armed Bandit.

    Parameters
    ----------

    laws: list of laws.
        laws of the arms. can either be a frozen scipy law or any class that
        has a method .rvs() to sample according to the given law.

    """

    def __init__(self, laws=[]):
        Model.__init__(self)
        self.laws = laws

    def step(self, action):
        # test that the action exists
        assert action < len(self.laws)

        reward = self.laws[action].rvs(rng=self.rng)
        done = True

        return 0, reward, done, {}

    def reset(self):
        """
        Reset the environment to a default state.
        """
        return 0


class CorruptedLaws:
    """
    Class for corrupted laws.

    Parameters
    ----------

    law: law
        Can either be a frozen scipy law or any class that
        has a method .rvs() to sample according to the given law.

    cor_prop: float in (0,1/2)
        Proportion of corruption

    cor_law: law
        Laws of corruption.
    """

    def __init__(self, law, cor_prop, cor_law):
        self.law = law
        self.cor_prop = cor_prop
        self.cor_law = cor_law

    def rvs(self, rng):
        is_corrupted = rng.binomial(1, self.cor_prop)
        if is_corrupted == 1:
            return self.cor_law.rvs(random_state=rng)
        else:
            return self.law.rvs(random_state=rng)


class CorruptedNormalBandit(Bandit):
    """
    Class for Bandits corrupted by nature.

    Parameters
    ----------

    means: array-like of size n_arms, default=array([0,1])
        means of the law of inliers of each of the arms.

    stds: array-like of size n_arms, default=array([1,1])
        stds of the law of inliers of each of the arms.

    cor_prop: float in (0,1/2), default=0.05
        proportion of corruption

    cor_laws: list of scipy frozen laws or None, default=None
        laws of corruption on each arm. If None, all the arms are corrupted by
        a normal of mean 1000 and std 1.
    """

    def __init__(
        self,
        means=np.array([0, 1]),
        stds=np.array([1, 1]),
        cor_prop=0.05,
        cor_laws=None,
    ):
        Bandit.__init__(self)
        self.laws = self.make_laws(means, stds, cor_prop, cor_laws)
        A = len(self.laws)
        self.action_space = spaces.Discrete(A)
        self._actions = np.arange(A)

    def make_laws(self, means, stds, cor_prop, cor_laws):
        if cor_laws is not None:
            self.cor_laws = cor_laws
        else:
            self.cor_laws = [stats.norm(loc=1000) for a in range(len(means))]
        assert len(means) == len(stds)
        assert cor_prop <= 0.5
        inlier_laws = [
            stats.norm(loc=means[a], scale=stds[a]) for a in range(len(means))
        ]
        return [
            CorruptedLaws(inlier_laws[a], cor_prop, self.cor_laws[a])
            for a in range(len(means))
        ]
