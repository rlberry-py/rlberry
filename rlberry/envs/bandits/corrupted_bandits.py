import numpy as np
from scipy import stats

from rlberry.envs.bandits import Bandit


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

    def rvs(self, random_state):
        is_corrupted = random_state.binomial(1, self.cor_prop)
        if is_corrupted == 1:
            return self.cor_law.rvs(random_state=random_state)
        else:
            return self.law.rvs(random_state=random_state)

    def mean(self):
        return (
            1 - self.cor_prop
        ) * self.law.mean() + self.cor_prop * self.cor_law.mean()


class CorruptedNormalBandit(Bandit):
    """
    Class for Bandits corrupted by nature.

    Parameters
    ----------
    means: array-like of size n_arms, default=array([0,1])
        means of the law of inliers of each of the arms.

    stds: array-like of size n_arms or None, default=None
        stds of the law of inliers of each of the arms. If None, use array with
        all ones.

    cor_prop: float in (0,1/2), default=0.05
        proportion of corruption

    cor_laws: list of scipy frozen laws or None, default=None
        laws of corruption on each arm. If None, all the arms are corrupted by
        a normal of mean 1000 and std 1.
    """

    def __init__(
        self,
        means=np.array([0, 1]),
        stds=None,
        cor_prop=0.05,
        cor_laws=None,
    ):
        laws = self.make_laws(means, stds, cor_prop, cor_laws)
        Bandit.__init__(self, laws=laws)

    def make_laws(self, means, stds, cor_prop, cor_laws):
        if cor_laws is not None:
            self.cor_laws = cor_laws
        else:
            self.cor_laws = [stats.norm(loc=1000) for a in range(len(means))]
        if stds is None:
            self.stds = np.ones(len(means))
        else:
            self.stds = stds
        assert len(means) == len(self.stds)
        assert cor_prop <= 0.5
        inlier_laws = [
            stats.norm(loc=means[a], scale=self.stds[a]) for a in range(len(means))
        ]
        return [
            CorruptedLaws(inlier_laws[a], cor_prop, self.cor_laws[a])
            for a in range(len(means))
        ]
