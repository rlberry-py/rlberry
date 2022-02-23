import numpy as np
from scipy import stats

from rlberry.envs.bandits import Bandit


class NormalBandit(Bandit):
    """
    Class for Normal Bandits

    Parameters
    ----------

    means: array-like of size n_arms, default=array([0,1])
        means of the law of each of the arms.

    stds: array-like of size n_arms or None, default=None
        stds of the law of each of the arms. If None, use array with
        all ones.

    """

    def __init__(
        self,
        means=np.array([0, 1]),
        stds=None,
    ):
        laws = self.make_laws(means, stds)
        Bandit.__init__(self, laws=laws)

    def make_laws(self, means, stds):
        if stds is None:
            self.stds = np.ones(len(means))
        else:
            self.stds = stds
        assert len(means) == len(self.stds)
        return [stats.norm(loc=means[a], scale=self.stds[a]) for a in range(len(means))]


class BernoulliBandit(Bandit):
    """
    Class for Bernoulli Bandits

    Parameters
    ----------

    p: array-like of size n_arms, default=array([0.1,0.9])
        means of the law of inliers of each of the arms.

    """

    def __init__(
        self,
        p=np.array([0.1, 0.9]),
    ):
        laws = self.make_laws(p)
        Bandit.__init__(self, laws=laws)

    def make_laws(self, p):
        return [stats.binom(n=1, p=p[a]) for a in range(len(p))]
