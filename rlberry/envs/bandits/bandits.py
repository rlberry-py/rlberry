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
        has a method .rvs().

    **kwargs: keywords arguments
        additional arguments sent to :class:`~rlberry.envs.interface.Model`

    """

    name = ""

    def __init__(self, laws=[], **kwargs):
        Model.__init__(self, **kwargs)
        self.laws = laws
        A = len(self.laws)
        self.action_space = spaces.Discrete(A)

    def step(self, action):
        """
        Sample the reward associated to the action.
        """
        # test that the action exists
        assert action < len(self.laws)

        reward = self.laws[action].rvs(random_state=self.rng)
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
