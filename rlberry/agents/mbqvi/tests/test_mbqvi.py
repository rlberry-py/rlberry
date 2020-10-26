import numpy as np
import pytest

import rlberry.seeding as seeding
from rlberry.agents.mbqvi import MBQVIAgent
from rlberry.envs.finite import FiniteMDP


@pytest.mark.parametrize("S, A", [(5, 2), (10, 4)])
def test_mbqvi(S, A):
    rng = seeding.get_rng()

    for sim in range(5):
        # generate random MDP with deterministic transitions
        R = rng.uniform(0.0, 1.0, (S, A))
        P = np.zeros((S, A, S))
        for ss in range(S):
            for aa in range(A):
                ns = rng.integers(0, S)
                P[ss, aa, ns] = 1

        # run MBQVI and check exactness of estimators
        env = FiniteMDP(R, P)
        agent = MBQVIAgent(env, n_samples=1)
        agent.fit()
        assert np.abs(R - agent.R_hat).max() < 1e-16
        assert np.abs(P - agent.P_hat).max() < 1e-16
