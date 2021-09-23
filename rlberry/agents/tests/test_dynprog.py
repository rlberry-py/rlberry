import numpy as np
import pytest

import rlberry.seeding as seeding
from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.agents.dynprog.utils import backward_induction
from rlberry.agents.dynprog.utils import backward_induction_in_place
from rlberry.agents.dynprog.utils import backward_induction_sd
from rlberry.agents.dynprog.utils import bellman_operator
from rlberry.agents.dynprog.utils import value_iteration
from rlberry.envs.finite import FiniteMDP

_rng = seeding.Seeder(123).rng


def get_random_mdp(S, A):
    R = _rng.uniform(0.0, 1.0, (S, A))
    P = _rng.uniform(0.0, 1.0, (S, A, S))
    for ss in range(S):
        for aa in range(A):
            P[ss, aa, :] /= P[ss, aa, :].sum()
    return R, P


@pytest.mark.parametrize("gamma, S, A",
                         [
                             (0.001, 2, 1),
                             (0.25, 2, 1),
                             (0.5, 2, 1),
                             (0.75, 2, 1),
                             (0.999, 2, 1),
                             (0.001, 4, 2),
                             (0.25, 4, 2),
                             (0.5, 4, 2),
                             (0.75, 4, 2),
                             (0.999, 4, 2),
                             (0.001, 20, 4),
                             (0.25, 20, 4),
                             (0.5, 20, 4),
                             (0.75, 20, 4),
                             (0.999, 20, 4)
                         ])
def test_bellman_operator_monotonicity_and_contraction(gamma, S, A):
    rng = seeding.Seeder(123).rng
    vmax = 1.0 / (1.0 - gamma)
    for _ in range(10):
        # generate random MDP
        R, P = get_random_mdp(S, A)

        # generate random Q functions
        Q0 = rng.uniform(-vmax, vmax, (S, A))
        Q1 = rng.uniform(-vmax, vmax, (S, A))
        # apply Bellman operator
        TQ0 = bellman_operator(Q0, R, P, gamma)
        TQ1 = bellman_operator(Q1, R, P, gamma)

        # test contraction
        norm_tq = np.abs(TQ1 - TQ0).max()
        norm_q = np.abs(Q1 - Q0).max()
        assert norm_tq <= gamma * norm_q

        # test monotonicity
        Q2 = rng.uniform(-vmax / 2, vmax / 2, (S, A))
        Q3 = Q2 + rng.uniform(0.0, vmax / 2, (S, A))
        TQ2 = bellman_operator(Q2, R, P, gamma)
        TQ3 = bellman_operator(Q3, R, P, gamma)
        assert np.greater(TQ2, TQ3).sum() == 0


@pytest.mark.parametrize("gamma, S, A",
                         [
                             (0.01, 10, 4),
                             (0.25, 10, 4),
                             (0.5, 10, 4),
                             (0.75, 10, 4),
                             (0.99, 10, 4)
                         ])
def test_value_iteration(gamma, S, A):
    for epsilon in np.logspace(-1, -6, num=5):
        for sim in range(5):
            # generate random MDP
            R, P = get_random_mdp(S, A)

            # run value iteration
            Q, V, n_it = value_iteration(R, P, gamma, epsilon)
            # check precision
            TQ = bellman_operator(Q, R, P, gamma)
            assert np.abs(TQ - Q).max() <= epsilon


@pytest.mark.parametrize("horizon, S, A",
                         [
                             (10, 5, 4),
                             (20, 10, 4)
                         ])
def test_backward_induction(horizon, S, A):
    for sim in range(5):
        # generate random MDP
        R, P = get_random_mdp(S, A)

        # run backward induction
        Q, V = backward_induction(R, P, horizon)

        assert Q.max() <= horizon
        assert V.max() <= horizon

        # run backward with clipping V to 1.0
        Q, V = backward_induction(R, P, horizon, vmax=1.0)
        assert V.max() <= 1.0

        # run bacward induction in place
        Q2 = np.zeros((horizon, S, A))
        V2 = np.zeros((horizon, S))
        backward_induction_in_place(Q2, V2, R, P, horizon, vmax=1.0)
        assert np.array_equal(Q, Q2)
        assert np.array_equal(V, V2)


@pytest.mark.parametrize("horizon, S, A",
                         [
                             (10, 5, 4),
                             (20, 10, 4)
                         ])
def test_backward_induction_sd(horizon, S, A):
    """
    Test stage-dependent MDPs
    """
    for sim in range(5):
        # generate random MDP
        Rstat, Pstat = get_random_mdp(S, A)
        R = np.zeros((horizon, S, A))
        P = np.zeros((horizon, S, A, S))
        for ii in range(horizon):
            R[ii, :, :] = Rstat
            P[ii, :, :, :] = Pstat

        # run backward induction in stationary MDP
        Qstat, Vstat = backward_induction(Rstat, Pstat, horizon)

        # run backward induction in statage-dependent MDP
        Q = np.zeros((horizon, S, A))
        V = np.zeros((horizon, S))
        backward_induction_sd(Q, V, R, P)

        assert np.array_equal(Q, Qstat)
        assert np.array_equal(V, Vstat)


@pytest.mark.parametrize("horizon, gamma, S, A",
                         [
                             (None, 0.5, 10, 4),
                             (10, 1.0, 10, 4)
                         ])
def test_value_iteration_agent(horizon, gamma, S, A):
    for sim in range(5):
        # generate random MDP
        R, P = get_random_mdp(S, A)
        # create env and agent
        env = FiniteMDP(R, P)
        agent = ValueIterationAgent(env, gamma=gamma, horizon=horizon)
        # run
        agent.fit()
