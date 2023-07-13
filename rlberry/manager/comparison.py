from itertools import combinations
import numpy as np
from scipy.stats import tukey_hsd
import pandas as pd
import rlberry
from rlberry.manager import AgentManager
import pathlib

logger = rlberry.logger

# TODO : be able to compare agents from pickle files.


def compare_agents(
    agent_source,
    method="tukey_hsd",
    eval_function=None,
    n_simulations=50,
    alpha=0.05,
    B=10_000,
):
    """
    Compare several trained agents using the mean over n_simulations evaluations for each agent.

    Parameters
    ----------
    agent_source : list of :class:`~rlberry.manager.AgentManager`, list of str

        - If list of AgentManager, load data from it (the agents must be fitted).

        - If str, each string must be the path of a agent_manager.obj.

    method: str in {"tukey_hsd", "permutation"}, default="tukey_hsd"
        Method used in the test. "tukey_hsd" use scipy implementation [1] and "permutation" use permutation test with Step-Down method for multiple testing [2]. Tukey HSD method suppose Gaussian model on the aggregated evaluations and permutation test is non-parametric and does not make assumption on the distribution. permutation is the safe choice when the reward is likely to be heavy-tailed or multimodal.
    eval_function: callable or None, default = None
        Function used the evaluate the agents. lambda manager : AgentManager, eval_budget : int or None, agent_id: int -> eval:float
        If None, the mean of the eval function of the agent is used over n_simulations evaluations.
    n_simulations: int, default = 50
        Number of evaluations to use if eval_function is None.
    alpha: float, default = 0.05
        Level of the test, control the Family-wise error.
    B: int, default = 10_000
        Number of random permutations used to approximate the permutation test if method = "permutation"

    Returns
    -------
    results: a DataFrame summarising the results.

    References
    ----------
    [1]: https://scipy.github.io/devdocs/reference/generated/scipy.stats.tukey_hsd.html

    [2]: Testing Statistical Hypotheses by E. L. Lehmann, Joseph P. Romano (Section 15.4.4), https://doi.org/10.1007/0-387-27605-X, Springer

    """
    # Construction of the array of evaluations
    df = pd.DataFrame()
    assert isinstance(agent_source, list)
    if isinstance(agent_source[0], str) or isinstance(
        agent_source[0], pathlib.PurePath
    ):
        agent_manager_list = [AgentManager(None) for _ in agent_source]
        for i, manager in enumerate(agent_manager_list):
            agent_manager_list[i] = manager.load(agent_source[i])
    else:
        agent_manager_list = agent_source

    for manager in agent_manager_list:
        n_fit = len(manager.agent_handlers)
        for id_agent in range(n_fit):
            if eval_function is None:
                eval_values = manager.eval_agents(100, agent_id=id_agent)
            else:
                eval_values = eval_function(
                    manager, eval_budget=n_simulations, agent_id=id_agent
                )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "mean_eval": [np.mean(eval_values)],
                            "agent": [manager.agent_name],
                        }
                    ),
                ],
                ignore_index=True,
            )

    agent_names = df["agent"].unique()
    assert len(agent_names) == len(
        agent_manager_list
    ), "Each agent must have unique name."
    data = np.array(
        [np.array(df.loc[df["agent"] == name, "mean_eval"]) for name in agent_names]
    )

    n_agents = len(agent_names)
    vs = [
        (agent_names[i], agent_names[j])
        for i in range(n_agents)
        for j in range(n_agents)
        if i < j
    ]

    mean_agent1 = [
        np.mean(df.loc[df["agent"] == vs[i][0], "mean_eval"]) for i in range(len(vs))
    ]

    mean_agent2 = [
        np.mean(df.loc[df["agent"] == vs[i][1], "mean_eval"]) for i in range(len(vs))
    ]
    mean_diff = [
        np.mean(
            np.array(df.loc[df["agent"] == vs[i][0], "mean_eval"])
            - np.array(df.loc[df["agent"] == vs[i][1], "mean_eval"])
        )
        for i in range(len(vs))
    ]
    std_diff = [
        np.std(
            np.array(df.loc[df["agent"] == vs[i][0], "mean_eval"])
            - np.array(df.loc[df["agent"] == vs[i][1], "mean_eval"])
        )
        for i in range(len(vs))
    ]

    # Use a multiple test to compare the agents
    if method == "tukey_hsd":
        results_pval = tukey_hsd(*tuple(data)).pvalue
        p_vals = [
            results_pval[i][j]
            for i in range(n_agents)
            for j in range(n_agents)
            if i < j
        ]
        significance = [
            sig_p_value(results_pval[i][j])
            for i in range(n_agents)
            for j in range(n_agents)
            if i < j
        ]
        decisions = ["accept" if p_val >= alpha else "reject" for p_val in p_vals]
        results = pd.DataFrame(
            {
                "Agent1 vs Agent2": [
                    "{0} vs {1}".format(vs[i][0], vs[i][1]) for i in range(len(vs))
                ],
                "mean Agent1": mean_agent1,
                "mean Agent2": mean_agent2,
                "mean diff": mean_diff,
                "std diff": std_diff,
                "decisions": decisions,
                "p-val": p_vals,
                "significance": significance,
            }
        )
    elif method == "permutation":
        results_perm = _permutation_test(data, B, alpha) == 1
        decisions = ["accept" if res else "reject" for res in results_perm]
        results = pd.DataFrame(
            {
                "Agent1 vs Agent2": [
                    "{0} vs {1}".format(vs[i][0], vs[i][1]) for i in range(len(vs))
                ],
                "mean Agent1": mean_agent1,
                "mean Agent2": mean_agent2,
                "mean diff": mean_diff,
                "std diff": std_diff,
                "decisions": decisions,
            }
        )
    else:
        raise NotImplemented(f"method {method} not implemented")

    return results


def _permutation_test(data, B, alpha):
    """
    Permutation test with Step-Down method
    """
    n_fit = len(data[0])
    n_agents = len(data)

    # We do all the pairwise comparisons.
    comparisons = np.array(
        [(i, j) for i in range(n_agents) for j in range(n_agents) if i < j]
    )

    decisions = np.array(["accept" for i in range(len(comparisons))])
    comparisons_alive = np.arange(len(comparisons))

    logger.info("Beginning permutationt test")
    while True:
        current_comparisons = comparisons[comparisons_alive]
        logger.info(f"Still {len(current_comparisons)} comparisons to test")

        # make a generator of permutations
        if B is None:
            permutations = combinations(2 * n_fit, n_fit)
        else:
            permutations = (np.random.permutation(2 * n_fit) for _ in range(B))

        # Test statistics
        T0_max = 0
        for id_comp, (i, j) in enumerate(current_comparisons):
            Z = np.hstack([data[i], data[j]])
            T = np.abs(np.mean(Z[:n_fit]) - np.mean(Z[n_fit : (2 * n_fit)]))
            if T > T0_max:
                T0_max = T
                id_comp_max = comparisons_alive[id_comp]

        # Permutation distribution of Tmax
        Tmax_values = []
        for perm in permutations:
            Tmax = 0
            for id_comp, (i, j) in enumerate(current_comparisons):
                Z = np.hstack([data[i], data[j]])
                Z = Z[perm]
                T = np.abs(np.mean(Z[:n_fit]) - np.mean(Z[n_fit : (2 * n_fit)]))
                if T > Tmax:
                    Tmax = T
            Tmax_values.append(Tmax)

        Tmax_values = np.sort(Tmax_values)
        icumulative_probas = (
            np.arange(len(Tmax_values))[::-1] / B
        )  # This corresponds to 1 - F(t) = P(T > t)

        admissible_values = Tmax_values[
            icumulative_probas <= alpha
        ]  # acceptance region
        if len(admissible_values) > 0:
            threshold = np.min(admissible_values)
        else:
            raise ValueError(
                f"There is not enough fits, the comparisons cannot be done with the precision {alpha}"
            )

        if T0_max > threshold:
            assert decisions[id_comp_max] == "accept"
            decisions[id_comp_max] = "reject"
            comparisons_alive = np.arange(len(comparisons))[decisions == "accept"]
        else:
            break
        if len(comparisons_alive) == 0:
            break

    # make a result array with 1 if accept and 0 if reject.
    results = np.zeros([n_agents, n_agents])
    for id_comp in range(len(comparisons)):
        if decisions[id_comp] == "reject":
            i, j = comparisons[id_comp]
            results[i, j] = 0
        else:
            i, j = comparisons[id_comp]
            results[i, j] = 1

    results = results + results.T + np.eye(n_agents)

    return results


def sig_p_value(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p <= 0.05:
        return "*"
    else:
        return ""
