from itertools import combinations
import numpy as np
from scipy.stats import tukey_hsd
import pandas as pd
import rlberry
import matplotlib.pyplot as plt
import seaborn as sns

logger = rlberry.logger


# TODO : be able to compare agents from pickle files.


def compare_agents(
    agent_manager_list,
    method="tukey_hsd",
    eval_function=None,
    n_simulations=50,
    alpha=0.05,
    B=10_000,
    plot=True,
    show=True,
):
    """
    Compare several trained agents using the mean over n_simulations evaluations for each agent.

    Parameters
    ----------
    agent_manager_list : list of AgentManager objects.
    method: str in {"tukey_hsd", "permutation"}, default="tukey_hsd"
        Method used in the test. "tukey_hsd" use scipy implementation [1] and "permutation" use permutation test with Step-Down method for multiple testing [2].
    eval_function: callable or None, default = None
        Function used the evaluate the agents. lambda manager : AgentManager, eval_budget : int or None, agent_id: int -> eval:float
        If None, the mean of the eval function of the agent is used over n_simulations evaluations.
    n_simulations: int, default = 50
        Number of evaluations to use if eval_function is None.
    alpha: float, default = 0.05
        Level of the test
    B: int, default = 10_000
        Number of random permutations used to approximate the permutation test if method = "permutation"
    plot: boolean, default = True
        If True, make a plot to illustrate the results.
    show: boolean, default = True
        If True, use plt.show() at the end.

    Returns
    -------
    decisions: 2D array of noolean for each comparison of 2 agents
        decisions of the test, True is accept and False is reject.

    References
    ----------
    [1]: https://scipy.github.io/devdocs/reference/generated/scipy.stats.tukey_hsd.html

    [2]: Testing Statistical Hypotheses by E. L. Lehmann, Joseph P. Romano (Section 15.4.4), https://doi.org/10.1007/0-387-27605-X, Springer

    Examples
    --------
    >>> for manager in managers :
    >>> manager.fit()
    >>>
    >>> def eval_function(manager , eval_budget = None, agent_id=0):
    >>>     # Compute the sum of regrets over training
    >>>     df = manager.get_writer_data()[agent_id]
    >>>     return T *np.max(means) - np.sum(df.loc[df['tag']=="reward", "value"])
    >>> compare_agents(managers, method = "permutation", eval_function=eval_function, B = 10_000)

    """
    # Construction of the array of evaluations
    df = pd.DataFrame()
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
    data = np.array([df.loc[df["agent"] == name, "mean_eval"] for name in agent_names])

    # Use a multiple test to compare the agents
    if method == "tukey_hsd":
        results_pval = tukey_hsd(*tuple(data)).pvalue
    elif method == "permutation":
        results_pval = _permutation_test(data, B, alpha)
    else:
        raise NotImplemented(f"method {method} not implemented")

    # Do the plot of the results
    if plot:
        mean_eval_values = np.mean(data, axis=1)
        id_sort = np.argsort(mean_eval_values)
        Z = np.array(data)[id_sort]

        links = results_pval <= alpha

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        vmin, vmax = links.min(), links.max()

        # Draw the heatmap with the mask and correct aspect ratio
        if method == "tukey_hsd":
            res = sns.heatmap(
                links[id_sort, :][:, id_sort],
                annot=results_pval[id_sort, :][:, id_sort],
                cmap=cmap,
                vmin=0,
                vmax=1,
                center=0.5,
                linewidths=0.5,
                ax=ax1,
                cbar=False,
                xticklabels=np.array(agent_names)[id_sort],
                yticklabels=np.array(agent_names)[id_sort],
            )
            ax1.set_title(
                "decision and adapted p-values\nred = reject and blue = accept"
            )
        else:
            res = sns.heatmap(
                links[id_sort, :][:, id_sort],
                cmap=cmap,
                vmin=0,
                vmax=1,
                center=0.5,
                linewidths=0.5,
                ax=ax1,
                cbar=False,
                xticklabels=np.array(agent_names)[id_sort],
                yticklabels=np.array(agent_names)[id_sort],
            )
            ax1.set_title("red = reject and blue = accept")

        # boxplots
        ax2.boxplot(np.array(Z).T, labels=np.array(agent_names)[id_sort])
        ax2.set_title("Boxplots of agents evaluations")
        fig.subplots_adjust(bottom=0.2)
        fig.suptitle(
            "Results for " + method + " multiple test",
            y=0.05,
            verticalalignment="bottom",
        )
        if show:
            plt.show()
        return links


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
