from itertools import combinations
from typing import Tuple, Union
import numpy as np
from scipy.stats import tukey_hsd
import pandas as pd
import rlberry
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

logger = rlberry.logger


# TODO : tukey test for gaussianss
# For now : permutation test + step-down method


def compare_agents(
    agent_manager_list,
    method="tukey_hsd",
    eval_function=None,
    n_simulations=50,
    alpha=0.05,
    B=10_000,
    show=True,
    plot=True,
):
    """
    Compare several trained agents using the mean over n_simulations evaluations for each agent.
    """
    # Evaluation
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

    # Test that the mean evaluations are not all equal
    agent_names = df["agent"].unique()

    data = np.array([df.loc[df["agent"] == name, "mean_eval"] for name in agent_names])
    if method == "tukey_hsd":
        results_pval = tukey_hsd(*tuple(data)).pvalue
    elif method == "permutation":
        results_pval = _permutation_test(data, B, alpha)
    else:
        raise NotImplemented(f"method {method} not implemented")
    if plot:
        mean_eval_values = np.mean(data, axis=1)
        id_sort = np.argsort(mean_eval_values)
        Z = np.array(data)[id_sort]

        links = results_pval <= alpha

        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 2], "hspace": 0}, figsize=(6, 5)
        )

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        vmin, vmax = links.min(), links.max()

        # Draw the heatmap with the mask and correct aspect ratio
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
            yticklabels=np.array(agent_names)[id_sort],
        )

        # Drawing the frame
        for _, spine in res.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)

        # boxplots
        ax2.boxplot(np.array(Z).T, labels=np.array(agent_names)[id_sort])
        if show:
            plt.show()


def _permutation_test(data, B, alpha):
    n_fit = len(data[0])
    n_agents = len(data)

    comparisons = np.array(
        [(i, j) for i in range(n_agents) for j in range(n_agents) if i != j]
    )
    decisions = np.array(["accept" for i in range(len(comparisons))])
    pvals = [np.nan for i in range(len(comparisons))]

    comparisons_alive = np.arange(len(comparisons))

    while True:
        current_comparisons = comparisons[comparisons_alive]
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
                id_comp_max = id_comp

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
            threshold = admissible_values[-1]
        else:
            raise ValueError(
                f"There is not enough fits, the comparisons cannot be done with the precision {alpha}"
            )

        if T0_max > threshold:
            decisions[id_comp_max] = "reject"
            pvals[id_comp_max] = np.sum(Tmax_values > T0_max) / B
            current_comparisons = np.arange(len(comparisons))[decisions == "accept"]
        else:
            break
        if len(current_comparisons) == 0:
            break

    results = np.zeros([n_agents, n_agents])
    for id_comp in range(len(comparisons)):
        if decisions[id_comp_max] == "reject":
            i, j = comparisons[id_comp]
            results[i, j] = pvals[id_comp]
        else:
            i, j = comparisons[id_comp]
            results[i, j] = 0.5
    results = results + results.T + np.eye(n_agents)

    return results
