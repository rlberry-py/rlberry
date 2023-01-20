from itertools import combinations
from typing import Tuple, Union
import numpy as np
from scipy.stats import friedmanchisquare
import pandas as pd
import rlberry
import matplotlib.pyplot as plt
import itertools

logger = rlberry.logger


# TODO : tukey test for gaussianss
# For now : permutation test + step-down method


def compare_agents(
    agent_manager_list, n_simulations=50, alpha=0.05, show=True, plot=True
):
    """
    Compare several trained agents using the mean over n_simulations evaluations for each agent.
    """
    # Evaluation
    df = pd.DataFrame()
    for manager in agent_manager_list:
        n_fit = len(manager.agent_handlers)
        for id_agent in range(n_fit):
            eval_values = manager.eval_agents(100, agent_id=id_agent)
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
    p_values = _permutation_test(data, alpha, B)

    if plot:
        mean_eval_values = np.mean(data, axis=0)
        id_sort = np.argsort(mean_eval_values)
        Z = np.array(data)[id_sort]

        links = p_values > alpha

        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 2], "hspace": 0}, figsize=(6, 5)
        )

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        res = sns.heatmap(
            links,
            cmap=cmap,
            vmax=1,
            center=0,
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
        ax2.boxplot(np.aray(data).T, labels=np.array(agent_names)[id_sort])
        if show:
            plt.show()


def _permutation_test(data, alpha, B):
    n_fit = len(data[0])
    n_agents = len(data)

    comparisons = [(i, j) for i in range(n_agents) for j in range(n_agents) if i != j]
    decisions = np.array(["accept" for i in range(len(comparisons))])

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
            np.arange(len(Tmax_values))[::-1] / self.B
        )  # This corresponds to 1 - F(t) = P(T > t)

        admissible_values = values[icumulative_probas <= alpha]  # acceptance region
        if len(admissible_values_sup) > 0:
            threshold = admissible_values_sup[0]
        else:
            raise ValueError(
                f"There is not enough fits, the comparisons cannot be done with the precision {alpha}"
            )

        if T0_max > threshold:
            decisions[id_comp_max] = "reject"
            current_comparisons = np.arange(len(comparisons))[decisions == "accept"]
        else:
            break
