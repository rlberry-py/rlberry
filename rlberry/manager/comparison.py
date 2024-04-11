from itertools import combinations
import numpy as np
from scipy.stats import tukey_hsd
import pandas as pd
import rlberry
from rlberry.manager import ExperimentManager
from rlberry.seeding import Seeder
import pathlib

from adastop import MultipleAgentsComparator

logger = rlberry.logger


class AdastopComparator(MultipleAgentsComparator):
    """
    Compare sequentially agents, with possible early stopping.
    At maximum, there can be n times K fits done.

    See adastop library for more details (https://github.com/TimotheeMathieu/adastop)

    Parameters
    ----------
    n: int, or array of ints of size self.n_agents, default=5
        If int, number of fits before each early stopping check. If array of int, a
        different number of fits is used for each agent.

    K: int, default=5
        number of check.

    B: int, default=None
        Number of random permutations used to approximate permutation distribution.

    comparisons: list of tuple of indices or None
        if None, all the pairwise comparison are done.
        If = [(0,1), (0,2)] for instance, the compare only 0 vs 1  and 0 vs 2

    alpha: float, default=0.01
        level of the test

    beta: float, default=0
        power spent in early accept.

    seed: int or None, default = None

    Attributes
    ----------
    agent_names: list of str
        list of the agents' names.
    managers_paths: dictionary
        managers_paths[agent_name] is a list of the paths to the trained experiment managers. Can be loaded with ExperimentManager.load.
    decision: dict
        decision of the tests for each comparison, keys are the comparisons and values are in {"equal", "larger", "smaller"}.
    n_iters: dict
        number of iterations (i.e. number of fits) used for each agent. Keys are the agents' names and values are ints.
    """

    def __init__(
        self,
        n=5,
        K=5,
        B=10000,
        comparisons=None,
        alpha=0.01,
        beta=0,
        seed=None,
    ):
        MultipleAgentsComparator.__init__(self, n, K, B, comparisons, alpha, beta, seed)
        self.managers_paths = {}

    def compare(self, manager_list, n_evaluations=50, verbose=True):
        """
        Run Adastop on the managers from manager_list

        Parameters
        ----------
        manager_list: list of ExperimentManager kwargs
            List of manager containing agents we want to compare.
        n_evaluations: int, default = 50
            number of evaluations used to estimate the score used for AdaStop.
        verbose: bool
            Print Steps.
        Returns
        -------
        decisions: dictionary with comparisons as index and with values str in {"equal", "larger", "smaller", "continue"}
           Decision of the test at this step.
        """
        eval_values = {
            ExperimentManager(**manager).agent_name: [] for manager in manager_list
        }
        self.managers_paths = {
            ExperimentManager(**manager).agent_name: [] for manager in manager_list
        }
        self.n_evaluations = n_evaluations
        seeder = Seeder(self.rng.randint(10000))
        seeders = seeder.spawn(len(manager_list) * self.K + 1)
        self.rng = seeders[-1].rng
        for k in range(self.K):
            eval_values = self._fit_evaluate(manager_list, eval_values, seeders)
            self.partial_compare(eval_values, verbose=True)
            if self.is_finished:
                break

        logger.info("Test finished")
        logger.info("Results are ")
        print(self.get_results())

    def print_results(self):
        """
        Print the results of the test.
        """
        print("Number of scores used for each agent:")
        for key in self.n_iters:
            print(key + ":" + str(self.n_iters[key]))

        print("")
        print("Mean of scores of each agent:")
        for key in self.eval_values:
            print(key + ":" + str(np.mean(self.eval_values[key])))

        print("")
        print("Decision for each comparison:")
        for c in self.comparisons:
            print(
                "{0} vs {1}".format(self.agent_names[c[0]], self.agent_names[c[1]])
                + ":"
                + str(self.decisions[str(c)])
            )

    def _fit_evaluate(self, managers, eval_values, seeders):
        """
        fit rlberry agents.
        """
        if isinstance(self.n, int):
            self.n = np.array([self.n] * len(managers))

        for i, kwargs in enumerate(managers):
            kwargs["n_fit"] = self.n[i]
        managers_in = []
        agent_names_in = []
        for i in range(len(managers)):
            if (self.current_comparisons is None) or (
                i in np.array(self.current_comparisons).ravel()
            ):
                manager_kwargs = managers[i]
                seeder = seeders[i]
                managers_in.append(ExperimentManager(**manager_kwargs, seed=seeder))
                agent_names_in.append(managers_in[-1].agent_name)

        if self.agent_names is None:
            self.agent_names = agent_names_in

        if len(set(self.agent_names)) != len(self.agent_names):
            raise ValueError("Error: there must be different names for each agent.")

        # Fit all the agents
        managers_in = [_fit_agent(manager) for manager in managers_in]

        # Save the managers' save path
        for i in range(len(managers_in)):
            self.managers_paths[agent_names_in[i]] = managers_in[i].save()

        # Get the evaluations
        idz = 0
        for i in range(len(managers_in)):
            eval_values[agent_names_in[i]] = np.hstack(
                [
                    eval_values[agent_names_in[i]],
                    self._get_evals(managers_in[i], self.n[i]),
                ]
            )

        return eval_values

    def _get_evals(self, manager, n):
        """
        Can be overwritten for alternative evaluation function.
        """
        eval_values = []
        for idx in range(n):
            logger.info("Evaluating agent " + str(idx))
            eval_values.append(
                np.mean(manager.eval_agents(self.n_evaluations, agent_id=idx))
            )
        return eval_values


def _fit_agent(manager):
    manager.fit()
    return manager


# TODO : be able to compare agents from dataframes and from pickle files.
def compare_agents(
    agent_source,
    method="tukey_hsd",
    eval_function=None,
    n_simulations=50,
    alpha=0.05,
    B=10_000,
    seed=None,
):
    """
    Compare several trained agents using the mean over n_simulations evaluations for each agent.

    Parameters
    ----------
    agent_source : list of :class:`~rlberry.manager.ExperimentManager`, list of str
        - If list of ExperimentManager, load data from it (the agents must be fitted).
        - If str, each string must be the path of a agent_manager.obj.
        - If pandas DataFrame with column agent (containing agent's names) and mean_eval containing
          the scores, it is used as input data.
        **Each agent must have unique name.**

    method: str in {"tukey_hsd", "permutation"}, default="tukey_hsd"
        Method used in the test. "tukey_hsd" use scipy implementation [1] and "permutation" use permutation test with Step-Down method for multiple testing [2]. Tukey HSD method suppose Gaussian model on the aggregated evaluations and permutation test is non-parametric and does not make assumption on the distribution. permutation is the safe choice when the reward is likely to be heavy-tailed or multimodal.
    eval_function: callable or None, default = None
        Function used the evaluate the agents. lambda manager : ExperimentManager, eval_budget : int or None, agent_id: int -> eval:float
        If None, the mean of the eval function of the agent is used over n_simulations evaluations.
    n_simulations: int, default = 50
        Number of evaluations to use if eval_function is None.
    alpha: float, default = 0.05
        Level of the test, control the Family-wise error.
    B: int, default = 10_000
        Number of random permutations used to approximate the permutation test if method = "permutation"
    seed: int or None,
        The seed of the random number generator from which we sample permutations. If None, create one.

    Returns
    -------
    results: a DataFrame summarising the results.

    References
    ----------
    [1]: https://scipy.github.io/devdocs/reference/generated/scipy.stats.tukey_hsd.html

    [2]: Testing Statistical Hypotheses by E. L. Lehmann, Joseph P. Romano (Section 15.4.4), https://doi.org/10.1007/0-387-27605-X, Springer

    """

    if isinstance(agent_source, list):
        df_agent_data = preprocess_agent_data(
            agent_source, eval_function, n_simulations
        )
    else:
        df_agent_data = agent_source
    return compare_agents_data(df_agent_data, method, alpha, B, seed)


def preprocess_agent_data(
    agent_source,
    eval_function=None,
    n_simulations=50,
):
    """
    Compare several trained agents using the mean over n_simulations evaluations for each agent.

    Parameters
    ----------
    agent_source : list of :class:`~rlberry.manager.ExperimentManager`, list of str
        - If list of ExperimentManager, load data from it (the agents must be fitted).
        - If str, each string must be the path of a agent_manager.obj.
        **Each agent must have unique name.**
    eval_function: callable or None, default = None
        Function used the evaluate the agents. lambda manager : ExperimentManager, eval_budget : int or None, agent_id: int -> eval:float
        If None, the mean of the eval function of the agent is used over n_simulations evaluations.
    n_simulations: int, default = 50
        Number of evaluations to use if eval_function is None.

    Returns
    -------
    results: a DataFrame with the evaluation results.
    """

    if isinstance(agent_source, list):
        # Construction of the array of evaluations
        df = pd.DataFrame()
        if isinstance(agent_source[0], str) or isinstance(
            agent_source[0], pathlib.PurePath
        ):
            agent_manager_list = [ExperimentManager(None) for _ in agent_source]
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
            agent_source
        ), "Each agent must have unique name."

    return df


def compare_agents_data(
    data_to_compare,
    method="tukey_hsd",
    alpha=0.05,
    B=10_000,
    seed=None,
):
    """
    Compare several trained agents using the mean over n_simulations evaluations for each agent.

    Parameters
    ----------
    data_to_compare : DataFrame
        Data of the agents to compare (result from their evaluation). The dataframe must have 2 columns : 'mean_eval' and 'agent'
    method: str in {"tukey_hsd", "permutation"}, default="tukey_hsd"
        Method used in the test. "tukey_hsd" use scipy implementation [1] and "permutation" use permutation test with Step-Down method for multiple testing [2]. Tukey HSD method suppose Gaussian model on the aggregated evaluations and permutation test is non-parametric and does not make assumption on the distribution. permutation is the safe choice when the reward is likely to be heavy-tailed or multimodal.
    alpha: float, default = 0.05
        Level of the test, control the Family-wise error.
    B: int, default = 10_000
        Number of random permutations used to approximate the permutation test if method = "permutation"
    seed: int or None,
        The seed of the random number generator from which we sample permutations. If None, create one.

    Returns
    -------
    results: a DataFrame summarising the results.

    References
    ----------
    [1]: https://scipy.github.io/devdocs/reference/generated/scipy.stats.tukey_hsd.html

    [2]: Testing Statistical Hypotheses by E. L. Lehmann, Joseph P. Romano (Section 15.4.4), https://doi.org/10.1007/0-387-27605-X, Springer

    """
    agent_names = data_to_compare["agent"].unique()
    data = np.array(
        [
            np.array(data_to_compare.loc[data_to_compare["agent"] == name, "mean_eval"])
            for name in agent_names
        ]
    )

    n_agents = len(agent_names)
    vs = [
        (agent_names[i], agent_names[j])
        for i in range(n_agents)
        for j in range(n_agents)
        if i < j
    ]

    mean_agent1 = [
        np.mean(data_to_compare.loc[data_to_compare["agent"] == vs[i][0], "mean_eval"])
        for i in range(len(vs))
    ]

    mean_agent2 = [
        np.mean(data_to_compare.loc[data_to_compare["agent"] == vs[i][1], "mean_eval"])
        for i in range(len(vs))
    ]
    mean_diff = [
        np.mean(
            np.array(
                data_to_compare.loc[data_to_compare["agent"] == vs[i][0], "mean_eval"]
            )
            - np.array(
                data_to_compare.loc[data_to_compare["agent"] == vs[i][1], "mean_eval"]
            )
        )
        for i in range(len(vs))
    ]
    std_diff = [
        np.std(
            np.array(
                data_to_compare.loc[data_to_compare["agent"] == vs[i][0], "mean_eval"]
            )
            - np.array(
                data_to_compare.loc[data_to_compare["agent"] == vs[i][1], "mean_eval"]
            )
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
        results_perm = _permutation_test(data, B, alpha, seed) == 1
        decisions = [
            "accept" if results_perm[i][j] else "reject"
            for i in range(n_agents)
            for j in range(n_agents)
            if i < j
        ]
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


def _permutation_test(data, B, alpha, seed):
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
    seeder = Seeder(seed)

    logger.info("Beginning permutationt test")
    while True:
        current_comparisons = comparisons[comparisons_alive]
        logger.info(f"Still {len(current_comparisons)} comparisons to test")

        # make a generator of permutations
        if B is None:
            permutations = combinations(2 * n_fit, n_fit)
        else:
            permutations = (seeder.rng.permutation(2 * n_fit) for _ in range(B))

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
