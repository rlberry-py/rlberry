"""
====================================================
A demo of lsviUCB algorithm in Gridworld environment
====================================================
 Illustration of how to set up an lsviUCB algorithm in rlberry and comparison
 with UCBVI and ValueIteration in GridWorld environment.

"""
import numpy as np
from rlberry.agents.features import FeatureMap
from rlberry.envs.finite import GridWorld
from rlberry.manager import AgentManager, plot_writer_data, evaluate_agents
from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.agents.linear import LSVIUCBAgent
from rlberry.agents.ucbvi import UCBVIAgent


class GridWorldFeatureMap(FeatureMap):
    def __init__(
        self, n_states, n_actions, n_rows, n_cols, index2coord, dim=15, sigma=0.25
    ):
        self.index2coord = index2coord
        self.n_states = n_states
        self.n_actions = n_actions
        self.state_repr_dim = dim
        self.sigma = sigma
        self.shape = (dim * self.n_actions,)

        # build similarity matrix
        sim_matrix = np.zeros((self.n_states, self.n_states))
        for ii in range(n_states):
            row_ii, col_ii = index2coord[ii]
            x_ii = row_ii / n_rows
            y_ii = col_ii / n_cols
            for jj in range(n_states):
                row_jj, col_jj = index2coord[jj]
                x_jj = row_jj / n_rows
                y_jj = col_jj / n_cols
                dist = np.sqrt((x_jj - x_ii) ** 2.0 + (y_jj - y_ii) ** 2.0)
                sim_matrix[ii, jj] = np.exp(-((dist / sigma) ** 2.0))

        # factorize similarity matrix to obtain features
        uu, ss, vh = np.linalg.svd(sim_matrix, hermitian=True)
        self.feats = vh[:dim, :]

    def map(self, observation, action):
        feat = np.zeros((self.state_repr_dim, self.n_actions))
        feat[:, action] = self.feats[:, observation]
        return feat.flatten()


# Function that returns an instance of a feature map
def feature_map_fn(env):
    return GridWorldFeatureMap(
        env.observation_space.n,
        env.action_space.n,
        env.nrows,
        env.ncols,
        env.index2coord,
    )


if __name__ == "__main__":
    # Parameters
    n_episodes = 750
    horizon = 10
    gamma = 0.99
    eval_kwargs = dict(eval_horizon=10)
    parallelization = "process"

    # Define environment (constructor, kwargs)
    env = (GridWorld, dict(nrows=5, ncols=5, walls=(), success_probability=0.95))

    params = dict(
        feature_map_fn=feature_map_fn,
        horizon=horizon,
        bonus_scale_factor=0.01,
        gamma=gamma,
    )

    params_ucbvi = dict(
        horizon=horizon,
        gamma=gamma,
        real_time_dp=False,
        stage_dependent=False,
        bonus_scale_factor=0.01,
    )

    params_greedy = dict(
        feature_map_fn=feature_map_fn,
        horizon=horizon,
        bonus_scale_factor=0.0,
        gamma=gamma,
    )

    params_oracle = dict(horizon=horizon, gamma=gamma)

    stats = AgentManager(
        LSVIUCBAgent,
        env,
        init_kwargs=params,
        fit_budget=n_episodes,
        eval_kwargs=eval_kwargs,
        n_fit=4,
        parallelization=parallelization,
    )

    # UCBVI baseline
    stats_ucbvi = AgentManager(
        UCBVIAgent,
        env,
        init_kwargs=params_ucbvi,
        fit_budget=n_episodes,
        eval_kwargs=eval_kwargs,
        n_fit=4,
        parallelization=parallelization,
    )

    # Random exploration baseline
    stats_random = AgentManager(
        LSVIUCBAgent,
        env,
        init_kwargs=params_greedy,
        fit_budget=n_episodes,
        eval_kwargs=eval_kwargs,
        n_fit=4,
        agent_name="LSVI (random exploration)",
        parallelization=parallelization,
    )

    # Oracle (optimal policy)
    oracle_stats = AgentManager(
        ValueIterationAgent,
        env,
        init_kwargs=params_oracle,
        fit_budget=n_episodes,
        eval_kwargs=eval_kwargs,
        n_fit=1,
    )

    # fit
    stats.fit()
    stats_ucbvi.fit()
    stats_random.fit()
    oracle_stats.fit()

    # visualize results
    plot_writer_data(
        [stats, stats_ucbvi, stats_random],
        tag="episode_rewards",
        preprocess_func=np.cumsum,
        title="Cumulative Rewards",
        show=False,
    )
    plot_writer_data(
        [stats, stats_ucbvi, stats_random], tag="dw_time_elapsed", show=False
    )
    evaluate_agents([stats, stats_ucbvi, stats_random, oracle_stats], n_simulations=20)
