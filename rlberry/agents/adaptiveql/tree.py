import gym.spaces as spaces
import numpy as np
import matplotlib.pyplot as plt
from rlberry.agents.adaptiveql.utils import bounds_contains, split_bounds


class TreeNode:
    """
    Node representing an l-infinity ball in R^d, that points
    to sub-balls (node children).
    Stores a value, a number of visits, and (possibly) rewards and transition probability
    to a list of other nodes.

    This class is used to represent (and store data about)
    a tuple (state, action, stage) = (x, a, h).

    Parameters
    ----------
    bounds : numpy.ndarray
        Bounds of each dimension [ [x0, y0], [x1, y1], ..., [xd, yd] ],
        representing the cartesian product in R^d:
        [x0, y0] X [x1, y1] X ... X [xd, yd]
    depth: int
        Node depth, root is at depth 0.
    qvalue : double, default: 0
        Initial node Q value
    n_visits : int, default = 0
        Number of visits to the node.

    """

    def __init__(self, bounds, depth, qvalue=0.0, n_visits=0):
        self.dim = len(bounds)

        self.radius = (bounds[:, 1] - bounds[:, 0]).max() / 2.0
        assert self.radius > 0.0

        self.bounds = bounds
        self.depth = depth
        self.qvalue = qvalue
        self.n_visits = n_visits
        self.children = []

        #
        # For AdaMB
        #

        # Value V, initialized as Q
        self.vvalue = qvalue
        # Reward estimate
        self.reward_est = 0.0
        # Dictionary node_id -> transition_prob
        # node_id = id(node), where id() is a built-in python function
        self.transition_probs = {}
        # Dictionary node_id -> node
        self.transition_nodes = {}

    def is_leaf(self):
        return len(self.children) == 0

    def contains(self, x):
        """Check if `x` is contained in the node/ball."""
        return bounds_contains(self.bounds, x)

    def split(self):
        """Spawn children nodes by splitting the ball."""
        child_bounds = split_bounds(self.bounds)
        for bounds in child_bounds:
            self.children.append(
                TreeNode(bounds, self.depth + 1, self.qvalue, self.n_visits)
            )


class TreePartition:
    """
    Tree-based partition of an l-infinity ball in R^d.

    Each node is of type TreeNode.

    Parameters
    ----------
    space: gym.spaces.Box
        Domain of the function.
    initial_value: double
        Value to initialize the root node.
    """

    def __init__(self, space, initial_value=0.0):
        assert isinstance(space, spaces.Box)
        assert space.is_bounded()

        bounds = np.vstack((space.low, space.high)).T
        self.root = TreeNode(bounds, depth=0, qvalue=initial_value)
        self.dim = bounds.shape[0]
        self.dmax = self.root.radius

    def traverse(self, x, update=False):
        """
        Returns leaf node containing x.

        If `update=true`, increments number of visits of each
        node in the path.

        Parameters
        ----------
        x : numpy.ndarray
            Array of shape (d,)
        """
        node = self.root

        # traverse the tree until leaf
        while True:
            if update:
                node.n_visits += 1
            if node.is_leaf():
                break
            for cc in node.children:
                if cc.contains(x):
                    node = cc
                    break

        # return value at leaf
        return node

    def plot(self,
             fignum="tree plot",
             colormap_name='cool',
             max_value=10,
             node=None,
             root=True, ):
        """
        Visualize the function (2d domain only).
        Shows the hierarchical partition.
        """
        if root:
            assert self.dim == 2, "TreePartition plot only available for 2-dimensional spaces."
            node = self.root
            plt.figure(fignum)

        # draw region corresponding to the leaf
        if node.is_leaf():
            x0, x1 = node.bounds[0, :]
            y0, y1 = node.bounds[1, :]

            colormap_fn = plt.get_cmap(colormap_name)
            color = colormap_fn(node.qvalue / max_value)
            rectangle = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, ec="black", color=color)
            plt.gca().add_patch(rectangle)
            plt.axis('scaled')

        else:
            for cc in node.children:
                self.plot(max_value=max_value, colormap_name=colormap_name, node=cc, root=False)


class MDPTreePartition:
    """
    Set of H x A TreePartition instances.

    Used to store/manipulate a Q function, a reward function and a transition model.
    """

    def __init__(self, observation_space, action_space, horizon):
        self.horizon = horizon
        self.n_actions = action_space.n
        self.trees = []
        for hh in range(horizon):
            self.trees.append({})
            for aa in range(self.n_actions):
                self.trees[hh][aa] = TreePartition(observation_space,
                                                   initial_value=horizon - hh)

        self.dmax = self.trees[0][0].dmax

    def get_argmax_and_node(self, x, hh):
        """
        Returns a* = argmax_a Q_h(x, a) and the node corresponding to (x, a*).
        """
        # trees for each action at hh
        trees_hh = self.trees[hh]

        best_action = 0
        best_node = trees_hh[0].traverse(x, update=False)
        best_val = best_node.qvalue
        for aa in range(1, self.n_actions):
            node = trees_hh[aa].traverse(x, update=False)
            val = node.qvalue
            if val > best_val:
                best_val = val
                best_action = aa
                best_node = node

        return best_action, best_node

    def update_counts(self, x, aa, hh):
        """
        Increment counters associated to (x, aa, hh) and returns the node.
        """
        tree = self.trees[hh][aa]
        node = tree.traverse(x, update=True)
        return node

    def plot(self, a, h):
        """
        Visualize Q_h(x, a)
        """
        self.trees[h][a].plot(max_value=self.horizon - h)
