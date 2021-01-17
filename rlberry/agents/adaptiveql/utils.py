from rlberry.utils.jit_setup import numba_jit
import gym.spaces as spaces
import numpy as np
import matplotlib.pyplot as plt


@numba_jit
def bounds_contains(bounds, x):
    """
    Returns True if `x` is contained in the bounds, and False otherwise.

    Parameters
    ----------
    bounds : numpy.ndarray
        Array of shape (d, 2).
        Bounds of each dimension [ [x0, y0], [x1, y1], ..., [xd, yd] ],
        representing the cartesian product in R^d:
        [x0, y0] X [x1, y1] X ... X [xd, yd].
    x : numpy.ndarray
        Array of shape (d,)
    """
    dim = x.shape[0]
    for dd in range(dim):
        if x[dd] < bounds[dd, 0] or x[dd] > bounds[dd, 1]:
            return False
    return True


def split_bounds(bounds, dim=0):
    """
    Split an array representing an l-infinity ball in R^d in R^d
    into a list of 2^d arrays representing the ball split.

    Parameters
    ----------
    bounds : numpy.ndarray
        Array of shape (d, 2).
        Bounds of each dimension [ [x0, y0], [x1, y1], ..., [xd, yd] ],
        representing the cartesian product in R^d:
        [x0, y0] X [x1, y1] X ... X [xd, yd].

    dim : int, default: 0
        Dimension from which to start splitting.

    Returns
    -------
    List of arrays of shape (d, 2) containing the bounds to be split.
    """
    if dim == bounds.shape[0]:
        return [bounds]
    left = bounds[dim, 0]
    right = bounds[dim, 1]
    middle = (left+right)/2.0

    left_interval = bounds.copy()
    right_interval = bounds.copy()

    left_interval[dim, 0] = left
    left_interval[dim, 1] = middle

    right_interval[dim, 0] = middle
    right_interval[dim, 1] = right

    return split_bounds(left_interval, dim+1) + split_bounds(right_interval, dim+1)


class TreeNode:
    """
    Node representing an l-infinity ball in R^d, that points
    to sub-balls (node children).

    Stores a value and a number of visits.

    Parameters
    ----------
    bounds : numpy.ndarray
        Bounds of each dimension [ [x0, y0], [x1, y1], ..., [xd, yd] ],
        representing the cartesian product in R^d:
        [x0, y0] X [x1, y1] X ... X [xd, yd]
    value : double, default: 0
        Initial node value
    n_visits : int, default = 0
        Number of visits to the node.
    """
    def __init__(self, bounds, value=0.0, n_visits=0):
        self.dim = len(bounds)

        self.radius = (bounds[:, 1] - bounds[:, 0]).max() / 2.0
        assert self.radius > 0.0

        self.bounds = bounds
        self.value = value
        self.n_visits = n_visits
        self.children = []

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
                TreeNode(bounds, self.value, self.n_visits)
            )


class FunctionTreePartition:
    """
    Partition-based representation of a function
    whose domain is an l-infinity ball in R^d.

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
        self.root = TreeNode(bounds, value=initial_value)
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
             root=True,):
        """
        Visualize the function (2d domain only).
        Shows the hierarchical partition.
        """
        if root:
            assert self.dim == 2, "FunctionTreePartition plot only available for 2-dimensional spaces."
            node = self.root
            plt.figure(fignum)

        # draw region corresponding to the leaf
        if node.is_leaf():
            x0, x1 = node.bounds[0, :]
            y0, y1 = node.bounds[1, :]

            colormap_fn = plt.get_cmap(colormap_name)
            color = colormap_fn(node.value/max_value)
            rectangle = plt.Rectangle((x0, y0), x1-x0, y1-y0, ec="black", color=color)
            plt.gca().add_patch(rectangle)
            plt.axis('scaled')

        else:
            for cc in node.children:
                self.plot(max_value=max_value, colormap_name=colormap_name, node=cc, root=False)


class QFunctionTreePartition:
    """
    Partition-based representation of Q function
    whose domain is the product of an l-infinity ball in R^d
    and a discrete set of actions.
    """
    def __init__(self, observation_space, action_space, horizon):
        self.horizon = horizon
        self.n_actions = action_space.n
        self.trees = []
        for hh in range(horizon):
            self.trees.append({})
            for aa in range(self.n_actions):
                self.trees[hh][aa] = FunctionTreePartition(observation_space,
                                                           initial_value=horizon-hh)

        self.dmax = self.trees[0][0].dmax

    def get_argmax_and_node(self, x, hh):
        """
        Returns a* = argmax_a Q_h(x, a) and the node corresponding to (x, a*).
        """
        # trees for each action at hh
        trees_hh = self.trees[hh]

        best_action = 0
        best_node = trees_hh[0].traverse(x, update=False)
        best_val = best_node.value
        for aa in range(1, self.n_actions):
            node = trees_hh[aa].traverse(x, update=False)
            val = node.value
            if val > best_val:
                best_val = val
                best_action = aa
                best_node = node

        return best_action, best_node

    def update(self, x, aa, hh):
        """
        Increment counters associated to (x, aa, hh), split the corresponding node
        if necessary, and returns the node.
        """
        tree = self.trees[hh][aa]
        node = tree.traverse(x, update=True)
        if node.n_visits >= (self.dmax/node.radius)**2.0:
            node.split()
        return node

    def plot(self, a, h):
        """
        Visualize Q_h(x, a)
        """
        self.trees[h][a].plot(max_value=self.horizon-h)
