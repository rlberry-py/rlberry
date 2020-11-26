from abc import ABC, abstractmethod


class FeatureMap(ABC):
    """
    Class representing a feature map, from (observation, action) pairs
    to numpy arrays.

    Attributes
    ----------
    shape : tuple
        Shape of feature array.

    Methods
    --------
    map()
        Maps a (observation, action) pair to a numpy array.
    """
    def __init__(self):
        ABC.__init__(self)
        self.shape = ()

    @abstractmethod
    def map(self, observation, action):
        """
        Maps a (observation, action) pair to a numpy array.
        """
        pass
