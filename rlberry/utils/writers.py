import logging
from typing import Optional
from timeit import default_timer as timer


logger = logging.getLogger(__name__)


class DefaultWriter:
    """
    Default writer to be used by the agents.

    Can be used in the fit() method of the agents, so
    that training data can be visualized later.

    Parameters
    ----------
    name : str
        Name of the writer.
    log_interval : int
        Minimum number of seconds between consecutive logs.
    """
    def __init__(self, name: str, log_interval: int = 3):
        self._name = name
        self._log_interval = log_interval
        self._data = None
        self._data_global_steps = None
        self._time_last_log = None
        self.reset()

    def reset(self):
        """Clear all data."""
        self._data = dict()
        self._data_global_steps = dict()
        self._time_last_log = timer()

    @property
    def data(self):
        return self._data

    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None):
        """
        Store scalar value.

        Parameters
        ----------
        tag : str
            Tag for the scalar.
        scalar_value : float
            Value of the scalar.
        global_step : int
            Step where scalar was added. If None, global steps will no longer be stored for the current tag.
        """
        # Update data structures
        if tag not in self._data:
            self._data[tag] = []
            self._data_global_steps[tag] = []

        self._data[tag].append(scalar_value)
        if (self._data_global_steps[tag] is not None) and (global_step is not None):
            self._data_global_steps[tag].append(global_step)
        if global_step is None:
            self._data_global_steps[tag] = None

        # Log
        self._log()

    def _log(self):
        # time since last log
        t_now = timer()
        time_elapsed = t_now - self._time_last_log
        # log if enough time has passed since the last log
        max_global_step = 0
        if time_elapsed > self._log_interval:
            self._time_last_log = t_now
            message = ''
            for tag in self._data:
                message += f'{tag} = {self._data[tag][-1]} | '
                if self._data_global_steps[tag] is not None:
                    max_global_step = max(max_global_step, self._data_global_steps[tag][-1])

            message = f'[{self._name}] | step = {max_global_step} | ' + message
            logger.info(message)

    def __getattr__(self, attr):
        """
        Avoid raising exceptions when invalid method is called, so
        that DefaultWriter does not raise exceptions when
        the code expects a tensorboard writer.
        """
        if attr[:2] == '__':
            raise AttributeError(attr)

        def method(*args, **kwargs):
            pass
        return method
