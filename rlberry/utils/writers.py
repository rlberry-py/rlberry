import logging
import numpy as np
import pandas as pd
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
        self._time_last_log = None
        self._log_time = True
        self.reset()

    def reset(self):
        """Clear all data."""
        self._data = dict()
        self._initial_time = timer()
        self._time_last_log = timer()

    @property
    def data(self):
        df = pd.DataFrame(columns=('name', 'tag', 'value', 'global_step'))
        for tag in self._data:
            df = df.append(pd.DataFrame(self._data[tag]), ignore_index=True)
        return df

    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None):
        """
        Store scalar value.

        Note: the tag 'dw_time_elapsed' is reserved and updated internally.
        It logs automatically the number of seconds elapsed

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
            self._data[tag] = dict()
            self._data[tag]['name'] = []
            self._data[tag]['tag'] = []
            self._data[tag]['value'] = []
            self._data[tag]['global_step'] = []

        self._data[tag]['name'].append(self._name)  # used in plots, when aggregating several writers
        self._data[tag]['tag'].append(tag)   # useful to convert all data to a single DataFrame
        self._data[tag]['value'].append(scalar_value)
        if global_step is None:
            self._data[tag]['global_step'].append(np.nan)
        else:
            self._data[tag]['global_step'].append(global_step)

        # Append time interval corresponding to global_step
        if global_step is not None and self._log_time:
            assert tag != 'dw_time_elapsed', 'The tag dw_time_elapsed is reserved.'
            self._log_time = False
            self.add_scalar(tag='dw_time_elapsed', scalar_value=timer() - self._initial_time, global_step=global_step)
            self._log_time = True

        # Log
        if not self._log_time:
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
                val = self._data[tag]['value'][-1]
                gstep = self._data[tag]['global_step'][-1]
                message += f'{tag} = {val} | '
                if not np.isnan(gstep):
                    max_global_step = max(max_global_step, gstep)

            message = f'[{self._name}] | max_global_step = {max_global_step} | ' + message
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
