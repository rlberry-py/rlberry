import numpy as np
import pandas as pd
from collections import deque
from typing import Optional
from timeit import default_timer as timer
from rlberry import check_packages
from rlberry import metadata_utils

if check_packages.TENSORBOARD_INSTALLED:
    from torch.utils.tensorboard import SummaryWriter

import rlberry

logger = rlberry.logger


class DefaultWriter:
    """
    Default writer to be used by the agents, optionally wraps an instance of tensorboard.SummaryWriter.

    Can be used in the fit() method of the agents, so
    that training data can be handled by AgentManager and RemoteAgentManager.

    Parameters
    ----------
    name : str
        Name of the writer.
    print_log : bool, default=True
        If True, print logs to stderr.
    log_interval : int
        Minimum number of seconds between consecutive logs (with logging module).
    tensorboard_kwargs : Optional[dict]
        Parameters for tensorboard SummaryWriter. If provided, DefaultWriter
        will behave as tensorboard.SummaryWriter, and will keep utilities to handle
        data added with the add_scalar method.
    execution_metadata : metadata_utils.ExecutionMetadata
        Execution metadata about the object that is using the writer.
    maxlen : Optional[int], default: None
        If given, data stored by self._data (accessed through the property self.data) is limited
        to `maxlen` entries.
    maxlen_by_tag: Optional[dict], default: {}
        If given, applies the maxlen logic tag by tag, using the above maxlen as default.
    """

    def __init__(
        self,
        name: str,
        print_log: bool = True,
        log_interval: int = 3,
        tensorboard_kwargs: Optional[dict] = None,
        execution_metadata: Optional[metadata_utils.ExecutionMetadata] = None,
        maxlen: Optional[int] = None,
        maxlen_by_tag: Optional[dict] = {},
    ):
        self._name = name
        self._print_log = print_log
        self._log_interval = log_interval
        self._execution_metadata = execution_metadata
        self._data = None
        self._time_last_log = None
        self._log_time = True
        self._maxlen = maxlen
        self._maxlen_by_tag = maxlen_by_tag
        self.reset()

        # initialize tensorboard
        if (tensorboard_kwargs is not None) and (
            not check_packages.TENSORBOARD_INSTALLED
        ):
            logger.warning(
                "[DefaultWriter]: received tensorboard_kwargs, but tensorboard is not installed."
            )
        self._tensorboard_kwargs = tensorboard_kwargs
        self._tensorboard_logdir = None
        self._summary_writer = None
        if (tensorboard_kwargs is not None) and check_packages.TENSORBOARD_INSTALLED:
            self._summary_writer = SummaryWriter(**self._tensorboard_kwargs)
            self._tensorboard_logdir = self._summary_writer.get_logdir()

    def reset(self):
        """Clear data."""
        self._data = dict()
        self._initial_time = timer()
        self._time_last_log = timer()

    @property
    def summary_writer(self):
        return self._summary_writer

    @property
    def data(self):
        df = pd.DataFrame(columns=("name", "tag", "value", "global_step"))
        for tag in self._data:
            df = pd.concat([df, pd.DataFrame(self._data[tag])], ignore_index=True)
        return df

    def read_tag_value(self, tag, main_tag: str = ""):
        """
        Reads the values for the tag `tag`.
        If a `main_tag` is given, the tag will be a concatenation of
        `main_tag`, underscore and `tag`.

        Parameters
        ----------
        tag: string
            Tag to be searched
        main_tag: string, default=""
            Main tag. If `main_tag == ""`  then use only `tag`.

        Returns
        -------
        The writer values for the tag, a pandas Series.
        """
        full_tag = str(main_tag) + "_" + str(tag) if str(main_tag) else str(tag)
        return self._data[full_tag]["value"]

    def read_first_tag_value(self, tag, main_tag: str = ""):
        """
        Reads the first value for the tag `tag`.
        If a `main_tag` is given, the tag will be a concatenation of `main_tag`, underscore and `tag`.

        Parameters
        ----------
        tag: string
            Tag to be searched
        main_tag: string, default=""
            Main tag. If `main_tag == ""`  then use only `tag`.

        Returns
        -------
        The first value encountered with tag `tag`.
        """
        full_tag = str(main_tag) + "_" + str(tag) if str(main_tag) else str(tag)
        return self._data[full_tag]["value"][0]

    def read_last_tag_value(self, tag, main_tag: str = ""):
        """
        Reads the last value for the tag `tag`.
        If a `main_tag` is given, the tag will be a concatenation of `main_tag`, underscore and `tag`.

        Parameters
        ----------
        tag: string
            Tag to be searched
        main_tag: string, default=""
            Main tag. If `main_tag == ""`  then use only `tag`.

        Returns
        -------
        The last value with tag `tag` encountered by the writer.
        """
        full_tag = str(main_tag) + "_" + str(tag) if str(main_tag) else str(tag)
        return self._data[full_tag]["value"][-1]

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: Optional[int] = None,
        walltime=None,
        new_style=False,
    ):
        """
        Behaves as SummaryWriter.add_scalar().

        WARNING: 'global_step' can be confusing when a scalar is written at each episode
        and another is written at each iteration. The global step 1 may be associated to
        first episode and first step in environment.

        Parameters
        ----------
        tag : str
            Tag for the scalar.
        scalar_value : float
            Value of the scalar.
        global_step : int
            Step where scalar was added. If None, global steps will no longer be stored
            for the current tag.
        walltime : float
            Optional override default walltime (time.time()) with seconds after epoch of event
        new_style : bool
            Whether to use new style (tensor field) or old
            style (simple_value field). New style could lead to faster data loading.
        """
        if self._summary_writer:
            self._summary_writer.add_scalar(
                tag, scalar_value, global_step, walltime, new_style
            )
        self._add_scalar(tag, scalar_value, global_step)

    def _add_scalar(
        self, tag: str, scalar_value: float, global_step: Optional[int] = None
    ):
        """
        Store scalar value in self._data.
        """
        # Update data structures
        if tag not in self._data:
            self._data[tag] = dict()
            self._data[tag]["name"] = deque(
                maxlen=self._maxlen_by_tag.get(tag, self._maxlen)
            )
            self._data[tag]["tag"] = deque(
                maxlen=self._maxlen_by_tag.get(tag, self._maxlen)
            )
            self._data[tag]["value"] = deque(
                maxlen=self._maxlen_by_tag.get(tag, self._maxlen)
            )
            self._data[tag]["dw_time_elapsed"] = deque(
                maxlen=self._maxlen_by_tag.get(tag, self._maxlen)
            )
            self._data[tag]["global_step"] = deque(
                maxlen=self._maxlen_by_tag.get(tag, self._maxlen)
            )

        self._data[tag]["name"].append(
            self._name
        )  # used in plots, when aggregating several writers
        self._data[tag]["tag"].append(
            tag
        )  # useful to convert all data to a single DataFrame
        self._data[tag]["value"].append(scalar_value)
        self._data[tag]["dw_time_elapsed"].append(timer() - self._initial_time)
        if global_step is None:
            self._data[tag]["global_step"].append(np.nan)
        else:
            self._data[tag]["global_step"].append(global_step)

        # change _log_time
        if global_step is not None and self._log_time:
            self._log_time = False

        # Log
        if (not self._log_time) and (self._print_log):
            self._log()

    def add_scalars(
        self,
        main_tag: str = "",
        tag_scalar_dict: dict = dict(),
        global_step: Optional[int] = None,
        walltime=None,
    ):
        """
        Behaves as add_scalar, but for a list instead of a single scalar value.

        WARNING: 'global_step' can be confusing when a scalar is written at each episode
        and another is written at each iteration. The global step 1 may be associated to
        first episode and first step in environment.

        Parameters
        ----------
        main_tag : string
            The parent name for the tags.
        tag_scalar_dict : dict
            Key-value pair storing the tag and corresponding values.
        global_step : int
            Step where scalar was added. If None, global steps will no longer be stored for the current tag.
        walltime : float
            Optional override default walltime (time.time()) with seconds after epoch of event
        """
        if self._summary_writer:
            self._summary_writer.add_scalars(
                main_tag, tag_scalar_dict, global_step, walltime
            )
        self._add_scalars(main_tag, tag_scalar_dict, global_step)

    def _add_scalars(
        self, main_tag: str, tag_scalar_dict: dict, global_step: Optional[int] = None
    ):
        """
        Store scalar values in self._data.
        """

        for tag, scalar_value in tag_scalar_dict.items():
            full_tag = str(main_tag) + "_" + str(tag) if str(main_tag) else str(tag)
            self._add_scalar(full_tag, scalar_value, global_step)

    def _log(self):
        # time since last log
        t_now = timer()
        time_elapsed = t_now - self._time_last_log
        # log if enough time has passed since the last log
        max_global_step = 0
        if time_elapsed > self._log_interval:
            self._time_last_log = t_now
            message = ""
            for tag in self._data:
                val = self._data[tag]["value"][-1]
                gstep = self._data[tag]["global_step"][-1]
                message += f"{tag} = {val} | "
                if not np.isnan(gstep):
                    max_global_step = max(max_global_step, gstep)

            header = self._name
            if self._execution_metadata:
                header += f"[worker: {self._execution_metadata.obj_worker_id}]"
            message = f"[{header}] | max_global_step = {max_global_step} | " + message
            logger.info(message)

    def __getattr__(self, attr):
        """
        Calls SummaryWriter methods, if self._summary_writer is not None.
        Otherwise, does nothing.
        """
        if attr[:2] == "__":
            raise AttributeError(attr)
        if attr in self.__dict__:
            return getattr(self, attr)
        if self._summary_writer:
            return getattr(self._summary_writer, attr)

        def method(*args, **kwargs):
            pass

        return method

    #
    # For pickle
    #
    def __getstate__(self):
        if self._summary_writer:
            self._summary_writer.close()
        state = self.__dict__.copy()
        return state

    def __setstate__(self, newstate):
        # Re-create summary writer with the same logdir
        if newstate["_summary_writer"]:
            newstate["_tensorboard_kwargs"].update(
                dict(log_dir=newstate["_tensorboard_logdir"])
            )
            newstate["_summary_writer"] = SummaryWriter(
                **newstate["_tensorboard_kwargs"]
            )
        self.__dict__.update(newstate)
