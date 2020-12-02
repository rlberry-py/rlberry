import time
import logging

logger = logging.getLogger(__name__)


class PeriodicWriter:
    """
    Writes info to stdout periodically.

    Useful for simple experiments where tensorboard is too much.

    Parameters
    ----------
    name : str
        Name of the writer, printed in the logs.
    log_every : int
        Interval used to log data. Measured in the number of calls
        to add_scalar()
    """
    def __init__(self, name, log_every=10):
        self.name = name
        self.log_every = log_every
        #
        self.data = None
        self._calls_since_last_log = None
        self._time_last_log = None
        self.reset()

    def reset(self):
        self.data = {}
        self._calls_since_last_log = 0
        self._time_last_log = time.process_time()

    def add_scalar(self, tag, scalar_value, global_step=None):
        self._calls_since_last_log += 1
        if tag not in self.data:
            self.data[tag] = []
        self.data[tag].append((scalar_value, global_step))

        if self.log_every > 0 and \
                self._calls_since_last_log % self.log_every == 0:
            self.log()
            self._calls_since_last_log = 0

    def log(self):
        # time since last log
        t_now = time.process_time()
        time_elapsed = t_now - self._time_last_log
        self._time_last_log = t_now

        # write message
        message = "[{}]".format(self.name)

        for tag in self.data:
            last_value, last_step = self.data[tag][-1]
            if last_step is not None:
                message += \
                    " | step ={}, {} = {:0.3f}".format(last_step,
                                                       tag,
                                                       last_value)
            else:
                message += \
                    " | {} = {:0.3f}".format(tag,
                                             last_value)
        # append time per log
        if time_elapsed > 0:
            logs_per_ms = self.log_every / time_elapsed
            message += " | freq = {:0.3f} logs/ms".format(logs_per_ms)
        else:
            message += " | freq = N.A. logs/ms"

        logger.info(message)

    def __getattr__(self, attr):
        """
        Avoid raising exceptions when invalid method is called, so
        that PeriodicWriter does not raise exceptions when
        the code expects a tensorboard writer.
        """
        if attr[:2] == '__':
            raise AttributeError(attr)

        def method(*args, **kwargs):
            pass
        return method
