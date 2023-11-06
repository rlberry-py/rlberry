import logging
import logging.config
from pathlib import Path
from typing import Optional

import gymnasium as gym
import rlberry


def set_level(level: str = "INFO"):
    """
    Set rlberry's logger level.

    Parameters
    ----------
    level: str in {'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        Level of the logger.
    """
    rlberry.logger.setLevel(level)
    gym.logger.set_level(logging.getLevelName(level) + 10)
    for ch in rlberry.logger.handlers:
        ch.setLevel(level)


# colors
class ColoredFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt: str):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.blue + self.fmt + self.reset,
            logging.INFO: self.grey + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M")
        return formatter.format(record)


def configure_logging(
    level: str = "INFO",
    file_path: Optional[Path] = None,
    file_level: str = "DEBUG",
    default_msg: str = "",
) -> None:
    """
    Set the logging configuration

    Parameters
    ----------
    level
        Level of verbosity for the default (console) handler
    file_path
        Path to a log file
    file_level
        Level of verbosity for the file handler
    default_msg
        Message to append to the beginning all logs (e.g. thread id).
    """
    standard_msg_fmt = default_msg + "[%(levelname)s] %(asctime)s: %(message)s "
    # WARNING : if this standard message is changed, then the multi_line writer log style will bugg.
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": standard_msg_fmt},
            "detailed": {
                "format": default_msg
                + "[%(name)s:%(levelname)s] %(asctime)s: %(message)s "
            },
            "colored_standard": {"()": lambda: ColoredFormatter(standard_msg_fmt)},
        },
        "handlers": {
            "default": {
                "level": level,
                "formatter": "colored_standard",
                "class": "logging.StreamHandler",
            }
        },
        "loggers": {
            "rlberry_logger": {
                "handlers": ["default"],
                "level": level,
                "propagate": True,
            }
        },
    }
    if file_path:
        config["handlers"][file_path.name] = {
            "class": "logging.FileHandler",
            "filename": file_path,
            "level": file_level,
            "formatter": "detailed",
            "mode": "w",
        }
        config["loggers"]["rlberry_logger"]["handlers"].append(file_path.name)

    logging.config.dictConfig(config)
    gym.logger.set_level(
        logging.getLevelName(level) + 10
    )  # If info -> go to warning gym level. If debug, go to info.
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
