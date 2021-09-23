import logging.config
from pathlib import Path
import gym


def configure_logging(level: str = "INFO",
                      file_path: Path = None,
                      file_level: str = "DEBUG",
                      default_msg: str = "") -> None:
    """
    Set the logging configuration

    This default config can be further edited to only enable logging
    in specific modules, by providing the name of its logger.

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
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": default_msg + "[%(levelname)s] %(message)s "
            },
            "detailed": {
                "format": default_msg + "[%(name)s:%(levelname)s] %(message)s "
            }
        },
        "handlers": {
            "default": {
                "level": level,
                "formatter": "standard",
                "class": "logging.StreamHandler"
            }
        },
        "loggers": {
            "": {
                "handlers": [
                    "default"
                ],
                "level": "DEBUG",
                "propagate": True
            }
        }
    }
    if file_path:
        config["handlers"][file_path.name] = {
            "class": "logging.FileHandler",
            "filename": file_path,
            "level": file_level,
            "formatter": "detailed",
            "mode": 'w'
        }
        config["loggers"][""]["handlers"].append(file_path.name)
    logging.config.dictConfig(config)
    gym.logger.set_level(logging.getLevelName(level))
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
