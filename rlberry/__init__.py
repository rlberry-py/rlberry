from ._version import __version__
import logging

logger = logging.getLogger("rlberry_logger")

from rlberry.utils.logging import configure_logging


__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# Initialize logging level
configure_logging(level="INFO")


# define __version__

__all__ = ["__version__", "logger"]
