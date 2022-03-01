from rlberry.utils.logging import configure_logging
import dunamai as _dunamai

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# Initialize logging level

configure_logging(level="INFO")

# define __version__

__version__ = _dunamai.get_version(
    "your-library", third_choice=_dunamai.Version.from_any_vcs
).serialize()
