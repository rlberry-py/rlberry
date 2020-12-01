__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# Initialize seeding
from rlberry.seeding import seeding
seeding.set_global_seed()

# Initialize logging level
from rlberry.utils.logging import configure_logging
configure_logging(level="INFO")
