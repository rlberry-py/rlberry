__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# Initialize seeding
from rlberry.seeding import seeding
seeding.set_global_seed()

# gym wrapper
# Importing here avoids circular dependencies,
# rlberry.wrappers depend on rlberry.envs,
# and envs.gym_make depends on rlberry.wrappers
import rlberry.envs.gym_make

# Initialize logging level
from rlberry.utils.logging import configure_logging
configure_logging(level="INFO")
