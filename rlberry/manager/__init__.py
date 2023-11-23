from .experiment_manager import ExperimentManager
from .experiment_manager import preset_manager
from .multiple_managers import MultipleManagers
from .evaluation import evaluate_agents, plot_writer_data, read_writer_data
from .comparison import compare_agents
from .env_tools import with_venv, run_venv_xp, with_guix, run_guix_xp

# (Remote)AgentManager alias for the (Remote)ExperimentManager class, for backward compatibility
AgentManager = ExperimentManager
