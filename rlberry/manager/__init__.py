from .experiment_manager import ExperimentManager, preset_manager
from .multiple_managers import MultipleManagers
from .evaluation import evaluate_agents, plot_writer_data, read_writer_data

# AgentManager alias for the ExperimentManager class, for backward compatibility
AgentManager = ExperimentManager
