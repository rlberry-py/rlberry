from .experiment_manager import ExperimentManager, preset_manager
from .multiple_managers import MultipleManagers
from .evaluation import evaluate_agents, plot_writer_data, read_writer_data
from .comparison import compare_agents

# (Remote)AgentManager alias for the (Remote)ExperimentManager class, for backward compatibility
AgentManager = ExperimentManager
