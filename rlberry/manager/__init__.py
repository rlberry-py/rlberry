from .experiment_manager import ExperimentManager
from .experiment_manager import preset_manager
from .multiple_managers import MultipleManagers
from .evaluation import evaluate_agents, plot_writer_data, read_writer_data
from .comparison import compare_agents
from .plotting import plot_smoothed_curve, plot_writer_data

# AgentManager alias for the ExperimentManager class, for backward compatibility
AgentManager = ExperimentManager
