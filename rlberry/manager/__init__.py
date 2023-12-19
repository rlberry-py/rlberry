from .experiment_manager import ExperimentManager
from .experiment_manager import preset_manager
from .multiple_managers import MultipleManagers
from .evaluation import evaluate_agents, read_writer_data
from .comparison import compare_agents
from .plotting import plot_smoothed_curves, plot_writer_data, plot_synchronized_curves

# AgentManager alias for the ExperimentManager class, for backward compatibility
AgentManager = ExperimentManager
