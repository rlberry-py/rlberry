from .experiment_manager import ExperimentManager, preset_manager
from .multiple_managers import MultipleManagers
from .remote_experiment_manager import RemoteExperimentManager
from .evaluation import evaluate_agents, read_writer_data
from .comparison import compare_agents
from .plotting import plot_smoothed_curve, plot_writer_data

# (Remote)AgentManager alias for the (Remote)ExperimentManager class, for backward compatibility
AgentManager = ExperimentManager
RemoteAgentManager = RemoteExperimentManager
