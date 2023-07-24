from .evaluation import evaluate_agents, plot_writer_data, read_writer_data
from .experiment_manager import ExperimentManager, preset_manager
from .multiple_managers import MultipleManagers
from .remote_experiment_manager import RemoteExperimentManager

# (Remote)AgentManager alias for the (Remote)ExperimentManager class, for backward compatibility
AgentManager = ExperimentManager
RemoteAgentManager = RemoteExperimentManager
