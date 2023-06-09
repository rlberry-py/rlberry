# Interfaces
from .agent import Agent
from .agent import AgentWithSimplePolicy
from .agent import AgentTorch

# Basic agents (in alphabetical order)
# basic = does not require torch, jax, etc...
from .adaptiveql import AdaptiveQLAgent
from .dynprog import ValueIterationAgent
from .kernel_based import RSUCBVIAgent, RSKernelUCBVIAgent
from .linear import LSVIUCBAgent
from .mbqvi import MBQVIAgent
from .optql import OptQLAgent
from .psrl import PSRLAgent
from .rlsvi import RLSVIAgent
from .ucbvi import UCBVIAgent
