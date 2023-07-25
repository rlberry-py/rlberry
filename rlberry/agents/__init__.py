# Interfaces
# Basic agents (in alphabetical order)
# basic = does not require torch, jax, etc...
from .adaptiveql import AdaptiveQLAgent
from .agent import Agent, AgentTorch, AgentWithSimplePolicy
from .dynprog import ValueIterationAgent
from .kernel_based import RSKernelUCBVIAgent, RSUCBVIAgent
from .linear import LSVIUCBAgent
from .mbqvi import MBQVIAgent
from .optql import OptQLAgent
from .psrl import PSRLAgent
from .rlsvi import RLSVIAgent
from .tabular_rl import QLAgent, SARSAAgent
from .ucbvi import UCBVIAgent
