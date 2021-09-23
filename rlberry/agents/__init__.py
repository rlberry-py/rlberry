# Import interfaces
from .agent import Agent
from .agent import AgentWithSimplePolicy

# basic agents (alphabetical!)
# basic = does not require torch, jax etc
from .adaptiveql import *
from .dynprog import *
from .kernel_based import *
from .mbqvi import *
from .optql import *
from .ucbvi import *
