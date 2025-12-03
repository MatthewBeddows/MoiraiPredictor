"""
Modular Agent System for Agricultural Forecasting

Each agent is a self-contained module that performs a specific task.
The Agent Manager coordinates which agents to use based on data characteristics.
"""

from .diversity_agent import DiversityAgent
from .feature_scoring_agent import FeatureScoringAgent
from .aggregation_agent import AggregationAgent
from .selection_agent import SelectionAgent
from .agent_manager import AgentManager

# AI-Powered Agents
from .base_agent import BaseAgent, Critique, Memory, Tool
from .weak_plot_agent import WeakPlotIdentifierAgent
from .meta_learning_agent import MetaLearningAgent
from .ensemble_agent import EnsembleStrategyAgent
from .data_explorer_agent import DataExplorerAgent
from .prediction_correction_agent import PredictionCorrectionAgent

__all__ = [
    # Deterministic Agents
    'DiversityAgent',
    'FeatureScoringAgent',
    'AggregationAgent',
    'SelectionAgent',
    'AgentManager',
    # AI-Powered Agents
    'BaseAgent',
    'Critique',
    'Memory',
    'Tool',
    'WeakPlotIdentifierAgent',
    'MetaLearningAgent',
    'EnsembleStrategyAgent',
    'DataExplorerAgent',
    'PredictionCorrectionAgent'
]
