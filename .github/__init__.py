"""
CX-Intel Agent System
Intelligent orchestration layer for customer experience analysis
"""

from .action_intelligence import (
    ActionInsight,
    CXActionIntelligenceAgent,
    calculate_priority_score,
    generate_recommendation,
    generate_soft_cascades,
    priority_label_for_score,
)
from .px_agent import (
    PXIntelAgent,
    AgentInsight,
    PriorityIssue,
    AnalysisType,
    Severity
)

from .experiments import ExperimentPipeline, ExperimentResult
from .visualizations import AgentVisualizer, create_agent_report_with_visualizations
from .integration import render_agent_tab

__version__ = "1.0.0"
__all__ = [
    "PXIntelAgent",
    "AgentInsight",
    "PriorityIssue",
    "AnalysisType",
    "Severity",
    "ActionInsight",
    "CXActionIntelligenceAgent",
    "calculate_priority_score",
    "generate_recommendation",
    "generate_soft_cascades",
    "priority_label_for_score",
    "ExperimentPipeline",
    "ExperimentResult",
    "AgentVisualizer",
    "create_agent_report_with_visualizations",
    "render_agent_tab"
]
