"""Agent factories and output models for distillcore-agents."""

from .base import AgentEvent, iter_agent_events
from .models import (
    Citation,
    ProcessingDecision,
    QADecision,
    QARecommendation,
    ResearchResult,
    TriageDecision,
)
from .processing import create_processing_agent
from .qa import create_qa_agent
from .research import create_research_agent
from .triage import create_triage_agent

__all__ = [
    "AgentEvent",
    "iter_agent_events",
    "TriageDecision",
    "ProcessingDecision",
    "QADecision",
    "QARecommendation",
    "ResearchResult",
    "Citation",
    "create_triage_agent",
    "create_processing_agent",
    "create_qa_agent",
    "create_research_agent",
]
