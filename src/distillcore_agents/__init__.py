"""distillcore-agents — pydantic-ai agent layer for distillcore."""

__version__ = "0.2.0"

from .agents.models import (
    Citation,
    ProcessingDecision,
    QADecision,
    QARecommendation,
    ResearchResult,
    TriageDecision,
)
from .client import DistillcoreClient
from .errors import (
    DistillcoreAgentError,
    ProcessingError,
    QAError,
    ResearchError,
    TriageError,
)
from .orchestrator import BatchOutput, ItemFailure, Orchestrator, PipelineResult

__all__ = [
    # Orchestrator
    "Orchestrator",
    "PipelineResult",
    "BatchOutput",
    "ItemFailure",
    # Client
    "DistillcoreClient",
    # Agent output models
    "TriageDecision",
    "ProcessingDecision",
    "QADecision",
    "QARecommendation",
    "ResearchResult",
    "Citation",
    # Errors
    "DistillcoreAgentError",
    "TriageError",
    "ProcessingError",
    "QAError",
    "ResearchError",
]
