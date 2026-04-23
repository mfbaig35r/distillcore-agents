"""Exception hierarchy for distillcore-agents."""

from __future__ import annotations


class DistillcoreAgentError(Exception):
    """Base exception for all agent errors."""

    def __init__(self, message: str = "", agent: str = "") -> None:
        self.agent = agent
        self.message = message
        super().__init__(f"[{agent}] {message}" if agent else message)


class TriageError(DistillcoreAgentError):
    """Triage agent failed to assess the document."""


class ProcessingError(DistillcoreAgentError):
    """Processing agent failed to run the pipeline."""


class QAError(DistillcoreAgentError):
    """QA agent failed to validate the result."""


class ResearchError(DistillcoreAgentError):
    """Research agent failed to search/synthesize."""
