"""Structured output types for each pipeline agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TriageDecision(BaseModel):
    """Output from the triage agent — configures the processing pipeline."""

    source_filename: str
    page_count: int = 0
    detected_format: str = "unknown"
    preset: str = "generic"
    needs_ocr: bool = False
    target_tokens: int = 500
    overlap_chars: int = 200
    chunk_strategy: str = "auto"
    min_tokens: int = 0
    enable_enrichment: bool = True
    llm_page_window_size: int = 15
    reasoning: str = ""


class ProcessingDecision(BaseModel):
    """Output from the processing agent — summarizes pipeline execution."""

    source_filename: str
    document_type: str = "unknown"
    document_title: str | None = None
    page_count: int = 0
    section_count: int = 0
    chunk_count: int = 0
    structuring_coverage: float = 0.0
    chunking_coverage: float = 0.0
    end_to_end_coverage: float = 0.0
    validation_passed: bool = False
    warnings: list[str] = Field(default_factory=list)
    document_id: str | None = None
    reasoning: str = ""


class QARecommendation(BaseModel):
    """A single recommendation from the QA agent."""

    issue: str
    action: str
    parameter: str | None = None
    suggested_value: str | None = None


class QADecision(BaseModel):
    """Output from the QA agent — validates processing quality."""

    verified: bool = False
    structuring_coverage: float = 0.0
    chunking_coverage: float = 0.0
    end_to_end_coverage: float = 0.0
    empty_chunk_count: int = 0
    chunks_missing_topics: int = 0
    recommendations: list[QARecommendation] = Field(default_factory=list)
    reasoning: str = ""


class Citation(BaseModel):
    """A citation from a search result."""

    document_id: str
    source_filename: str
    chunk_index: int
    text_snippet: str
    score: float


class ResearchResult(BaseModel):
    """Output from the research agent — answers questions using stored documents."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    documents_searched: int = 0
    reasoning: str = ""
