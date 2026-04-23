"""Triage agent — assesses a document and configures the processing pipeline."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext

from ..client import DistillcoreClient
from .models import TriageDecision

_SYSTEM_PROMPT = """\
You are a document triage agent for the distillcore processing pipeline. Your job \
is to examine the first page of a document and decide how to configure the \
processing pipeline for optimal results.

Rules:
1. Call preview_document to see the first page text, page count, and format.
2. Call list_available_presets to see what domain presets are available.
3. Based on the content:
   - If the document contains legal terminology (case numbers, court names, \
motions, orders, attorneys, filing dates), use the "legal" preset.
   - Otherwise, use the "generic" preset.
4. If the first page has very little text (<50 chars per page average), the \
document is likely scanned — set needs_ocr=True.
5. For documents >30 pages, increase target_tokens to 800 and \
llm_page_window_size to 20.
6. For short documents (<5 pages), decrease target_tokens to 300.
7. Always return a complete TriageDecision with your reasoning."""


async def _preview_document(
    ctx: RunContext[DistillcoreClient],
    source: str,
) -> str:
    """Extract and preview the first page of a document to assess its content.

    Returns the format, page count, and first page text for triage decisions.

    Args:
        source: Path to the document file.
    """
    extraction = ctx.deps.extract_document(source)
    first_page = extraction.pages[0].text if extraction.pages else ""
    avg_chars = (
        sum(len(p.text) for p in extraction.pages) / len(extraction.pages)
        if extraction.pages
        else 0
    )
    lines = [
        f"Format: {extraction.format}",
        f"Pages: {extraction.page_count}",
        f"Avg chars/page: {avg_chars:.0f}",
        "First page text (truncated to 2000 chars):",
        first_page[:2000],
    ]
    return "\n".join(lines)


async def _list_available_presets(
    ctx: RunContext[DistillcoreClient],
) -> str:
    """List all available domain presets for document processing.

    Returns the names of registered presets (e.g., "generic", "legal").
    """
    presets = ctx.deps.list_presets()
    return f"Available presets: {', '.join(presets)}"


def create_triage_agent(
    model: str = "openai:gpt-4o-mini",
) -> Agent[DistillcoreClient, TriageDecision]:
    """Create a triage agent that assesses documents and picks pipeline config."""
    return Agent(
        model,
        deps_type=DistillcoreClient,
        output_type=TriageDecision,
        system_prompt=_SYSTEM_PROMPT,
        tools=[_preview_document, _list_available_presets],
    )
