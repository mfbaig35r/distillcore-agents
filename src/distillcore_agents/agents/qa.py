"""QA agent — validates processing quality and recommends adjustments."""

from __future__ import annotations

import json

from pydantic_ai import Agent, RunContext

from ..client import DistillcoreClient
from .models import QADecision

_SYSTEM_PROMPT = """\
You are a quality assurance agent for document processing. Your job is to \
validate the processing result and recommend adjustments if quality is below \
thresholds.

Rules:
1. Call check_coverage to get the validation metrics.
2. Call check_chunks to inspect chunk quality (empty chunks, missing topics).
3. Coverage thresholds:
   - structuring_coverage >= 0.95 (PASS)
   - chunking_coverage >= 0.98 (PASS)
   - end_to_end_coverage >= 0.93 (PASS)
4. If any threshold fails, set verified=False and provide recommendations:
   - Low structuring coverage -> recommend changing preset or reducing \
llm_page_window_size with more overlap.
   - Low chunking coverage -> recommend reducing target_tokens (smaller chunks \
capture more text).
   - Low end-to-end coverage -> recommend reprocessing with adjusted params.
   - Empty chunks -> recommend setting min_tokens=50 to merge tiny chunks, \
and/or reducing target_tokens.
   - Missing topics -> recommend enabling enrichment.
5. Be specific in recommendations — name the parameter and suggested value."""


async def _check_coverage(
    ctx: RunContext[DistillcoreClient],
    source_text: str,
    chunk_texts_json: str,
) -> str:
    """Check how much of the original text is preserved in the chunks.

    Args:
        source_text: The original document full text.
        chunk_texts_json: JSON array of chunk text strings.
    """
    chunk_texts = json.loads(chunk_texts_json)
    combined = " ".join(chunk_texts)
    coverage = ctx.deps.compute_coverage(source_text, combined)
    return f"End-to-end coverage: {coverage:.4f}"


async def _check_chunks(
    ctx: RunContext[DistillcoreClient],
    chunks_json: str,
) -> str:
    """Inspect chunk quality — count empty chunks and chunks missing topics.

    Args:
        chunks_json: JSON array of chunk objects with 'text', 'topic',
                     'token_estimate', and 'key_concepts' fields.
    """
    chunks = json.loads(chunks_json)
    empty = sum(1 for c in chunks if len(c.get("text", "").strip()) < 10)
    missing_topic = sum(1 for c in chunks if not c.get("topic"))
    missing_concepts = sum(1 for c in chunks if not c.get("key_concepts"))
    total = len(chunks)
    lines = [
        f"Total chunks: {total}",
        f"Empty chunks (<10 chars): {empty}",
        f"Chunks missing topics: {missing_topic}",
        f"Chunks missing key_concepts: {missing_concepts}",
    ]
    return "\n".join(lines)


def create_qa_agent(
    model: str = "openai:gpt-4o-mini",
) -> Agent[DistillcoreClient, QADecision]:
    """Create a QA agent that validates processing quality."""
    return Agent(
        model,
        deps_type=DistillcoreClient,
        output_type=QADecision,
        system_prompt=_SYSTEM_PROMPT,
        tools=[_check_coverage, _check_chunks],
    )
