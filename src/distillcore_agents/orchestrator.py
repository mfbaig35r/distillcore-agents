"""Agent orchestrator for distillcore document processing runs.

Usage::

    async with Orchestrator(openai_api_key="sk-...") as orc:
        result = await orc.process_one("/path/to/doc.pdf")
        batch = await orc.process_batch(["/path/to/a.pdf", "/path/to/b.pdf"])
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from .agents.base import AgentEvent, iter_agent_events
from .agents.models import (
    ProcessingDecision,
    QADecision,
    ResearchResult,
    TriageDecision,
)
from .store import AgentResultStore, ensure_schema

logger = logging.getLogger(__name__)


class PipelineResult(BaseModel):
    """Complete result from the 4-agent pipeline."""

    session_id: str
    source: str
    triage: TriageDecision
    processing: ProcessingDecision
    qa: QADecision
    research: ResearchResult | None = None


class ItemFailure(BaseModel):
    """A failed item in a batch run."""

    source: str
    error: str


class BatchOutput(BaseModel):
    """Aggregate result from a batch processing run."""

    session_id: str
    batch_id: str
    results: list[PipelineResult] = Field(default_factory=list)
    failures: list[ItemFailure] = Field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0


class Orchestrator:
    """Async context manager that runs the 4-agent document processing pipeline."""

    def __init__(
        self,
        *,
        model: str = "openai:gpt-4o-mini",
        store_path: str | Path = "~/.distillcore/agents.db",
        doc_store_path: str | Path = "~/.distillcore/store.db",
        openai_api_key: str = "",
        tenant_id: str | None = None,
        max_concurrency: int = 3,
    ) -> None:
        self._model = model
        self._store_path = Path(store_path).expanduser().resolve()
        self._doc_store_path = doc_store_path
        self._openai_api_key = openai_api_key
        self._tenant_id = tenant_id
        self._max_concurrency = max_concurrency
        self._session_id = uuid4().hex[:12]

        self._client: Any = None
        self._triage_agent: Any = None
        self._processing_agent: Any = None
        self._qa_agent: Any = None
        self._research_agent: Any = None
        self._result_store: AgentResultStore | None = None

    async def __aenter__(self) -> Orchestrator:
        from .agents.processing import create_processing_agent
        from .agents.qa import create_qa_agent
        from .agents.research import create_research_agent
        from .agents.triage import create_triage_agent
        from .client import DistillcoreClient

        self._client = DistillcoreClient(
            store_path=self._doc_store_path,
            openai_api_key=self._openai_api_key,
            tenant_id=self._tenant_id,
        )
        await self._client.__aenter__()

        self._triage_agent = create_triage_agent(self._model)
        self._processing_agent = create_processing_agent(self._model)
        self._qa_agent = create_qa_agent(self._model)
        self._research_agent = create_research_agent(self._model)

        ensure_schema(self._store_path)
        self._result_store = AgentResultStore(self._store_path)

        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.__aexit__(*exc)
            self._client = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def result_store(self) -> AgentResultStore:
        if self._result_store is None:
            raise RuntimeError("Orchestrator not entered — use `async with`")
        return self._result_store

    async def process_one(
        self,
        source: str | Path,
        *,
        batch_id: str | None = None,
    ) -> PipelineResult:
        """Run the full 4-agent pipeline on a single document."""
        source_str = str(source)

        # 1. Triage
        logger.info(f"[{self._session_id}] Triage: {source_str}")
        triage_result = await self._triage_agent.run(
            f"Triage document: {source_str}", deps=self._client
        )
        triage: TriageDecision = triage_result.output

        # 2. Processing
        logger.info(f"[{self._session_id}] Processing: {source_str} (preset={triage.preset})")
        processing_prompt = (
            f"Process document: {source_str}\n"
            f"Triage config: preset={triage.preset}, "
            f"target_tokens={triage.target_tokens}, "
            f"overlap_chars={triage.overlap_chars}, "
            f"needs_ocr={triage.needs_ocr}, "
            f"enable_enrichment={triage.enable_enrichment}, "
            f"llm_page_window_size={triage.llm_page_window_size}, "
            f"strategy={triage.chunk_strategy}, "
            f"min_tokens={triage.min_tokens}"
        )
        proc_result = await self._processing_agent.run(
            processing_prompt, deps=self._client
        )
        processing: ProcessingDecision = proc_result.output

        # 3. QA
        logger.info(f"[{self._session_id}] QA: {source_str}")
        qa_prompt = (
            f"Validate processing result for: {source_str}\n"
            f"Structuring coverage: {processing.structuring_coverage}\n"
            f"Chunking coverage: {processing.chunking_coverage}\n"
            f"End-to-end coverage: {processing.end_to_end_coverage}\n"
            f"Warnings: {processing.warnings}\n"
            f"Chunk count: {processing.chunk_count}\n"
            f"Validation passed: {processing.validation_passed}"
        )
        qa_result = await self._qa_agent.run(qa_prompt, deps=self._client)
        qa: QADecision = qa_result.output

        # 4. Research (conditional)
        research: ResearchResult | None = None
        if not qa.verified:
            logger.info(
                f"[{self._session_id}] QA failed for {source_str} — "
                f"{len(qa.recommendations)} recommendations"
            )

        pipeline_result = PipelineResult(
            session_id=self._session_id,
            source=source_str,
            triage=triage,
            processing=processing,
            qa=qa,
            research=research,
        )

        # Persist
        self.result_store.save(
            pipeline_result,
            session_id=self._session_id,
            batch_id=batch_id,
        )

        return pipeline_result

    async def process_one_stream(
        self,
        source: str | Path,
    ) -> AsyncIterator[tuple[AgentEvent, PipelineResult | None]]:
        """Streaming version — yields AgentEvents as each agent works."""
        source_str = str(source)

        # Triage
        triage: TriageDecision | None = None
        async for event, output in iter_agent_events(
            self._triage_agent, f"Triage document: {source_str}", deps=self._client
        ):
            yield AgentEvent(
                event_type=event.event_type,
                data={"agent": "triage", **event.data},
            ), None
            if output is not None:
                triage = output

        if triage is None:
            yield AgentEvent(
                event_type="error", data={"error": "Triage failed"}
            ), None
            return

        # Processing
        processing_prompt = (
            f"Process document: {source_str}\n"
            f"Triage config: preset={triage.preset}, "
            f"target_tokens={triage.target_tokens}, "
            f"strategy={triage.chunk_strategy}, "
            f"min_tokens={triage.min_tokens}"
        )
        processing: ProcessingDecision | None = None
        async for event, output in iter_agent_events(
            self._processing_agent, processing_prompt, deps=self._client
        ):
            yield AgentEvent(
                event_type=event.event_type,
                data={"agent": "processing", **event.data},
            ), None
            if output is not None:
                processing = output

        if processing is None:
            yield AgentEvent(
                event_type="error", data={"error": "Processing failed"}
            ), None
            return

        # QA
        qa_prompt = (
            f"Validate processing result for: {source_str}\n"
            f"Coverage: {processing.structuring_coverage:.3f} / "
            f"{processing.chunking_coverage:.3f} / {processing.end_to_end_coverage:.3f}"
        )
        qa: QADecision | None = None
        async for event, output in iter_agent_events(
            self._qa_agent, qa_prompt, deps=self._client
        ):
            yield AgentEvent(
                event_type=event.event_type,
                data={"agent": "qa", **event.data},
            ), None
            if output is not None:
                qa = output

        if qa is None:
            yield AgentEvent(
                event_type="error", data={"error": "QA failed"}
            ), None
            return

        pipeline_result = PipelineResult(
            session_id=self._session_id,
            source=source_str,
            triage=triage,
            processing=processing,
            qa=qa,
        )

        yield AgentEvent(
            event_type="completed",
            data={"result": pipeline_result.model_dump()},
        ), pipeline_result

    async def process_batch(
        self,
        sources: list[str | Path],
        *,
        batch_id: str | None = None,
        on_progress: Callable[[int, int, PipelineResult | ItemFailure], Any] | None = None,
    ) -> BatchOutput:
        """Process multiple documents concurrently with semaphore."""
        if batch_id is None:
            batch_id = uuid4().hex[:8]

        sem = asyncio.Semaphore(self._max_concurrency)
        results: list[PipelineResult] = []
        failures: list[ItemFailure] = []

        async def _run_one(src: str | Path) -> PipelineResult | ItemFailure:
            async with sem:
                try:
                    return await self.process_one(src, batch_id=batch_id)
                except Exception as exc:
                    return ItemFailure(source=str(src), error=str(exc))

        tasks = [asyncio.create_task(_run_one(s)) for s in sources]
        for coro in asyncio.as_completed(tasks):
            out = await coro
            if isinstance(out, ItemFailure):
                failures.append(out)
            else:
                results.append(out)
            if on_progress:
                on_progress(len(results) + len(failures), len(sources), out)

        return BatchOutput(
            session_id=self._session_id,
            batch_id=batch_id,
            results=results,
            failures=failures,
            total=len(sources),
            succeeded=len(results),
            failed=len(failures),
        )

    async def research(self, query: str) -> ResearchResult:
        """Run the research agent standalone for ad-hoc queries."""
        result = await self._research_agent.run(
            f"Research: {query}", deps=self._client
        )
        return result.output
