"""Tests for distillcore_agents.store."""

from pathlib import Path

import pytest

from distillcore_agents.agents.models import (
    ProcessingDecision,
    QADecision,
    QARecommendation,
    TriageDecision,
)
from distillcore_agents.orchestrator import PipelineResult
from distillcore_agents.store import AgentResultStore, ensure_schema


@pytest.fixture
def store(tmp_path: Path) -> AgentResultStore:
    return AgentResultStore(tmp_path / "test.db")


@pytest.fixture
def sample_result() -> PipelineResult:
    return PipelineResult(
        session_id="test123",
        source="/tmp/test.pdf",
        triage=TriageDecision(
            source_filename="test.pdf",
            preset="legal",
            needs_ocr=True,
            target_tokens=800,
            reasoning="Contains case numbers",
        ),
        processing=ProcessingDecision(
            source_filename="test.pdf",
            document_type="motion",
            chunk_count=12,
            structuring_coverage=0.97,
            chunking_coverage=0.99,
            end_to_end_coverage=0.95,
            validation_passed=True,
            document_id="doc-abc",
        ),
        qa=QADecision(
            verified=True,
            structuring_coverage=0.97,
            chunking_coverage=0.99,
            end_to_end_coverage=0.95,
            reasoning="All thresholds met",
        ),
    )


class TestEnsureSchema:
    def test_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "schema_test.db"
        ensure_schema(db)
        ensure_schema(db)  # should not raise


class TestSave:
    def test_save_returns_id(
        self, store: AgentResultStore, sample_result: PipelineResult
    ) -> None:
        row_id = store.save(sample_result, session_id="s1")
        assert row_id > 0

    def test_save_with_batch(
        self, store: AgentResultStore, sample_result: PipelineResult
    ) -> None:
        row_id = store.save(sample_result, session_id="s1", batch_id="b1")
        row = store.get(row_id)
        assert row is not None
        assert row["batch_id"] == "b1"


class TestGet:
    def test_found(
        self, store: AgentResultStore, sample_result: PipelineResult
    ) -> None:
        row_id = store.save(sample_result, session_id="s1")
        row = store.get(row_id)
        assert row is not None
        assert row["source"] == "/tmp/test.pdf"
        assert row["triage_preset"] == "legal"
        assert row["document_type"] == "motion"
        assert row["qa_verified"] == 1

    def test_not_found(self, store: AgentResultStore) -> None:
        assert store.get(9999) is None


class TestListSession:
    def test_lists(
        self, store: AgentResultStore, sample_result: PipelineResult
    ) -> None:
        store.save(sample_result, session_id="s1")
        store.save(sample_result, session_id="s1")
        store.save(sample_result, session_id="s2")
        rows = store.list_session("s1")
        assert len(rows) == 2

    def test_empty(self, store: AgentResultStore) -> None:
        assert store.list_session("nonexistent") == []


class TestWithQARecommendations:
    def test_persists_recommendations(
        self, store: AgentResultStore
    ) -> None:
        result = PipelineResult(
            session_id="s1",
            source="/tmp/test.pdf",
            triage=TriageDecision(source_filename="test.pdf"),
            processing=ProcessingDecision(source_filename="test.pdf"),
            qa=QADecision(
                verified=False,
                recommendations=[
                    QARecommendation(
                        issue="Low coverage",
                        action="change_preset",
                        parameter="preset",
                        suggested_value="legal",
                    ),
                ],
            ),
        )
        row_id = store.save(result, session_id="s1")
        row = store.get(row_id)
        assert row is not None
        import json

        recs = json.loads(row["qa_recommendations_json"])
        assert len(recs) == 1
        assert recs[0]["action"] == "change_preset"
