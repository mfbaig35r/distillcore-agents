"""Tests for distillcore_agents.agents.models."""

from distillcore_agents.agents.models import (
    Citation,
    ProcessingDecision,
    QADecision,
    QARecommendation,
    ResearchResult,
    TriageDecision,
)


class TestTriageDecision:
    def test_defaults(self) -> None:
        t = TriageDecision(source_filename="test.pdf")
        assert t.preset == "generic"
        assert t.needs_ocr is False
        assert t.target_tokens == 500
        assert t.chunk_strategy == "auto"
        assert t.min_tokens == 0

    def test_chunk_strategy(self) -> None:
        t = TriageDecision(
            source_filename="scan.pdf",
            chunk_strategy="sentence",
            min_tokens=50,
        )
        assert t.chunk_strategy == "sentence"
        assert t.min_tokens == 50

    def test_legal_preset(self) -> None:
        t = TriageDecision(
            source_filename="motion.pdf",
            preset="legal",
            needs_ocr=True,
            target_tokens=800,
            reasoning="Contains case numbers and court header",
        )
        assert t.preset == "legal"
        assert t.needs_ocr is True


class TestProcessingDecision:
    def test_defaults(self) -> None:
        p = ProcessingDecision(source_filename="test.txt")
        assert p.document_type == "unknown"
        assert p.validation_passed is False
        assert p.document_id is None

    def test_with_metrics(self) -> None:
        p = ProcessingDecision(
            source_filename="report.pdf",
            document_type="report",
            chunk_count=12,
            structuring_coverage=0.97,
            chunking_coverage=0.99,
            end_to_end_coverage=0.95,
            validation_passed=True,
            document_id="abc-123",
        )
        assert p.validation_passed is True
        assert p.chunk_count == 12


class TestQADecision:
    def test_passed(self) -> None:
        qa = QADecision(
            verified=True,
            structuring_coverage=0.98,
            chunking_coverage=1.0,
            end_to_end_coverage=0.96,
        )
        assert qa.verified is True
        assert qa.recommendations == []

    def test_failed_with_recommendations(self) -> None:
        qa = QADecision(
            verified=False,
            structuring_coverage=0.72,
            recommendations=[
                QARecommendation(
                    issue="Low structuring coverage",
                    action="change_preset",
                    parameter="preset",
                    suggested_value="legal",
                ),
            ],
        )
        assert qa.verified is False
        assert len(qa.recommendations) == 1
        assert qa.recommendations[0].action == "change_preset"


class TestResearchResult:
    def test_with_citations(self) -> None:
        r = ResearchResult(
            query="What are the custody arrangements?",
            answer="Based on the documents...",
            citations=[
                Citation(
                    document_id="doc-1",
                    source_filename="order.pdf",
                    chunk_index=3,
                    text_snippet="The court orders joint custody...",
                    score=0.92,
                ),
            ],
            documents_searched=5,
        )
        assert len(r.citations) == 1
        assert r.citations[0].score == 0.92

    def test_no_results(self) -> None:
        r = ResearchResult(
            query="unrelated question",
            answer="No relevant documents found.",
            documents_searched=0,
        )
        assert r.citations == []


class TestSerialization:
    def test_roundtrip(self) -> None:
        qa = QADecision(
            verified=False,
            recommendations=[
                QARecommendation(issue="test", action="reprocess"),
            ],
        )
        data = qa.model_dump()
        restored = QADecision.model_validate(data)
        assert restored.recommendations[0].action == "reprocess"
