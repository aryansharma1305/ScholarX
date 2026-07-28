"""Network-free tests for the LangGraph RAG orchestration."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.generator import RAGResponse
from rag.langchain_models import (
    deterministic_answer_grade,
    deterministic_evidence_grade,
)
from rag.langgraph_pipeline import _initial_state, build_research_graph, run_research_graph
from vectorstore.query import QueryResult


def _chunk(score: float = 0.8, index: int = 0) -> QueryResult:
    return QueryResult(
        chunk_id=f"paper-1_chunk_{index}",
        paper_id="paper-1",
        chunk_index=index,
        text=(
            "Retrieval augmented generation combines external retrieval with "
            "language-model generation to ground answers in source evidence."
        ),
        score=score,
        metadata={"title": "RAG Study", "year": 2024},
    )


def _state(fetch_papers: bool = True):
    return _initial_state(
        "How does retrieval augmented generation improve grounding?",
        top_k=3,
        fetch_papers=fetch_papers,
        selected_paper_ids=None,
        system_prompt=None,
        use_hybrid_search=True,
        use_reranking=True,
        use_citation_boost=False,
    )


class LangGraphPipelineTests(unittest.TestCase):
    def setUp(self):
        self.llm_grader_patch = patch(
            "rag.langchain_models.settings.langgraph_use_llm_grader",
            False,
        )
        self.llm_grader_patch.start()

    def tearDown(self):
        self.llm_grader_patch.stop()

    @patch("rag.langgraph_pipeline.validate_answer")
    @patch("rag.langgraph_pipeline.generate_grounded_answer")
    @patch("rag.langgraph_pipeline.apply_citation_boost")
    @patch("rag.langgraph_pipeline.rerank_results")
    @patch("rag.langgraph_pipeline.hybrid_search")
    @patch("rag.langgraph_pipeline.expand_query_with_llm")
    def test_sufficient_local_evidence_skips_external_search(
        self,
        expand_mock,
        hybrid_mock,
        rerank_mock,
        citation_mock,
        generate_mock,
        validate_mock,
    ):
        expand_mock.return_value = ["retrieval augmented generation grounding"]
        local_chunks = [_chunk(0.82, index) for index in range(3)]
        hybrid_mock.return_value = local_chunks
        rerank_mock.side_effect = lambda results, metadata: results
        citation_mock.side_effect = lambda results: results
        generate_mock.return_value = (
            "Grounded generation uses retrieved evidence to support its answer "
            "and reduce unsupported claims [Context 1]."
        )
        validate_mock.return_value = {
            "supported": True,
            "confidence": 0.9,
            "reason": "supported",
            "method": "test",
        }

        graph = build_research_graph()
        with patch("rag.langgraph_pipeline.fetch_papers_by_topic") as search_mock:
            result = graph.invoke(_state(fetch_papers=True))

        search_mock.assert_not_called()
        self.assertTrue(result["evidence_sufficient"])
        self.assertTrue(result["answer_supported"])
        self.assertEqual(result["external_searches"], 0)
        self.assertEqual(result["answer_retries"], 0)
        self.assertEqual(len(result["citations"]), 2)

    @patch("rag.langgraph_pipeline.validate_answer")
    @patch("rag.langgraph_pipeline.generate_grounded_answer")
    @patch("rag.langgraph_pipeline._paper_is_ingested")
    @patch("rag.langgraph_pipeline.ingest_pdf_from_url")
    @patch("rag.langgraph_pipeline.fetch_papers_by_topic")
    @patch("rag.langgraph_pipeline.rerank_results")
    @patch("rag.langgraph_pipeline.hybrid_search")
    @patch("rag.langgraph_pipeline.expand_query_with_llm")
    def test_insufficient_evidence_searches_and_ingests_once(
        self,
        expand_mock,
        hybrid_mock,
        rerank_mock,
        search_mock,
        ingest_mock,
        ingested_mock,
        generate_mock,
        validate_mock,
    ):
        expand_mock.return_value = ["RAG grounding evaluation"]
        hybrid_mock.side_effect = [
            [],
            [_chunk(0.85, index) for index in range(3)],
            [_chunk(0.8, index) for index in range(3)],
        ]
        rerank_mock.side_effect = lambda results, metadata: results
        search_mock.return_value = [{
            "paper_id": "paper-1",
            "title": "RAG Grounding Study",
            "authors_string": "Researcher",
            "abstract": "Evaluation of grounded RAG.",
            "year": 2024,
            "pdf_url": "https://example.com/paper.pdf",
            "citation_count": 50,
            "source": "core",
        }]
        ingested_mock.return_value = False
        ingest_mock.return_value = "paper-1"
        generate_mock.return_value = (
            "The retrieved study reports that grounding answers in external "
            "evidence improves factual support [Context 1]."
        )
        validate_mock.return_value = {
            "supported": True,
            "confidence": 0.9,
            "reason": "supported",
            "method": "test",
        }

        result = build_research_graph().invoke(_state(fetch_papers=True))

        self.assertEqual(search_mock.call_count, 1)
        self.assertEqual(ingest_mock.call_count, 1)
        self.assertEqual(result["external_searches"], 1)
        self.assertEqual(result["ingested_paper_ids"], ["paper-1"])
        self.assertEqual(result["retrieval_attempts"], 2)

    @patch("rag.langgraph_pipeline.validate_answer")
    @patch("rag.langgraph_pipeline.generate_grounded_answer")
    @patch("rag.langgraph_pipeline.rerank_results")
    @patch("rag.langgraph_pipeline.hybrid_search")
    @patch("rag.langgraph_pipeline.expand_query_with_llm")
    @patch("rag.langgraph_pipeline.rewrite_query_for_retry")
    def test_answer_validation_allows_only_one_retry(
        self,
        rewrite_mock,
        expand_mock,
        hybrid_mock,
        rerank_mock,
        generate_mock,
        validate_mock,
    ):
        rewrite_mock.return_value = "RAG grounding empirical evidence"
        expand_mock.return_value = ["RAG grounding"]
        hybrid_mock.return_value = [_chunk(0.8, index) for index in range(3)]
        rerank_mock.side_effect = lambda results, metadata: results
        generate_mock.return_value = (
            "This is a sufficiently long generated answer with a citation "
            "reference to the retrieved research context [Context 1]."
        )
        validate_mock.return_value = {
            "supported": False,
            "confidence": 0.3,
            "reason": "not fully supported",
            "method": "test",
        }

        result = build_research_graph().invoke(_state(fetch_papers=False))

        self.assertEqual(result["answer_retries"], 1)
        self.assertEqual(generate_mock.call_count, 2)
        self.assertIn("Evidence note:", result["answer"])

    @patch("rag.langgraph_pipeline.validate_answer")
    @patch("rag.langgraph_pipeline.generate_grounded_answer")
    @patch("rag.langgraph_pipeline.rerank_results")
    @patch("rag.langgraph_pipeline.hybrid_search")
    @patch("rag.langgraph_pipeline.expand_query_with_llm")
    def test_sqlite_checkpoint_and_progress_events(
        self,
        expand_mock,
        hybrid_mock,
        rerank_mock,
        generate_mock,
        validate_mock,
    ):
        expand_mock.return_value = ["RAG grounding"]
        hybrid_mock.return_value = [_chunk(0.8, index) for index in range(3)]
        rerank_mock.side_effect = lambda results, metadata: results
        generate_mock.return_value = (
            "Retrieved evidence grounds this answer in research paper context "
            "and supplies valid citations [Context 1]."
        )
        validate_mock.return_value = {
            "supported": True,
            "confidence": 0.9,
            "reason": "supported",
            "method": "test",
        }

        events = []
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "graph.sqlite3"
            with patch(
                "rag.langgraph_pipeline.settings.langgraph_checkpoint_path",
                str(checkpoint_path),
            ):
                result = run_research_graph(
                    "How does RAG improve grounding?",
                    top_k=3,
                    fetch_papers=False,
                    thread_id="test-thread",
                    use_citation_boost=False,
                    progress_callback=lambda stage, message: events.append(stage),
                )

            self.assertTrue(checkpoint_path.exists())
            self.assertGreater(checkpoint_path.stat().st_size, 0)

        self.assertEqual(result["workflow"]["thread_id"], "test-thread")
        self.assertIn("retrieve", events)
        self.assertIn("validate_answer", events)

    def test_deterministic_graders(self):
        chunks = [
            {
                "paper_id": "paper-1",
                "chunk_index": index,
                "score": 0.8,
                "text": "Relevant research evidence.",
            }
            for index in range(3)
        ]
        evidence = deterministic_evidence_grade(chunks)
        answer = deterministic_answer_grade(
            "A grounded answer that is long enough to be useful and reviewable.",
            [{"paper_id": "paper-1", "chunk_index": 0, "score": 0.8}],
            chunks,
        )
        self.assertTrue(evidence["sufficient"])
        self.assertTrue(answer["supported"])


if __name__ == "__main__":
    unittest.main()
