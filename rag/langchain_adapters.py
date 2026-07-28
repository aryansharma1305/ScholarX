"""LangChain adapters around ScholarX's existing retrieval implementation."""
from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from processing.embeddings import generate_embedding
from rag.hybrid_search import hybrid_search
from vectorstore.query import query_vectors


class ScholarXRetriever(BaseRetriever):
    """Expose existing hybrid/filtered retrieval through LangChain's interface."""

    top_k: int = 10
    paper_ids: list[str] = Field(default_factory=list)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if self.paper_ids:
            query_embedding = generate_embedding(query)
            results = []
            for paper_id in self.paper_ids:
                results.extend(
                    query_vectors(
                        query_embedding=query_embedding,
                        top_k=self.top_k,
                        filter_metadata={"paper_id": paper_id},
                    )
                )
            results.sort(key=lambda result: result.score, reverse=True)
            results = results[:self.top_k]
        else:
            results = hybrid_search(query, top_k=self.top_k)

        return [
            Document(
                page_content=result.text,
                metadata={
                    **(result.metadata or {}),
                    "chunk_id": result.chunk_id,
                    "paper_id": result.paper_id,
                    "chunk_index": result.chunk_index,
                    "score": result.score,
                },
            )
            for result in results
        ]
