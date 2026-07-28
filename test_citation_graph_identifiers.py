"""Regression tests for citation graph identifier handling."""
from unittest.mock import Mock, patch

import pytest
import requests

from ingestion.semantic_scholar_enhanced import _request_with_retries
from rag.citation_graph_retriever import (
    _paper_aliases,
    _semantic_scholar_query_id,
)


def test_arxiv_identifier_is_version_free() -> None:
    metadata = {"source": "arxiv", "arxiv_id": "2307.10652v5"}

    assert _semantic_scholar_query_id("2307.10652v5", metadata) == "ARXIV:2307.10652"
    assert _paper_aliases("2307.10652v5", metadata) == {"arxiv:2307.10652"}


def test_local_identifier_is_not_sent_to_semantic_scholar() -> None:
    assert _semantic_scholar_query_id("compat_test", {}) is None


def test_external_ids_match_local_aliases() -> None:
    remote = _paper_aliases(
        "a" * 40,
        external_ids={"ArXiv": "2307.10652", "DOI": "10.1000/example"},
    )

    assert "arxiv:2307.10652" in remote
    assert "doi:10.1000/example" in remote
    assert "s2:" + ("a" * 40) in remote


def test_permanent_http_error_is_not_retried() -> None:
    response = Mock(status_code=404, headers={})
    response.raise_for_status.side_effect = requests.HTTPError(response=response)

    with patch("requests.request", return_value=response) as request:
        with pytest.raises(requests.HTTPError):
            _request_with_retries("GET", "https://example.test", max_retries=3)

    request.assert_called_once()
