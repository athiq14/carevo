import pytest
from unittest.mock import patch, MagicMock
from core.research.web_researcher import WebResearcher, SearchResult, ResearchDossier


def test_empty_query_returns_offline():
    researcher = WebResearcher()
    dossier = researcher.search("")
    assert dossier.status == "OFFLINE"
    assert dossier.is_live_research is False
    assert len(dossier.results) == 0


def test_network_failure_handled_gracefully():
    researcher = WebResearcher()
    with patch("httpx.post", side_effect=Exception("Connection refused")):
        dossier = researcher.search("Python Developer")
        assert dossier.status == "OFFLINE"
        assert dossier.is_live_research is False
        assert "temporarily unreachable" in dossier.summary.lower()


def test_successful_html_parsing():
    sample_html = """
    <html>
    <body>
        <a class="result__url" href="https://example.com/guide">example.com</a>
        <a class="result__a" href="https://example.com/guide">IRDAI IC-38 Guide 2026</a>
        <a class="result__snippet" href="https://example.com/guide">Complete certification guide for insurance advisors.</a>
    </body>
    </html>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = sample_html

    researcher = WebResearcher()
    with patch("httpx.post", return_value=mock_resp):
        dossier = researcher.search("Insurance Advisor")
        assert dossier.status == "LIVE_SUCCESS"
        assert dossier.is_live_research is True
        assert len(dossier.results) == 1
        assert dossier.results[0].title == "IRDAI IC-38 Guide 2026"
        assert dossier.results[0].url == "https://example.com/guide"
        assert "insurance advisors" in dossier.results[0].snippet.lower()

