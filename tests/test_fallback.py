"""
Comprehensive unit and integration tests for Phase 3.4 Intelligent LLM Provider Fallback.
Guarantees 100% mock isolation, zero external network sockets, and zero API quota consumption.
"""
import pytest
from unittest.mock import MagicMock

from core.llm.base import BaseLLMProvider
from core.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMFallbackError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    is_transient_error,
)
from core.llm.manager import LLMProviderManager
from core.llm.mock_provider import MockLLMProvider
from core.llm.models import (
    LLMMessage,
    LLMProviderConfig,
    LLMRequest,
    LLMResponse,
    LLMUsageMetadata,
    MessageRole,
)


def _make_dummy_provider(
    name: str,
    model: str = "test-model",
    available: bool = True,
    api_key_configured: bool = True,
) -> MagicMock:
    """Helper creating a typed mock conforming to BaseLLMProvider."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.provider_name = name
    provider.model_name = model
    provider.is_available.return_value = available
    cfg = MagicMock(spec=LLMProviderConfig)
    cfg.provider_name = name
    cfg.model_name = model
    cfg.is_api_key_configured = api_key_configured
    provider.config = cfg
    return provider


def _make_request(
    prompt: str = "Test prompt",
    allow_fallback: bool = True,
    is_sensitive: bool = False,
    allow_cross_provider_fallback: bool = False,
) -> LLMRequest:
    """Helper constructing validated LLMRequest with privacy controls."""
    return LLMRequest.from_prompt(
        prompt=prompt,
        allow_fallback=allow_fallback,
        is_sensitive=is_sensitive,
        allow_cross_provider_fallback=allow_cross_provider_fallback,
    )


class TestIntelligentFallback:
    """Test suite verifying cross-provider fallback orchestration, privacy, and error mapping."""

    def test_primary_success_no_fallback_triggered(self):
        """When primary (Groq) succeeds, response is returned immediately; secondary is never called."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq", "openai/gpt-oss-120b")
        gemini_mock = _make_dummy_provider("gemini", "gemini-3.8-flash")

        groq_resp = LLMResponse(
            content="Groq direct output",
            provider_name="groq",
            model_name="openai/gpt-oss-120b",
            finish_reason="stop",
            raw_metadata={"id": "groq-123"},
        )
        groq_mock.generate.return_value = groq_resp

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Hello career assistant")
        resp = mgr.generate(req)

        assert resp.content == "Groq direct output"
        assert resp.provider_name == "groq"
        assert resp.model_name == "openai/gpt-oss-120b"
        assert resp.raw_metadata.get("fallback_occurred") is not True

        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_not_called()

    def test_primary_transient_429_triggers_fallback_to_secondary(self):
        """When primary encounters rate limit (429), manager falls back to secondary exactly once."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq", "openai/gpt-oss-120b")
        gemini_mock = _make_dummy_provider("gemini", "gemini-3.8-flash")

        groq_mock.generate.side_effect = LLMRateLimitError(
            "Rate limit exceeded: 429 Too Many Requests", provider_name="groq"
        )
        gemini_resp = LLMResponse(
            content="Gemini fallback response",
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            finish_reason="stop",
            raw_metadata={"latency_sec": 1.2},
        )
        gemini_mock.generate.return_value = gemini_resp

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Evaluate resume alignment")
        resp = mgr.generate(req)

        assert resp.content == "Gemini fallback response"
        assert resp.provider_name == "gemini"
        assert resp.model_name == "gemini-3.8-flash"
        assert resp.raw_metadata["fallback_occurred"] is True
        assert resp.raw_metadata["primary_provider"] == "groq"
        assert "429" in resp.raw_metadata["primary_error"]
        assert resp.raw_metadata["secondary_provider"] == "gemini"

        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_called_once_with(req)

    def test_primary_transient_503_triggers_fallback(self):
        """When primary encounters 503 service unavailable, manager falls back to Gemini."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMProviderError(
            "503 Service Unavailable: upstream server capacity exhausted", provider_name="groq"
        )
        gemini_mock.generate.return_value = LLMResponse(
            content="Recovered via Gemini",
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            finish_reason="stop",
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        resp = mgr.generate(_make_request("Analyze skills"))
        assert resp.content == "Recovered via Gemini"
        assert resp.raw_metadata["fallback_occurred"] is True

    def test_primary_transient_timeout_triggers_fallback(self):
        """When primary times out, manager falls back to secondary."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMTimeoutError(
            "Request timed out after 30.0s", provider_name="groq"
        )
        gemini_mock.generate.return_value = LLMResponse(
            content="Gemini recovered after timeout",
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            finish_reason="stop",
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        resp = mgr.generate(_make_request("Generate interview questions"))
        assert resp.content == "Gemini recovered after timeout"
        assert resp.raw_metadata["fallback_occurred"] is True

    def test_primary_transient_connection_error_triggers_fallback(self):
        """When primary encounters network connection drop, manager falls back to secondary."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMProviderError(
            "Network error: connection reset by peer", provider_name="groq"
        )
        gemini_mock.generate.return_value = LLMResponse(
            content="Gemini recovered after network break",
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            finish_reason="stop",
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        resp = mgr.generate(_make_request("Review profile"))
        assert resp.content == "Gemini recovered after network break"
        assert resp.raw_metadata["fallback_occurred"] is True

    def test_primary_permanent_401_auth_error_no_fallback(self):
        """When primary fails with 401 unauthorized, fallback is NOT triggered; error raised immediately."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMAuthenticationError(
            "Groq API key is missing or expired. 401 Unauthorized.", provider_name="groq"
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Analyze resume")
        with pytest.raises(LLMAuthenticationError) as exc_info:
            mgr.generate(req)

        assert "401" in str(exc_info.value) or "Unauthorized" in str(exc_info.value)
        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_not_called()

    def test_primary_permanent_400_bad_request_no_fallback(self):
        """When primary fails with 400 bad request / configuration error, secondary is never invoked."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMConfigurationError(
            "400 Bad Request: Context length 135000 exceeds maximum window", provider_name="groq"
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Huge payload")
        with pytest.raises(LLMConfigurationError):
            mgr.generate(req)

        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_not_called()

    def test_primary_permanent_404_model_not_found_no_fallback(self):
        """When primary fails with 404 model not found, secondary is never invoked."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMProviderError(
            "404 model_not_found: The model deprecated-model does not exist", provider_name="groq"
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Check model")
        with pytest.raises(LLMProviderError, match="model_not_found"):
            mgr.generate(req)

        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_not_called()

    def test_primary_permanent_safety_block_no_fallback(self):
        """When primary triggers a safety or content block, secondary is never invoked."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMProviderError(
            "Generation blocked by safety filter: harmful content detected", provider_name="groq"
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Safety test")
        with pytest.raises(LLMProviderError, match="safety filter"):
            mgr.generate(req)

        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_not_called()

    def test_primary_and_secondary_both_fail_raises_fallback_error(self):
        """When both primary and secondary fail, LLMFallbackError is raised preserving both errors."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMRateLimitError(
            "429 rate limit exceeded", provider_name="groq"
        )
        gemini_mock.generate.side_effect = LLMTimeoutError(
            "Gemini timed out after 30s", provider_name="gemini"
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Both fail test")
        with pytest.raises(LLMFallbackError) as exc_info:
            mgr.generate(req)

        err = exc_info.value
        assert err.primary_provider == "groq"
        assert err.secondary_provider == "gemini"
        assert "429 rate limit" in str(err.primary_error)
        assert "timed out after 30s" in str(err.secondary_error)
        assert "Fallback chain exhausted" in str(err)

        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_called_once_with(req)

    def test_fallback_disabled_on_manager(self):
        """When manager has fallback_enabled=False, primary error is re-raised; secondary never called."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=False)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMRateLimitError(
            "429 rate limit", provider_name="groq"
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Fallback disabled test")
        with pytest.raises(LLMRateLimitError):
            mgr.generate(req)

        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_not_called()

    def test_fallback_disabled_on_request(self):
        """When request sets allow_fallback=False, primary error is re-raised; secondary never called."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMTimeoutError(
            "504 gateway timeout", provider_name="groq"
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Request fallback disabled", allow_fallback=False)
        assert req.is_fallback_permitted() is False

        with pytest.raises(LLMTimeoutError):
            mgr.generate(req)

        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_not_called()

    def test_sensitive_request_blocks_cross_provider_fallback_by_default(self):
        """Sensitive candidate PII request without explicit opt-in strictly blocks secondary fallback."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMRateLimitError(
            "429 rate limit", provider_name="groq"
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request(
            prompt="Candidate Resume: John Doe, SSN: 000-11-2222",
            is_sensitive=True,
            allow_cross_provider_fallback=False,
        )
        assert req.is_fallback_permitted() is False

        with pytest.raises(LLMRateLimitError):
            mgr.generate(req)

        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_not_called()

    def test_sensitive_request_allows_fallback_with_explicit_permission(self):
        """Sensitive request with allow_cross_provider_fallback=True successfully triggers secondary."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMRateLimitError(
            "429 rate limit", provider_name="groq"
        )
        gemini_mock.generate.return_value = LLMResponse(
            content="Sensitive data evaluated by Gemini with permission",
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            finish_reason="stop",
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request(
            prompt="Candidate Resume: Jane Smith",
            is_sensitive=True,
            allow_cross_provider_fallback=True,
        )
        assert req.is_fallback_permitted() is True

        resp = mgr.generate(req)
        assert resp.content == "Sensitive data evaluated by Gemini with permission"
        assert resp.raw_metadata["fallback_occurred"] is True

        groq_mock.generate.assert_called_once_with(req)
        gemini_mock.generate.assert_called_once_with(req)

    def test_direct_provider_routing_bypasses_fallback_chain(self):
        """Explicit provider_name parameter dispatches directly without fallback."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        gemini_mock.generate.return_value = LLMResponse(
            content="Direct Gemini invocation",
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            finish_reason="stop",
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Direct test")
        resp = mgr.generate(req, provider_name="gemini")

        assert resp.content == "Direct Gemini invocation"
        groq_mock.generate.assert_not_called()
        gemini_mock.generate.assert_called_once_with(req)

    def test_mock_mode_active_never_falls_back_to_production(self):
        """When MockLLMProvider is active, it executes directly without contacting Groq or Gemini."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")
        mock_prov = MockLLMProvider(canned_response="Deterministic mock response")

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)
        mgr.register_provider(mock_prov, set_active=True)

        req = _make_request("Offline testing")
        resp = mgr.generate(req)

        assert resp.content == "Deterministic mock response"
        assert resp.provider_name == "mock"
        groq_mock.generate.assert_not_called()
        gemini_mock.generate.assert_not_called()

    def test_max_two_attempts_strictly_enforced(self):
        """Verifies exactly 1 primary attempt and at most 1 secondary attempt (2 attempts total)."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMRateLimitError("429 rate limit", provider_name="groq")
        gemini_mock.generate.side_effect = LLMProviderError("503 overloaded", provider_name="gemini")

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = _make_request("Two attempts verify")
        with pytest.raises(LLMFallbackError):
            mgr.generate(req)

        assert groq_mock.generate.call_count == 1
        assert gemini_mock.generate.call_count == 1

    def test_secret_redaction_in_fallback_error_and_metadata(self):
        """Ensures API keys are never exposed in LLMFallbackError or response metadata."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMRateLimitError(
            "Rate limited with gsk_LiveGroqSecretKey12345678901234567890", provider_name="groq"
        )
        gemini_mock.generate.side_effect = LLMTimeoutError(
            "Timeout connecting with key AIzaSyFakeGeminiKey12345678901234567890", provider_name="gemini"
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        with pytest.raises(LLMFallbackError) as exc_info:
            mgr.generate(_make_request("Redaction test"))

        err_str = str(exc_info.value)
        assert "gsk_" not in err_str
        assert "AIzaSy" not in err_str
        assert "[REDACTED_API_KEY]" in err_str

    def test_generate_text_helper_with_fallback(self):
        """Verifies generate_text convenience method supports fallback parameters seamlessly."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq")
        gemini_mock = _make_dummy_provider("gemini")

        groq_mock.generate.side_effect = LLMTimeoutError("Request timed out", provider_name="groq")
        gemini_mock.generate.return_value = LLMResponse(
            content="Helper fallback verified",
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            finish_reason="stop",
        )

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        resp = mgr.generate_text(
            prompt="Prompt text",
            allow_fallback=True,
            is_sensitive=False,
        )
        assert resp.content == "Helper fallback verified"
        assert resp.raw_metadata["fallback_occurred"] is True

    def test_manager_status_diagnostics(self):
        """Verifies get_status reports correct fallback status states."""
        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        groq_mock = _make_dummy_provider("groq", available=True)
        gemini_mock = _make_dummy_provider("gemini", available=True)

        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        status = mgr.get_status()
        assert status["fallback_enabled"] is True
        assert status["fallback_status"] == "Active"
        assert status["primary_provider"] == "groq"
        assert status["secondary_provider"] == "gemini"

        # Toggle fallback off
        mgr.set_fallback_enabled(False)
        assert mgr.get_status()["fallback_status"] == "Disabled"

        # Enable with degraded availability
        mgr.set_fallback_enabled(True)
        gemini_mock.is_available.return_value = False
        assert mgr.get_status()["fallback_status"] == "Degraded"

        # Both unavailable
        groq_mock.is_available.return_value = False
        assert mgr.get_status()["fallback_status"] == "Unavailable"


class TestTransientErrorClassification:
    """Verifies is_transient_error classification across all error categories."""

    def test_typed_exceptions_classification(self):
        assert is_transient_error(LLMRateLimitError("Quota reached")) is True
        assert is_transient_error(LLMTimeoutError("Socket timeout")) is True
        assert is_transient_error(LLMAuthenticationError("Bad credentials")) is False
        assert is_transient_error(LLMConfigurationError("Invalid temperature")) is False
        assert is_transient_error(LLMResponseError("Malformed JSON")) is False
        assert is_transient_error(LLMFallbackError("p", Exception(), "s", Exception())) is False

    def test_status_code_classification(self):
        class DummyHTTPError(Exception):
            def __init__(self, code: int, msg: str):
                self.status_code = code
                super().__init__(msg)

        assert is_transient_error(DummyHTTPError(429, "Rate limited")) is True
        assert is_transient_error(DummyHTTPError(503, "Service unavailable")) is True
        assert is_transient_error(DummyHTTPError(502, "Bad gateway")) is True
        assert is_transient_error(DummyHTTPError(504, "Gateway timeout")) is True
        assert is_transient_error(DummyHTTPError(408, "Request timeout")) is True

        assert is_transient_error(DummyHTTPError(401, "Unauthorized")) is False
        assert is_transient_error(DummyHTTPError(403, "Forbidden")) is False
        assert is_transient_error(DummyHTTPError(400, "Bad Request")) is False
        assert is_transient_error(DummyHTTPError(404, "Not Found")) is False
        assert is_transient_error(DummyHTTPError(422, "Unprocessable")) is False

    def test_message_content_classification(self):
        assert is_transient_error(LLMProviderError("503 Service Unavailable")) is True
        assert is_transient_error(LLMProviderError("Capacity overloaded")) is True
        assert is_transient_error(LLMProviderError("Network error: connection reset")) is True
        assert is_transient_error(LLMProviderError("Request deadline exceeded")) is True

        assert is_transient_error(LLMProviderError("401 Unauthorized API key")) is False
        assert is_transient_error(LLMProviderError("404 model_not_found")) is False
        assert is_transient_error(LLMProviderError("Content blocked by safety filter")) is False
        assert is_transient_error(LLMProviderError("Bad request: invalid parameter")) is False

    def test_chained_cause_classification(self):
        class InnerTimeout(Exception):
            pass

        class OuterError(Exception):
            pass

        inner = InnerTimeout("ReadTimeout from socket")
        outer = OuterError("Provider wrapper failure")
        outer.__cause__ = inner

        assert is_transient_error(outer) is True

