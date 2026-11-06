"""
Unit tests for the Groq LLM Provider (Phase 3.3).
Uses mocked Groq SDK client to verify initialization, message conversion,
generation parameter propagation, usage accounting, finish reason mapping,
typed error translation, and secret redaction with zero external network calls.
"""
from unittest.mock import MagicMock
import httpx
import pytest
import groq

from core.llm import (
    GeminiProvider,
    GroqProvider,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMGenerationConfig,
    LLMMessage,
    LLMProviderConfig,
    LLMProviderError,
    LLMProviderManager,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    MessageRole,
    MockLLMProvider,
)


def _make_mock_groq_response(
    content: str = "Mocked Groq text output",
    finish_reason: str = "stop",
    prompt_tokens: int = 15,
    completion_tokens: int = 25,
    total_tokens: int = 40,
    has_usage: bool = True,
    empty_choices: bool = False,
    message_content_none: bool = False,
    reasoning_tokens: int = None,
):
    """Helper to synthesize a mock Groq ChatCompletion response."""
    mock_resp = MagicMock()
    mock_resp.id = "chatcmpl-mock-groq-98765"
    mock_resp.model = "openai/gpt-oss-120b"
    mock_resp.system_fingerprint = "fp_groq_mock_123"

    if empty_choices:
        mock_resp.choices = []
        mock_resp.usage = None
        mock_resp.x_groq = None
        return mock_resp

    mock_choice = MagicMock()
    mock_choice.finish_reason = finish_reason
    mock_choice.index = 0

    mock_message = MagicMock()
    mock_message.role = "assistant"
    mock_message.content = None if message_content_none else content
    mock_choice.message = mock_message

    mock_resp.choices = [mock_choice]

    if has_usage:
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = prompt_tokens
        mock_usage.completion_tokens = completion_tokens
        mock_usage.total_tokens = total_tokens
        if reasoning_tokens is not None:
            mock_details = MagicMock()
            mock_details.reasoning_tokens = reasoning_tokens
            mock_usage.completion_tokens_details = mock_details
        else:
            mock_usage.completion_tokens_details = None
        mock_resp.usage = mock_usage
    else:
        mock_resp.usage = None

    mock_x_groq = MagicMock()
    mock_x_groq.id = "req_mock_groq_internal_id"
    mock_resp.x_groq = mock_x_groq

    return mock_resp


def _make_dummy_http_pair(status_code: int = 400):
    """Helper to synthesize dummy httpx Request and Response for Groq exceptions."""
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(status_code, request=req)
    return req, resp


class TestGroqProviderInitialization:
    """Verifies GroqProvider initialization, configuration loading, and credential checks."""

    def test_missing_api_key_initialization(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "")
        provider = GroqProvider()
        assert provider.provider_name == "groq"
        assert provider.is_available() is False
        assert provider.api_key is None
        assert provider.model_name == "openai/gpt-oss-120b"

        with pytest.raises(LLMAuthenticationError) as exc_info:
            provider.validate_configuration()
        assert "Groq API key is missing" in str(exc_info.value)
        assert exc_info.value.provider_name == "groq"

    def test_explicit_api_key_configuration(self):
        config = LLMProviderConfig(
            provider_name="groq",
            model_name="openai/gpt-oss-120b",
            api_key="gsk_explicit_test_key_1234567890123456",
        )
        provider = GroqProvider(config=config)
        assert provider.is_available() is True
        assert provider.model_name == "openai/gpt-oss-120b"
        assert provider.api_key == "gsk_explicit_test_key_1234567890123456"
        provider.validate_configuration()

    def test_environment_variable_fallback(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_env_test_token_9876543210123456")
        monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
        provider = GroqProvider()
        assert provider.is_available() is True
        assert provider.model_name == "llama-3.1-8b-instant"
        assert provider.api_key == "gsk_env_test_token_9876543210123456"

    def test_invalid_temperature_rejected(self):
        config = LLMProviderConfig(
            provider_name="groq",
            model_name="openai/gpt-oss-120b",
            api_key="gsk_test_token_1234567890123456",
        )
        provider = GroqProvider(config=config)
        provider._config = provider._config.model_copy(update={"temperature": 2.5})
        with pytest.raises(LLMConfigurationError) as exc_info:
            provider.validate_configuration()
        assert "Invalid temperature" in str(exc_info.value)

    def test_invalid_timeout_rejected(self):
        config = LLMProviderConfig(
            provider_name="groq",
            model_name="openai/gpt-oss-120b",
            api_key="gsk_test_token_1234567890123456",
        )
        provider = GroqProvider(config=config)
        provider._config = provider._config.model_copy(update={"timeout": -5.0})
        with pytest.raises(LLMConfigurationError) as exc_info:
            provider.validate_configuration()
        assert "Invalid timeout" in str(exc_info.value)


class TestGroqMessageConversion:
    """Verifies mapping from LLMMessage domain objects to Groq chat completions format."""

    def test_system_user_assistant_message_conversion(self):
        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            )
        )
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content="You are a career assistant."),
            LLMMessage(role=MessageRole.USER, content="Critique this resume."),
            LLMMessage(role=MessageRole.ASSISTANT, content="Here is the critique."),
            LLMMessage(role=MessageRole.USER, content="Refine skills section."),
        ]
        groq_msgs = provider._convert_messages(messages)
        assert len(groq_msgs) == 4
        assert groq_msgs[0] == {"role": "system", "content": "You are a career assistant."}
        assert groq_msgs[1] == {"role": "user", "content": "Critique this resume."}
        assert groq_msgs[2] == {"role": "assistant", "content": "Here is the critique."}
        assert groq_msgs[3] == {"role": "user", "content": "Refine skills section."}

    def test_empty_messages_defaults_to_single_user_message(self):
        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            )
        )
        groq_msgs = provider._convert_messages([])
        assert len(groq_msgs) == 1
        assert groq_msgs[0] == {"role": "user", "content": " "}


class TestGroqGenerationAndParameters:
    """Verifies generation request execution, hyperparameter propagation, and client delegation."""

    def test_successful_generation(self):
        mock_client = MagicMock()
        mock_response = _make_mock_groq_response(
            content="Senior Software Engineer with 8 years of experience in distributed systems.",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=40,
            total_tokens=60,
        )
        mock_client.chat.completions.create.return_value = mock_response

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        request = LLMRequest(
            messages=[LLMMessage(role=MessageRole.USER, content="Extract job title")],
            generation_config=LLMGenerationConfig(temperature=0.3, max_output_tokens=100),
        )

        response = provider.generate(request)
        assert isinstance(response, LLMResponse)
        assert response.content == "Senior Software Engineer with 8 years of experience in distributed systems."
        assert response.provider_name == "groq"
        assert response.model_name == "openai/gpt-oss-120b"
        assert response.finish_reason == "stop"
        assert response.usage.prompt_tokens == 20
        assert response.usage.completion_tokens == 40
        assert response.usage.total_tokens == 60
        assert response.raw_metadata["id"] == "chatcmpl-mock-groq-98765"
        assert response.raw_metadata["system_fingerprint"] == "fp_groq_mock_123"
        assert response.raw_metadata["x_groq_id"] == "req_mock_groq_internal_id"

    def test_generation_parameters_propagation(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_groq_response()

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
                timeout=45.0,
            ),
            client=mock_client,
        )

        request = LLMRequest(
            messages=[
                LLMMessage(role=MessageRole.SYSTEM, content="System prompt"),
                LLMMessage(role=MessageRole.USER, content="User prompt"),
            ],
            generation_config=LLMGenerationConfig(
                temperature=0.2,
                top_p=0.85,
                max_output_tokens=50,
                stop_sequences=["###", "END"],
                timeout=25.0,
            ),
        )

        provider.generate(request)

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        assert call_kwargs["model"] == "openai/gpt-oss-120b"
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["top_p"] == 0.85
        assert call_kwargs["max_tokens"] == 50
        assert call_kwargs["stop"] == ["###", "END"]
        assert call_kwargs["timeout"] == 25.0
        assert len(call_kwargs["messages"]) == 2

    def test_max_output_tokens_fallback_to_provider_config(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_groq_response()

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
                max_output_tokens=256,
            ),
            client=mock_client,
        )

        request = LLMRequest(
            messages=[LLMMessage(role=MessageRole.USER, content="Prompt without max tokens")]
        )
        provider.generate(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 256


class TestGroqResponseHandling:
    """Verifies parsing of responses, token usage accounting, and rejection of invalid payloads."""

    def test_missing_usage_metadata_handled_gracefully(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_groq_response(has_usage=False)

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        resp = provider.generate(
            LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
        )
        assert resp.usage is None
        assert resp.content == "Mocked Groq text output"

    def test_finish_reason_stop_and_length(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_groq_response(finish_reason="length")

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        resp = provider.generate(
            LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
        )
        assert resp.finish_reason == "length"

    def test_empty_choices_raises_response_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_groq_response(empty_choices=True)

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMResponseError) as exc_info:
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )
        assert "empty choices list" in str(exc_info.value)
        assert exc_info.value.provider_name == "groq"

    def test_empty_text_content_raises_response_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_groq_response(content="")

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMResponseError) as exc_info:
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )
        assert "empty or blank text content" in str(exc_info.value)

    def test_empty_text_with_length_finish_reason_raises_specific_diagnostic(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_groq_response(
            content="",
            finish_reason="length",
            reasoning_tokens=10,
        )

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMResponseError) as exc_info:
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )
        msg = str(exc_info.value)
        assert "reached token limit before visible text was generated" in msg
        assert "10 reasoning tokens consumed" in msg
        assert "Increase max_output_tokens" in msg
        assert exc_info.value.provider_name == "groq"

    def test_reasoning_tokens_captured_in_usage_and_raw_metadata(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_groq_response(
            content="Valid response",
            reasoning_tokens=42,
            completion_tokens=50,
            total_tokens=65,
        )

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        response = provider.generate(
            LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
        )
        assert response.content == "Valid response"
        assert response.usage.reasoning_tokens == 42
        assert response.raw_metadata["reasoning_tokens"] == 42
        assert response.raw_metadata["finish_reason"] == "stop"

    def test_reasoning_effort_propagation(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_groq_response()

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
            reasoning_effort="low",
        )

        request = LLMRequest(
            messages=[LLMMessage(role=MessageRole.USER, content="Test with low effort")]
        )
        provider.generate(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "low"

    def test_none_message_content_raises_response_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_groq_response(message_content_none=True)

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMResponseError) as exc_info:
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )
        assert "empty or blank text content" in str(exc_info.value)


class TestGroqErrorHandlingAndRedaction:
    """Verifies mapping of Groq SDK exceptions to domain exceptions and secret scrubbing."""

    def test_authentication_error_mapping(self):
        req, resp = _make_dummy_http_pair(401)
        groq_err = groq.AuthenticationError(
            "Invalid API Key: gsk_secret_token_12345678901234567890",
            response=resp,
            body=None,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq_err

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_secret_token_12345678901234567890",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMAuthenticationError) as exc_info:
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )
        msg = str(exc_info.value)
        assert "[groq]" in msg
        assert "gsk_secret_token" not in msg
        assert "[REDACTED_API_KEY]" in msg

    def test_permission_denied_error_mapping(self):
        req, resp = _make_dummy_http_pair(403)
        groq_err = groq.PermissionDeniedError("Forbidden access", response=resp, body=None)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq_err

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMAuthenticationError):
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )

    def test_rate_limit_error_mapping(self):
        req, resp = _make_dummy_http_pair(429)
        groq_err = groq.RateLimitError("Rate limit reached. Try again in 2s.", response=resp, body=None)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq_err

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMRateLimitError) as exc_info:
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )
        assert "Rate limit" in str(exc_info.value)
        assert exc_info.value.provider_name == "groq"

    def test_timeout_error_mapping(self):
        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        groq_err = groq.APITimeoutError(request=req)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq_err

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )
        assert "timed out" in str(exc_info.value).lower()
        assert exc_info.value.provider_name == "groq"

    def test_httpx_timeout_error_mapping(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = httpx.ReadTimeout("Read timed out after 30s")

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )
        assert "Request timed out" in str(exc_info.value)

    def test_bad_request_configuration_error_mapping(self):
        req, resp = _make_dummy_http_pair(400)
        groq_err = groq.BadRequestError("Invalid parameter: temperature must be <= 2", response=resp, body=None)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq_err

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMConfigurationError) as exc_info:
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )
        assert "Invalid parameter" in str(exc_info.value)

    def test_api_connection_error_mapping(self):
        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        groq_err = groq.APIConnectionError(request=req)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq_err

        provider = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_test_key_1234567890123456",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMProviderError) as exc_info:
            provider.generate(
                LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Test")])
            )
        assert "Network error" in str(exc_info.value)


class TestGroqManagerIntegration:
    """Verifies GroqProvider registration and coexistence in LLMProviderManager."""

    def test_manager_registers_mock_gemini_groq(self):
        manager = LLMProviderManager()
        mock_p = MockLLMProvider()
        gemini_p = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="AIzaSyMockKeyForGeminiTest123456",
            )
        )
        groq_p = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_mock_groq_key_1234567890123456",
            )
        )

        manager.register_provider(mock_p, set_active=True)
        manager.register_provider(gemini_p, set_active=False)
        manager.register_provider(groq_p, set_active=False)

        assert manager.active_provider_name == "mock"
        assert set(manager.list_providers()) == {"mock", "gemini", "groq"}

        status = manager.get_status()
        assert status["provider_count"] == 3
        assert status["active_provider"] == "mock"
        assert "groq" in status["providers"]
        assert status["providers"]["groq"]["model_name"] == "openai/gpt-oss-120b"
        assert status["providers"]["groq"]["is_available"] is True
        assert status["providers"]["groq"]["is_api_key_configured"] is True

    def test_manager_dispatches_to_groq_when_selected(self):
        manager = LLMProviderManager()
        mock_p = MockLLMProvider()

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = _make_mock_groq_response(
            content="Groq dispatched successfully."
        )

        groq_p = GroqProvider(
            config=LLMProviderConfig(
                provider_name="groq",
                model_name="openai/gpt-oss-120b",
                api_key="gsk_mock_groq_key_1234567890123456",
            ),
            client=mock_groq_client,
        )

        manager.register_provider(mock_p, set_active=True)
        manager.register_provider(groq_p, set_active=False)

        # Explicit dispatch to groq by name
        req = LLMRequest(messages=[LLMMessage(role=MessageRole.USER, content="Hello Groq")])
        resp = manager.generate(req, provider_name="groq")
        assert resp.content == "Groq dispatched successfully."
        assert resp.provider_name == "groq"

        # Switching active provider to groq
        manager.set_active_provider("groq")
        assert manager.active_provider_name == "groq"
        resp2 = manager.generate(req)
        assert resp2.content == "Groq dispatched successfully."

