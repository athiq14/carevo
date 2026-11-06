"""
Unit tests for the Google Gemini LLM Provider (Phase 3.2).
Uses mocked Google GenAI client to verify both the modern Interactions API
(client.interactions.create) and models API (client.models.generate_content),
covering message mapping, parameter propagation, privacy, safety blocks,
and error translation with zero external network calls or quota consumption.
"""
from unittest.mock import MagicMock
import httpx
import pytest
from google.genai import errors, types

from core.llm import (
    GeminiProvider,
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
)


def _make_mock_response(
    text: str = "Mocked Gemini text",
    finish_reason: str = "STOP",
    prompt_tokens: int = 12,
    completion_tokens: int = 18,
    total_tokens: int = 30,
    has_usage: bool = True,
    empty_candidates: bool = False,
    prompt_blocked: bool = False,
):
    """Helper to synthesize a mock GenerateContentResponse."""
    mock_resp = MagicMock(spec=types.GenerateContentResponse)
    mock_resp.text = text
    mock_resp.model_version = "gemini-3.8-flash-001"
    mock_resp.response_id = "mock-resp-id-12345"

    if empty_candidates:
        mock_resp.candidates = []
        if prompt_blocked:
            mock_feedback = MagicMock()
            mock_feedback.block_reason = "SAFETY_PROMPT_BLOCKED"
            mock_resp.prompt_feedback = mock_feedback
        else:
            mock_resp.prompt_feedback = None
        return mock_resp

    mock_candidate = MagicMock()
    mock_candidate.finish_reason = finish_reason
    mock_resp.candidates = [mock_candidate]

    if has_usage:
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = prompt_tokens
        mock_usage.candidates_token_count = completion_tokens
        mock_usage.total_token_count = total_tokens
        mock_resp.usage_metadata = mock_usage
    else:
        mock_resp.usage_metadata = None

    return mock_resp


def _make_mock_interaction(
    output_text: str = "Mocked interaction output text",
    status: str = "completed",
    input_tokens: int = 15,
    output_tokens: int = 25,
    total_tokens: int = 40,
    has_usage: bool = True,
    errors_list: list = None,
):
    """Helper to synthesize a mock Interaction object from client.interactions.create."""
    mock_interaction = MagicMock()
    mock_interaction.id = "interaction-xyz-987"
    mock_interaction.status = status
    mock_interaction.output_text = output_text
    mock_interaction.errors = errors_list

    if has_usage:
        mock_usage = MagicMock()
        mock_usage.total_input_tokens = input_tokens
        mock_usage.total_output_tokens = output_tokens
        mock_usage.total_tokens = total_tokens
        mock_interaction.usage = mock_usage
    else:
        mock_interaction.usage = None

    return mock_interaction


class TestGeminiProviderInitialization:
    """Verifies initialization, missing-key safety, and availability checking."""

    def test_missing_api_key_initialization(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        provider = GeminiProvider()

        # Must not raise on instantiation
        assert provider.provider_name == "gemini"
        assert provider.is_available() is False

        # Attempting validation must raise LLMAuthenticationError
        with pytest.raises(LLMAuthenticationError) as exc_info:
            provider.validate_configuration()
        assert "API key is missing" in str(exc_info.value)

    def test_explicit_api_key_configuration(self):
        config = LLMProviderConfig(
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            api_key="AIzaSyTestMockKey1234567890abcdef123456",
        )
        provider = GeminiProvider(config=config)
        assert provider.is_available() is True
        provider.validate_configuration()  # Should not raise

    def test_environment_variable_fallback(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyEnvKey1234567890abcdef1234567")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")

        provider = GeminiProvider()
        assert provider.is_available() is True
        assert provider.model_name == "gemini-3.8-flash"
        provider.validate_configuration()

    def test_invalid_temperature_rejected(self):
        config = LLMProviderConfig(
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            api_key="valid_key",
            temperature=2.0,  # Valid boundary
        )
        provider = GeminiProvider(config=config)
        provider.validate_configuration()

        with pytest.raises(Exception):
            LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
                temperature=2.5,
            )


class TestGeminiInteractionsApi:
    """Tests Google's recommended Interactions API adapter (client.interactions.create)."""

    def test_interactions_api_successful_generation(self):
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = _make_mock_interaction(
            output_text="Analysis complete: Strong backend engineering experience.",
            status="completed",
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
        )

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="AIzaSyTestMockKey1234567890abcdef",
            ),
            client=mock_client,
            use_interactions_api=True,
        )

        req = LLMRequest.from_prompt("Evaluate skills", system_prompt="You are a career analyst.")
        resp = provider.generate(req)

        assert resp.content == "Analysis complete: Strong backend engineering experience."
        assert resp.provider_name == "gemini"
        assert resp.model_name == "gemini-3.8-flash"
        assert resp.finish_reason == "stop"
        assert resp.usage.total_tokens == 30
        assert resp.raw_metadata["interaction_id"] == "interaction-xyz-987"

        # Verify call arguments to interactions.create
        mock_client.interactions.create.assert_called_once()
        _, kwargs = mock_client.interactions.create.call_args
        assert kwargs["model"] == "gemini-3.8-flash"
        assert kwargs["input"] == "Evaluate skills"
        assert kwargs["system_instruction"] == "You are a career analyst."
        assert kwargs["store"] is False  # Privacy guarantee: no server-side retention

    def test_interactions_api_multi_turn_formatting(self):
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = _make_mock_interaction("Turn response")

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
            use_interactions_api=True,
        )

        messages = [
            LLMMessage(role=MessageRole.USER, content="First user message"),
            LLMMessage(role=MessageRole.ASSISTANT, content="First model response"),
            LLMMessage(role=MessageRole.USER, content="Second user message"),
        ]
        req = LLMRequest(messages=messages)
        resp = provider.generate(req)

        assert resp.content == "Turn response"
        _, kwargs = mock_client.interactions.create.call_args
        assert "User: First user message" in kwargs["input"]
        assert "Model: First model response" in kwargs["input"]
        assert "User: Second user message" in kwargs["input"]

    def test_interactions_api_parameters_propagation(self):
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = _make_mock_interaction("Text")

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
            use_interactions_api=True,
        )

        req = LLMRequest.from_prompt(
            prompt="Prompt",
            generation_config=LLMGenerationConfig(
                temperature=0.2,
                top_p=0.9,
                max_output_tokens=300,
                stop_sequences=["STOP_HERE"],
                timeout=45.0,
            ),
        )
        provider.generate(req)

        _, kwargs = mock_client.interactions.create.call_args
        assert kwargs["timeout"] == 45.0
        assert kwargs["generation_config"]["temperature"] == 0.2
        assert kwargs["generation_config"]["top_p"] == 0.9
        assert kwargs["generation_config"]["max_output_tokens"] == 300
        assert kwargs["generation_config"]["stop_sequences"] == ["STOP_HERE"]

    def test_interactions_api_incomplete_status_sets_length_finish_reason(self):
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = _make_mock_interaction(
            output_text="Partial generated text truncated",
            status="incomplete",
        )

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
            use_interactions_api=True,
        )

        resp = provider.generate_text("Prompt")
        assert resp.finish_reason == "length"

    def test_interactions_api_empty_output_raises_error(self):
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = _make_mock_interaction(output_text="")

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
            use_interactions_api=True,
        )

        with pytest.raises(LLMResponseError, match="empty output text"):
            provider.generate_text("Prompt")


class TestGeminiModelsApiFallback:
    """Tests the secondary models API (client.models.generate_content) path."""

    def _make_api_error(self, code: int, message: str) -> errors.APIError:
        return errors.APIError(code, {"error": {"message": message, "code": code}})

    def test_models_api_successful_generation(self):
        mock_client = MagicMock()
        mock_resp = _make_mock_response(
            text="Models API Output",
            finish_reason="STOP",
            prompt_tokens=25,
            completion_tokens=10,
            total_tokens=35,
        )
        mock_client.models.generate_content.return_value = mock_resp

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
            use_interactions_api=False,
        )

        resp = provider.generate_text("List competencies")
        assert resp.content == "Models API Output"
        assert resp.provider_name == "gemini"
        assert resp.model_name == "gemini-3.8-flash"
        assert resp.finish_reason == "stop"
        assert resp.usage.total_tokens == 35

        # Check call arguments
        mock_client.models.generate_content.assert_called_once()
        _, call_kwargs = mock_client.models.generate_content.call_args
        gen_config = call_kwargs["config"]
        # Verify milliseconds conversion in HttpOptions
        assert gen_config.http_options.timeout == 30000

    def test_models_api_empty_candidates_rejected(self):
        mock_client = MagicMock()
        mock_resp = _make_mock_response(empty_candidates=True)
        mock_client.models.generate_content.return_value = mock_resp

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
            use_interactions_api=False,
        )

        with pytest.raises(LLMResponseError, match="empty candidate list"):
            provider.generate_text("Test prompt")

    def test_models_api_prompt_blocked_by_safety(self):
        mock_client = MagicMock()
        mock_resp = _make_mock_response(empty_candidates=True, prompt_blocked=True)
        mock_client.models.generate_content.return_value = mock_resp

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
            use_interactions_api=False,
        )

        with pytest.raises(LLMResponseError, match="SAFETY_PROMPT_BLOCKED"):
            provider.generate_text("Test prompt")

    def test_models_api_candidate_blocked_by_safety_finish_reason(self):
        mock_client = MagicMock()
        mock_resp = _make_mock_response(text="", finish_reason="SAFETY")
        mock_client.models.generate_content.return_value = mock_resp

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
            use_interactions_api=False,
        )

        with pytest.raises(LLMResponseError, match="content policy"):
            provider.generate_text("Harmful content")


class TestGeminiProviderErrorMappingAndRedaction:
    """Verifies mapping of vendor SDK exceptions to project hierarchy and secret redaction."""

    def _make_api_error(self, code: int, message: str) -> errors.APIError:
        return errors.APIError(code, {"error": {"message": message, "code": code}})

    def test_authentication_error_mapping(self):
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = self._make_api_error(
            401, "API key not valid. Please pass a valid API key: AIzaSySecret1234567890abcdef"
        )

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_looking_key",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMAuthenticationError) as exc_info:
            provider.generate_text("Hello")

        err_msg = str(exc_info.value)
        assert "AIzaSy" not in err_msg
        assert "[REDACTED_API_KEY]" in err_msg
        assert "[gemini]" in err_msg

    def test_rate_limit_error_mapping(self):
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = self._make_api_error(
            429, "Resource exhausted. Quota exceeded for model gemini-3.8-flash"
        )

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMRateLimitError):
            provider.generate_text("Hello")

    def test_timeout_error_mapping(self):
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = self._make_api_error(
            504, "Gateway Timeout deadline exceeded."
        )

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMTimeoutError):
            provider.generate_text("Hello")

    def test_httpx_timeout_mapping(self):
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = httpx.ReadTimeout("Read timed out")

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMTimeoutError, match="timed out"):
            provider.generate_text("Hello")

    def test_invalid_parameter_bad_request_mapping(self):
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = self._make_api_error(
            400, "Bad Request: Invalid model argument."
        )

        provider = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="valid_key",
            ),
            client=mock_client,
        )

        with pytest.raises(LLMConfigurationError):
            provider.generate_text("Hello")


class TestGeminiProviderManagerIntegration:
    """Verifies manager registration and dispatching with GeminiProvider."""

    def test_manager_dispatches_to_gemini_interactions(self):
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = _make_mock_interaction("Managed Gemini Output")

        gemini = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="test_key",
            ),
            client=mock_client,
        )

        manager = LLMProviderManager()
        manager.register_provider(gemini, set_active=True)

        assert manager.active_provider_name == "gemini"
        resp = manager.generate_text("Test manager dispatch")
        assert resp.content == "Managed Gemini Output"
        assert resp.provider_name == "gemini"

    def test_manager_status_contains_gemini_diagnostics(self):
        gemini = GeminiProvider(
            config=LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-3.8-flash",
                api_key="AIzaSy1234567890abcdef1234567890",
            )
        )
        manager = LLMProviderManager()
        manager.register_provider(gemini)

        status = manager.get_status()
        assert "gemini" in status["registered_providers"]
        assert status["providers"]["gemini"]["model_name"] == "gemini-3.8-flash"
        assert status["providers"]["gemini"]["is_available"] is True
        assert status["providers"]["gemini"]["is_api_key_configured"] is True
