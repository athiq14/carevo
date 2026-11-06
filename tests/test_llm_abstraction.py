"""
Unit tests for the LLM Provider Abstraction Architecture (Phase 3.1).
Validates abstract base class contracts, data models, exception hierarchy, secret redaction,
MockLLMProvider determinism, LLMProviderManager lifecycle, and air-gapped zero network calls.
"""
import pytest
import socket
from pydantic import ValidationError

from core.llm import (
    BaseLLMProvider,
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
    LLMUsageMetadata,
    MessageRole,
    MockLLMProvider,
    redact_secrets,
)


class TestBaseLLMProviderContract:
    """Verifies that the abstract base class cannot be instantiated and enforces interface."""

    def test_cannot_instantiate_abstract_base_class(self):
        with pytest.raises(TypeError) as exc_info:
            BaseLLMProvider()  # type: ignore[abstract]
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class IncompleteProvider(BaseLLMProvider):
            @property
            def provider_name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore[abstract]


class TestLLMDataModels:
    """Tests Pydantic validation and serialization behavior for LLM models."""

    def test_message_roles(self):
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"

    def test_llm_message_validation(self):
        msg = LLMMessage(role=MessageRole.USER, content="Hello AI")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello AI"

        # Content must not be empty
        with pytest.raises(ValidationError):
            LLMMessage(role=MessageRole.USER, content="")

        # Role must be valid
        with pytest.raises(ValidationError):
            LLMMessage(role="invalid_role", content="Hello")  # type: ignore[arg-type]

    def test_llm_generation_config_bounds(self):
        default_cfg = LLMGenerationConfig()
        assert default_cfg.temperature == 0.7
        assert default_cfg.timeout == 30.0

        # Valid bounds
        valid_cfg = LLMGenerationConfig(temperature=0.0, top_p=0.9, timeout=10.0, max_output_tokens=500)
        assert valid_cfg.temperature == 0.0
        assert valid_cfg.max_output_tokens == 500

        # Invalid temperature (< 0.0 or > 2.0)
        with pytest.raises(ValidationError):
            LLMGenerationConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            LLMGenerationConfig(temperature=2.5)

        # Invalid top_p (< 0.0 or > 1.0)
        with pytest.raises(ValidationError):
            LLMGenerationConfig(top_p=1.5)

        # Invalid timeout (<= 0.0)
        with pytest.raises(ValidationError):
            LLMGenerationConfig(timeout=0.0)

        # Invalid max_output_tokens (must be >= 1)
        with pytest.raises(ValidationError):
            LLMGenerationConfig(max_output_tokens=0)

        with pytest.raises(ValidationError):
            LLMGenerationConfig(max_output_tokens=-10)

    def test_llm_usage_metadata(self):
        usage = LLMUsageMetadata(prompt_tokens=15, completion_tokens=25, total_tokens=40)
        assert usage.prompt_tokens == 15
        assert usage.completion_tokens == 25
        assert usage.total_tokens == 40

        # Usage metadata permits None (when provider omits accounting)
        empty_usage = LLMUsageMetadata()
        assert empty_usage.prompt_tokens is None
        assert empty_usage.total_tokens is None

        # Negative token count rejected
        with pytest.raises(ValidationError):
            LLMUsageMetadata(prompt_tokens=-1)

    def test_llm_request_construction_and_factory(self):
        msg = LLMMessage(role=MessageRole.USER, content="Summarize resume")
        req = LLMRequest(messages=[msg])
        assert len(req.messages) == 1
        assert req.generation_config.temperature == 0.7

        # from_prompt factory with user prompt only
        req_factory = LLMRequest.from_prompt("Analyze this career profile")
        assert len(req_factory.messages) == 1
        assert req_factory.messages[0].role == MessageRole.USER
        assert req_factory.messages[0].content == "Analyze this career profile"

        # from_prompt factory with system prompt
        req_sys = LLMRequest.from_prompt(
            prompt="Analyze this profile",
            system_prompt="You are a career strategist.",
        )
        assert len(req_sys.messages) == 2
        assert req_sys.messages[0].role == MessageRole.SYSTEM
        assert req_sys.messages[0].content == "You are a career strategist."
        assert req_sys.messages[1].role == MessageRole.USER

        # from_prompt factory with max_output_tokens and temperature
        req_custom = LLMRequest.from_prompt(
            prompt="Analyze this profile",
            max_output_tokens=256,
            temperature=0.2,
        )
        assert req_custom.generation_config.max_output_tokens == 256
        assert req_custom.generation_config.temperature == 0.2

        # Empty or whitespace prompt rejected
        with pytest.raises(ValueError, match="Prompt text must contain non-whitespace"):
            LLMRequest.from_prompt("   ")

    def test_llm_response_structure(self):
        resp = LLMResponse(
            content="Profile evaluated successfully.",
            provider_name="mock",
            model_name="mock-model",
            usage=LLMUsageMetadata(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            finish_reason="stop",
            raw_metadata={"latency_ms": 12},
        )
        assert resp.content == "Profile evaluated successfully."
        assert resp.provider_name == "mock"
        assert resp.usage.total_tokens == 15
        assert resp.raw_metadata["latency_ms"] == 12

    def test_provider_config_secret_masking_and_serialization(self):
        real_key = "AIzaSyABC1234567890XYZabcdefghijklmnop"
        config = LLMProviderConfig(
            provider_name="gemini",
            model_name="gemini-1.5-pro",
            api_key=real_key,
            max_output_tokens=1024,
        )

        assert config.is_api_key_configured is True
        assert config.api_key == real_key  # Direct programmatic access for SDKs
        assert config.max_output_tokens == 1024

        repr_str = repr(config)
        str_str = str(config)

        # 1. Raw key must NOT be leaked in representations
        assert real_key not in repr_str
        assert real_key not in str_str
        assert "[CONFIGURED]" in repr_str
        assert "[CONFIGURED]" in str_str
        assert "max_output_tokens=1024" in repr_str

        # 2. Raw key must NOT leak through model_dump() serialization
        dumped_dict = config.model_dump()
        assert dumped_dict["api_key"] == "[CONFIGURED]"
        assert real_key not in str(dumped_dict)

        # 3. Raw key must NOT leak through model_dump_json() serialization
        dumped_json = config.model_dump_json()
        assert real_key not in dumped_json
        assert '"api_key":"[CONFIGURED]"' in dumped_json

        # Unconfigured key
        config_no_key = LLMProviderConfig(provider_name="mock", model_name="mock-v1")
        assert config_no_key.is_api_key_configured is False
        assert "[NOT_CONFIGURED]" in repr(config_no_key)
        assert config_no_key.model_dump()["api_key"] is None

        # max_output_tokens validation in LLMProviderConfig
        with pytest.raises(ValidationError):
            LLMProviderConfig(provider_name="mock", model_name="m", max_output_tokens=0)


class TestExceptionHierarchyAndRedaction:
    """Validates structured exceptions and secret redaction."""

    def test_exception_inheritance(self):
        assert issubclass(LLMConfigurationError, LLMProviderError)
        assert issubclass(LLMAuthenticationError, LLMProviderError)
        assert issubclass(LLMRateLimitError, LLMProviderError)
        assert issubclass(LLMTimeoutError, LLMProviderError)
        assert issubclass(LLMResponseError, LLMProviderError)

    def test_secret_redaction_in_exceptions(self):
        raw_msg = (
            "Authentication failed using key AIzaSyD9481jfd_8314nd8374981hd74jd827 "
            "and groq key gsk_abc1234567890abcdef1234567890"
        )
        err = LLMAuthenticationError(raw_msg, provider_name="gemini")

        err_str = str(err)
        assert "AIzaSy" not in err_str
        assert "gsk_" not in err_str
        assert "[REDACTED_API_KEY]" in err_str
        assert "[gemini]" in err_str

    def test_redact_google_api_key(self):
        google_msg = "Google GenAI API call failed with key AIzaSyA1b2C3d4E5f6G7h8I9j0KlMnOpQrStUvWx"
        sanitized = redact_secrets(google_msg)
        assert "AIzaSy" not in sanitized
        assert "[REDACTED_API_KEY]" in sanitized

    def test_redact_groq_api_key(self):
        groq_msg = "Groq cloud returned 401 using key gsk_y783hsd823nd74hd784hdb78423hdb7834"
        sanitized = redact_secrets(groq_msg)
        assert "gsk_" not in sanitized
        assert "[REDACTED_API_KEY]" in sanitized

    def test_redact_generic_api_key(self):
        generic_msg = "Standard token error with sk-1234567890abcdef1234567890abcdef"
        sanitized = redact_secrets(generic_msg)
        assert "sk-12345" not in sanitized
        assert "[REDACTED_API_KEY]" in sanitized

    def test_redact_secrets_utility(self):
        assert redact_secrets("") == ""
        openai_msg = "Error with sk-abcdef123456789012345678901234"
        assert "sk-abcdef" not in redact_secrets(openai_msg)
        assert "[REDACTED_API_KEY]" in redact_secrets(openai_msg)


class TestMockLLMProvider:
    """Verifies the mock provider's behavior, offline execution, and error simulation."""

    def test_mock_provider_defaults(self):
        mock = MockLLMProvider()
        assert mock.provider_name == "mock"
        assert mock.model_name == "mock-career-v1"
        assert mock.is_available() is True
        mock.validate_configuration()  # Should not raise

    def test_mock_provider_deterministic_generation(self):
        mock = MockLLMProvider()
        req = LLMRequest.from_prompt("Extract key competencies from resume")
        resp = mock.generate(req)

        assert isinstance(resp, LLMResponse)
        assert resp.provider_name == "mock"
        assert resp.model_name == "mock-career-v1"
        assert "[Mock Response from mock-career-v1]" in resp.content
        assert resp.usage is not None
        assert resp.usage.total_tokens > 0
        assert len(mock.call_history) == 1

    def test_mock_provider_canned_response(self):
        mock = MockLLMProvider(canned_response="Standard mock analysis output.")
        resp = mock.generate_text("Any prompt")

        assert resp.content == "Standard mock analysis output."

    def test_mock_provider_error_simulation(self):
        mock = MockLLMProvider(
            simulate_error=LLMRateLimitError("Quota exceeded 429", provider_name="mock")
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            mock.generate_text("Should fail")

        assert "Quota exceeded 429" in str(exc_info.value)

    def test_mock_provider_max_output_tokens_truncation(self):
        mock = MockLLMProvider()
        # Request with max_output_tokens=3
        resp = mock.generate_text("Generate long detailed text", max_output_tokens=3)
        assert resp.finish_reason == "length"
        assert len(resp.content.split()) == 3
        assert resp.usage.completion_tokens == 3

        # Request with large max_output_tokens does not truncate
        resp_large = mock.generate_text("Short prompt", max_output_tokens=100)
        assert resp_large.finish_reason == "stop"

    def test_mock_provider_zero_network_calls(self, monkeypatch):
        """Proves MockLLMProvider makes zero socket network connections."""
        import http.client

        def block_connect(*args, **kwargs):
            raise AssertionError("Network connection attempted during MockLLMProvider execution!")

        monkeypatch.setattr(socket.socket, "connect", block_connect)
        monkeypatch.setattr(socket.socket, "connect_ex", block_connect)
        monkeypatch.setattr(http.client.HTTPConnection, "connect", block_connect)

        mock = MockLLMProvider()
        resp = mock.generate_text("Verify offline guarantee", max_output_tokens=50)
        assert resp is not None
        assert "[Mock Response" in resp.content


class TestLLMProviderManager:
    """Verifies manager registration, activation, status reporting, and dispatching."""

    def test_manager_registration_and_activation(self):
        mgr = LLMProviderManager()
        assert mgr.active_provider_name is None
        assert mgr.list_providers() == []

        mock_1 = MockLLMProvider()
        mgr.register_provider(mock_1)

        # First registered provider becomes active automatically
        assert mgr.active_provider_name == "mock"
        assert mgr.list_providers() == ["mock"]
        assert mgr.get_provider() == mock_1

        # Register second mock with custom name
        cfg_2 = LLMProviderConfig(provider_name="mock-2", model_name="mock-model-2")
        mock_2 = MockLLMProvider(config=cfg_2)
        mgr.register_provider(mock_2, set_active=True)

        assert mgr.active_provider_name == "mock-2"
        assert mgr.get_provider() == mock_2
        assert mgr.get_provider("mock") == mock_1

    def test_manager_switching_active_provider(self):
        mgr = LLMProviderManager()
        mock_a = MockLLMProvider(config=LLMProviderConfig(provider_name="alpha", model_name="m-a"))
        mock_b = MockLLMProvider(config=LLMProviderConfig(provider_name="beta", model_name="m-b"))

        mgr.register_provider(mock_a)
        mgr.register_provider(mock_b)

        assert mgr.active_provider_name == "alpha"
        mgr.set_active_provider("beta")
        assert mgr.active_provider_name == "beta"

        with pytest.raises(LLMProviderError, match="not registered"):
            mgr.set_active_provider("nonexistent")

    def test_manager_unregistered_provider_access(self):
        mgr = LLMProviderManager()
        with pytest.raises(LLMProviderError, match="No active LLM provider"):
            mgr.get_provider()

        with pytest.raises(LLMProviderError, match="not registered"):
            mgr.get_provider("unknown_provider")

    def test_manager_diagnostics_status(self):
        mgr = LLMProviderManager()
        mock = MockLLMProvider()
        mgr.register_provider(mock)

        status = mgr.get_status()
        assert status["active_provider"] == "mock"
        assert status["provider_count"] == 1
        assert "mock" in status["registered_providers"]
        assert status["providers"]["mock"]["model_name"] == "mock-career-v1"
        assert status["providers"]["mock"]["is_available"] is True
        assert status["providers"]["mock"]["is_api_key_configured"] is False

    def test_manager_dispatch_generate(self):
        mgr = LLMProviderManager()
        mock = MockLLMProvider(canned_response="Manager dispatch verified.")
        mgr.register_provider(mock)

        resp = mgr.generate_text("Prompt to manager")
        assert resp.content == "Manager dispatch verified."

    def test_manager_dispatch_with_max_output_tokens(self):
        mgr = LLMProviderManager()
        mock = MockLLMProvider()
        mgr.register_provider(mock)

        resp = mgr.generate_text("Test manager token limits", max_output_tokens=4)
        assert resp.finish_reason == "length"
        assert len(resp.content.split()) == 4

