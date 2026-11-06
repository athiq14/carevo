"""
Abstract base class definition for all Large Language Model (LLM) providers.
Defines a standard, uniform interface that all concrete providers (Mock, Gemini, Groq) must fulfill.
"""
from abc import ABC, abstractmethod
from typing import Optional

from core.llm.models import (
    LLMGenerationConfig,
    LLMProviderConfig,
    LLMRequest,
    LLMResponse,
)


class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    
    Guarantees that downstream services (analysis, generation, coaching)
    interact exclusively with this contract rather than vendor-specific SDKs.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Canonical name identifying this provider (e.g. 'mock', 'gemini', 'groq')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the active model deployed on this provider."""
        pass

    @property
    @abstractmethod
    def config(self) -> LLMProviderConfig:
        """Configuration instance governing this provider's runtime parameters."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is currently available and ready to accept requests.
        
        Returns:
            bool: True if required credentials and network prerequisites are satisfied.
        """
        pass

    @abstractmethod
    def validate_configuration(self) -> None:
        """
        Validate provider configuration and credentials.
        
        Raises:
            LLMConfigurationError: If configuration options are malformed or invalid.
            LLMAuthenticationError: If required credentials/API keys are missing or invalid.
        """
        pass

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Execute text generation for the supplied multi-message LLMRequest.
        
        Args:
            request: Validated LLMRequest containing conversation history and generation config.
            
        Returns:
            LLMResponse: Provider-agnostic response payload with content and metadata.
            
        Raises:
            LLMProviderError: On generation, network, rate-limit, or authentication failure.
        """
        pass

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Convenience helper to generate a response from a simple text prompt.
        
        Args:
            prompt: User prompt content.
            system_prompt: Optional system guidance instructions.
            generation_config: Optional override generation config.
            max_output_tokens: Optional maximum output tokens allowed in response.
            temperature: Optional sampling temperature override.
            
        Returns:
            LLMResponse: The generation response.
        """
        request = LLMRequest.from_prompt(
            prompt=prompt,
            system_prompt=system_prompt,
            generation_config=generation_config,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        return self.generate(request)

    def generate_structured(
        self,
        request: Any,
        schema_class: Optional[Any] = None,
        max_attempts: int = 2,
    ) -> Any:
        """
        Execute structured output generation directly against this provider.
        """
        from core.llm.structured.engine import StructuredOutputEngine
        engine = StructuredOutputEngine(dispatcher=self)
        return engine.generate(
            request=request,
            schema_class=schema_class,
            max_attempts=max_attempts,
            provider_name=self.provider_name,
        )

