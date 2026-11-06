"""
Provider manager coordinating LLM lifecycle, registration, request dispatch,
and intelligent cross-provider fallback.
Enforces explicit error reporting, privacy controls, and two-stage fallback.
"""
from typing import Any, Optional

from core.llm.base import BaseLLMProvider
from core.llm.exceptions import (
    LLMFallbackError,
    LLMProviderError,
    is_transient_error,
    redact_secrets,
)
from core.llm.models import (
    LLMGenerationConfig,
    LLMRequest,
    LLMResponse,
)


class LLMProviderManager:
    """
    Central manager for LLM provider registry, active provider selection,
    request dispatching, and intelligent two-stage fallback.

    Provider Priority:
        Primary: Groq (openai/gpt-oss-120b)
        Secondary: Google Gemini (configured GEMINI_MODEL)
        Development: MockLLMProvider (offline development & zero-network testing)

    Fallback Architecture:
        - Maximum TWO attempts total: 1 primary attempt + 1 secondary attempt.
        - Zero retry loops: no infinite loops, retries, or cascading chain cycles.
        - Transient failure qualification: Primary errors are inspected via `is_transient_error`.
          Only transient failures (429 rate limit, 503 unavailable, 502/504 gateway,
          timeouts, connection failures) are eligible for fallback.
        - Permanent failure short-circuit: Authentication (401/403), Bad Request (400),
          Model Not Found (404), safety blocks, and malformed responses fail immediately
          without attempting secondary fallback.
        - Privacy & sensitive data gate: Sensitive candidate requests (is_sensitive=True)
          default to cross-provider fallback DISABLED unless explicitly authorized by
          setting `allow_cross_provider_fallback=True`.
        - Development isolation: MockLLMProvider is strictly isolated for offline dev/testing
          and is never utilized as a fallback provider in production.
    """

    def __init__(
        self,
        primary_provider: str = "groq",
        secondary_provider: Optional[str] = "gemini",
        fallback_enabled: bool = True,
    ) -> None:
        """
        Initialize LLMProviderManager.

        Args:
            primary_provider: Default primary provider identifier (default: "groq").
            secondary_provider: Fallback secondary provider identifier (default: "gemini").
            fallback_enabled: Master flag enabling or disabling intelligent fallback.
        """
        self._providers: dict[str, BaseLLMProvider] = {}
        self._active_provider: Optional[str] = None
        self._primary_provider: str = primary_provider.lower().strip()
        self._secondary_provider: Optional[str] = (
            secondary_provider.lower().strip() if secondary_provider else None
        )
        self._fallback_enabled: bool = fallback_enabled

    @property
    def active_provider_name(self) -> Optional[str]:
        """Name of the currently designated active/default provider, or None if none registered."""
        return self._active_provider

    @property
    def primary_provider_name(self) -> str:
        """Name of the designated primary provider in the fallback chain."""
        return self._primary_provider

    @property
    def secondary_provider_name(self) -> Optional[str]:
        """Name of the designated secondary provider in the fallback chain."""
        return self._secondary_provider

    @property
    def is_fallback_enabled(self) -> bool:
        """Indicate whether cross-provider fallback is active on this manager."""
        return self._fallback_enabled

    def set_fallback_chain(
        self,
        primary: str,
        secondary: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """
        Configure the fallback chain sequence and enablement.

        Args:
            primary: Canonical identifier of primary provider.
            secondary: Canonical identifier of secondary provider (or None to disable fallback).
            enabled: Whether fallback is enabled.
        """
        self._primary_provider = primary.lower().strip()
        self._secondary_provider = secondary.lower().strip() if secondary else None
        self._fallback_enabled = enabled

    def set_fallback_enabled(self, enabled: bool) -> None:
        """Toggle fallback strategy on or off."""
        self._fallback_enabled = enabled

    def register_provider(self, provider: BaseLLMProvider, set_active: bool = False) -> None:
        """
        Register a provider instance with the manager.

        Args:
            provider: BaseLLMProvider instance.
            set_active: If True, or if this is the first registered provider,
                        sets it as the active provider.
        """
        key = provider.provider_name.lower().strip()
        self._providers[key] = provider

        if set_active or self._active_provider is None:
            self._active_provider = key

    def get_provider(self, name: Optional[str] = None) -> BaseLLMProvider:
        """
        Retrieve a registered provider by name, or return the active provider if name is None.

        Args:
            name: Optional provider name. If None, active provider is returned.

        Returns:
            BaseLLMProvider: The registered provider instance.

        Raises:
            LLMProviderError: If no active provider exists or if requested provider is unregistered.
        """
        if name is None:
            if self._active_provider is None:
                raise LLMProviderError("No active LLM provider configured.")
            target = self._active_provider
        else:
            target = name.lower().strip()

        if target not in self._providers:
            available = list(self._providers.keys())
            raise LLMProviderError(
                f"Provider '{target}' is not registered. Registered providers: {available}"
            )

        return self._providers[target]

    def set_active_provider(self, name: str) -> None:
        """
        Set the active provider for subsequent requests.

        Args:
            name: Name of a registered provider.

        Raises:
            LLMProviderError: If the provider is not registered.
        """
        target = name.lower().strip()
        if target not in self._providers:
            available = list(self._providers.keys())
            raise LLMProviderError(
                f"Cannot activate '{name}'. Provider is not registered. Registered: {available}"
            )
        self._active_provider = target

    def list_providers(self) -> list[str]:
        """Return a list of all registered provider names."""
        return list(self._providers.keys())

    def get_status(self) -> dict[str, Any]:
        """
        Compile diagnostics on all registered providers and fallback strategy
        without exposing sensitive secrets.

        Returns:
            Dictionary containing active provider, fallback chain configuration,
            and individual provider readiness statuses.
        """
        primary_avail = (
            self._providers[self._primary_provider].is_available()
            if self._primary_provider in self._providers
            else False
        )
        secondary_avail = (
            self._providers[self._secondary_provider].is_available()
            if self._secondary_provider and self._secondary_provider in self._providers
            else False
        )

        if not self._fallback_enabled:
            fallback_status = "Disabled"
        elif primary_avail and secondary_avail:
            fallback_status = "Active"
        elif primary_avail or secondary_avail:
            fallback_status = "Degraded"
        else:
            fallback_status = "Unavailable"

        return {
            "active_provider": self._active_provider,
            "primary_provider": self._primary_provider,
            "secondary_provider": self._secondary_provider,
            "fallback_enabled": self._fallback_enabled,
            "fallback_status": fallback_status,
            "registered_providers": list(self._providers.keys()),
            "provider_count": len(self._providers),
            "providers": {
                name: {
                    "provider_name": prov.provider_name,
                    "model_name": prov.model_name,
                    "is_available": prov.is_available(),
                    "is_api_key_configured": prov.config.is_api_key_configured,
                }
                for name, prov in self._providers.items()
            },
        }

    def generate(self, request: LLMRequest, provider_name: Optional[str] = None) -> LLMResponse:
        """
        Dispatch generation request according to intelligent fallback policy.

        Execution Rules:
        1. Explicit Provider: If `provider_name` is explicitly provided, dispatch directly
           to that provider with ZERO fallback.
        2. Offline Development Mode: If the active provider is explicitly set to 'mock',
           dispatch directly to MockLLMProvider with zero network and no fallback.
        3. Attempt 1 (Primary): Dispatch to primary provider (default: 'groq').
           - On success: return response immediately.
        4. Fallback Qualification: If primary encounters an error:
           - Verify fallback is enabled on manager.
           - Verify secondary provider is configured, registered, and distinct from primary.
           - Verify primary error is transient via `is_transient_error`.
           - Verify request permits fallback via `request.is_fallback_permitted()`.
           - If ANY condition fails: re-raise the primary error immediately without fallback.
        5. Attempt 2 (Secondary): Dispatch exactly ONE attempt to secondary provider (Gemini).
           - On success: enrich metadata with fallback details (`fallback_occurred=True`,
             `primary_provider`, `primary_error`, `secondary_provider`) and return response.
           - On failure: raise `LLMFallbackError` preserving both primary and secondary failure traces.

        Args:
            request: The validated LLMRequest to execute.
            provider_name: Specific provider name to route to, or None to use intelligent fallback.

        Returns:
            LLMResponse from primary or fallback secondary provider.

        Raises:
            LLMProviderError: On permanent failure, direct route error, or fallback failure.
            LLMFallbackError: When both primary and secondary providers fail in sequence.
        """
        # Case 1: Caller explicitly requested a specific provider -> direct dispatch (no fallback)
        if provider_name is not None:
            provider = self.get_provider(provider_name)
            return provider.generate(request)

        # Case 2: Active provider is explicitly 'mock' -> offline dev/test execution
        if self._active_provider == "mock":
            return self.get_provider("mock").generate(request)

        # Case 3: Resolve primary provider
        primary_name = (
            self._primary_provider
            if self._primary_provider in self._providers
            else self._active_provider
        )
        if primary_name is None or primary_name not in self._providers:
            raise LLMProviderError("No active or primary LLM provider registered.")

        # If primary resolves to mock, execute directly without fallback
        if primary_name == "mock":
            return self.get_provider("mock").generate(request)

        primary_provider = self.get_provider(primary_name)

        # Attempt 1: Execute primary provider
        import time
        start_t = time.time()
        try:
            resp = primary_provider.generate(request)
            lat = (time.time() - start_t) * 1000
            toks = getattr(resp, "usage", None)
            p_toks = getattr(toks, "prompt_tokens", 0) if toks else 0
            c_toks = getattr(toks, "completion_tokens", 0) if toks else 0
            t_toks = getattr(toks, "total_tokens", 0) if toks else 0
            try:
                from core.telemetry import telemetry_tracker
                telemetry_tracker.record_event(
                    feature="LLM Dispatcher",
                    provider=primary_name,
                    model=primary_provider.model_name,
                    status="SUCCESS",
                    latency_ms=lat,
                    prompt_tokens=p_toks,
                    completion_tokens=c_toks,
                    total_tokens=t_toks,
                    notes="Dispatched via primary provider",
                )
            except Exception:
                pass
            return resp
        except Exception as primary_exc:
            # Evaluate fallback qualification
            can_fallback = (
                self._fallback_enabled
                and self._secondary_provider is not None
                and self._secondary_provider in self._providers
                and self._secondary_provider != primary_name
                and is_transient_error(primary_exc)
                and request.is_fallback_permitted()
            )

            if not can_fallback:
                # Permanent failure, fallback disabled, sensitive data unpermitted, or no secondary -> re-raise
                raise primary_exc

            # Attempt 2: Exactly ONE attempt with secondary provider
            secondary_provider = self.get_provider(self._secondary_provider)
            try:
                sec_start = time.time()
                secondary_response = secondary_provider.generate(request)
                lat_sec = (time.time() - sec_start) * 1000

                # Enrich metadata with transparent fallback audit trail
                raw_meta = dict(secondary_response.raw_metadata or {})
                raw_meta["fallback_occurred"] = True
                raw_meta["primary_provider"] = primary_name
                raw_meta["primary_error"] = redact_secrets(str(primary_exc))
                raw_meta["secondary_provider"] = self._secondary_provider

                try:
                    from core.telemetry import telemetry_tracker
                    telemetry_tracker.record_event(
                        feature="LLM Dispatcher",
                        provider=self._secondary_provider,
                        model=secondary_provider.model_name,
                        status="FALLBACK",
                        latency_ms=lat_sec,
                        notes=f"Fallback from {primary_name}: {type(primary_exc).__name__}",
                    )
                except Exception:
                    pass

                return secondary_response.model_copy(update={"raw_metadata": raw_meta})
            except Exception as secondary_exc:
                # Both primary and secondary failed -> raise composite LLMFallbackError
                raise LLMFallbackError(
                    primary_provider=primary_name,
                    primary_error=primary_exc,
                    secondary_provider=self._secondary_provider,
                    secondary_error=secondary_exc,
                ) from secondary_exc

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
        provider_name: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        allow_fallback: bool = True,
        is_sensitive: bool = False,
        allow_cross_provider_fallback: bool = False,
    ) -> LLMResponse:
        """
        Convenience helper dispatching simple text generation through the fallback chain.
        """
        request = LLMRequest.from_prompt(
            prompt=prompt,
            system_prompt=system_prompt,
            generation_config=generation_config,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            allow_fallback=allow_fallback,
            is_sensitive=is_sensitive,
            allow_cross_provider_fallback=allow_cross_provider_fallback,
        )
        return self.generate(request, provider_name=provider_name)

    def generate_structured(
        self,
        request: Any,
        schema_class: Optional[Any] = None,
        max_attempts: int = 2,
        provider_name: Optional[str] = None,
    ) -> Any:
        """
        Execute structured generation against a target Pydantic schema using the configured
        provider chain and bounded error-correction retries.
        """
        from core.llm.structured.engine import StructuredOutputEngine
        engine = StructuredOutputEngine(dispatcher=self)
        return engine.generate(
            request=request,
            schema_class=schema_class,
            max_attempts=max_attempts,
            provider_name=provider_name,
        )
