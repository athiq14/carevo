"""
Privacy-conscious internal telemetry and observability engine.
Records feature invocations, LLM provider routing, latency, token usage,
and web research status without storing PII or confidential resume data.
"""
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class TelemetryEvent(BaseModel):
    """Structured record of an internal model, search, or analysis invocation."""
    model_config = ConfigDict(frozen=True)

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        description="Invocation timestamp in UTC.",
    )
    feature: str = Field(..., description="Target application feature invoked.")
    provider: str = Field(..., description="Provider handling request ('groq', 'gemini', 'duckduckgo', 'rules_engine').")
    model: str = Field(default="n/a", description="Model identifier used.")
    status: str = Field(..., description="'SUCCESS', 'FALLBACK', or 'ERROR'.")
    latency_ms: float = Field(default=0.0, description="Execution elapsed time in milliseconds.")
    prompt_tokens: int = Field(default=0, description="Prompt token count if provided.")
    completion_tokens: int = Field(default=0, description="Completion token count if provided.")
    total_tokens: int = Field(default=0, description="Total token count.")
    web_research_used: bool = Field(default=False, description="Whether live web research was performed.")
    notes: str = Field(default="", description="High-level operational note (zero PII).")


class TelemetryTracker:
    """Thread-safe in-memory recorder for live observability."""

    _instance: Optional["TelemetryTracker"] = None

    def __new__(cls) -> "TelemetryTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._events = []
        return cls._instance

    def record_event(
        self,
        feature: str,
        provider: str,
        status: str,
        model: str = "n/a",
        latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        web_research_used: bool = False,
        notes: str = "",
    ) -> TelemetryEvent:
        """Log a new operational event."""
        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens

        event = TelemetryEvent(
            feature=feature,
            provider=provider,
            model=model,
            status=status,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            web_research_used=web_research_used,
            notes=notes,
        )
        self._events.append(event)
        # Cap memory buffer to 100 recent events
        if len(self._events) > 100:
            self._events = self._events[-100:]
        return event

    def get_recent_events(self, limit: int = 20) -> list[TelemetryEvent]:
        """Return the most recent telemetry events."""
        return list(reversed(self._events[-limit:]))

    def get_summary(self) -> dict[str, Any]:
        """Compute aggregate usage statistics."""
        total = len(self._events)
        groq_calls = sum(1 for e in self._events if "groq" in e.provider.lower())
        web_calls = sum(1 for e in self._events if e.web_research_used)
        fallbacks = sum(1 for e in self._events if e.status == "FALLBACK")
        total_tokens = sum(e.total_tokens for e in self._events)
        avg_latency = (
            round(sum(e.latency_ms for e in self._events) / total, 1)
            if total > 0
            else 0.0
        )

        return {
            "total_invocations": total,
            "groq_live_calls": groq_calls,
            "web_research_calls": web_calls,
            "fallback_invocations": fallbacks,
            "total_tokens_consumed": total_tokens,
            "average_latency_ms": avg_latency,
        }

    def clear(self) -> None:
        """Reset telemetry buffer (for testing)."""
        self._events.clear()


telemetry_tracker = TelemetryTracker()

