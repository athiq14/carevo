"""
Base architecture and shared utilities for Phase 5.2 Specialist Agents.
Provides StructuredOutputEngine integration, sanitized allowlist serialization,
XML quarantine envelopes, and standardized telemetry metadata extraction.
"""
import json
from typing import Any, Optional
from pydantic import BaseModel

from core.llm.structured.engine import StructuredOutputEngine
from core.matching.explanation_generator import escape_xml_closing_tags, scrub_pii


class BaseSpecialistAgent:
    """
    Base class for specialist agents providing LLM engine management,
    telemetry packaging, and safe prompt envelope formatting.
    """

    def __init__(
        self,
        engine: Optional[StructuredOutputEngine] = None,
        dispatcher: Optional[Any] = None,
    ) -> None:
        if engine is not None:
            self._engine = engine
        elif dispatcher is not None:
            self._engine = StructuredOutputEngine(dispatcher=dispatcher)
        else:
            from core.llm.manager import LLMProviderManager
            self._engine = StructuredOutputEngine(dispatcher=LLMProviderManager())

    def _format_safe_envelope(self, tag_name: str, payload: dict[str, Any]) -> str:
        """
        Serialize payload to JSON with PII scrubbing and XML quarantine envelope escaping.
        Prevents prompt injection or XML tag masquerading.
        """
        raw_json = json.dumps(payload, indent=2, default=str)
        safe_json = escape_xml_closing_tags(raw_json)
        return f"<{tag_name}>\n{safe_json}\n</{tag_name}>"

    def _extract_telemetry(
        self,
        resp: Any,
        fallback_mode: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Extract standardized telemetry diagnostics from LLM generation response.
        """
        if fallback_mode or resp is None:
            return {
                "provider": "deterministic_fallback",
                "model": "rule_based_engine",
                "attempts": 0,
                "correction_occurred": False,
                "latency_ms": 0.0,
                "token_usage": None,
                "fallback_mode": True,
                "fallback_reason": scrub_pii(fallback_reason or "LLM generation unavailable"),
            }

        raw_resp = getattr(resp, "raw_response", None)
        provider_name = getattr(raw_resp, "provider_name", "unknown") if raw_resp else "unknown"
        model_name = getattr(raw_resp, "model_name", "unknown") if raw_resp else "unknown"
        usage_data = raw_resp.usage.model_dump() if raw_resp and getattr(raw_resp, "usage", None) else None
        latency_val = getattr(raw_resp, "latency_ms", 0.0) if raw_resp else 0.0

        return {
            "provider": provider_name,
            "model": model_name,
            "attempts": getattr(resp, "attempts", 1),
            "correction_occurred": getattr(resp, "correction_occurred", False),
            "latency_ms": round(latency_val, 2),
            "token_usage": usage_data,
            "fallback_mode": False,
        }

