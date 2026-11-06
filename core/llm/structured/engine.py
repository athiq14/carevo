"""
Provider-independent structured generation engine.
Coordinates schema injection, LLM generation, JSON extraction, Pydantic validation,
and strictly bounded error-correction retries.
"""
from typing import Any, Optional, Type, TypeVar, Union
from pydantic import BaseModel

from core.llm.base import BaseLLMProvider
from core.llm.exceptions import redact_secrets
from core.llm.models import (
    LLMGenerationConfig,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
)
from core.llm.structured.exceptions import (
    CorrectionExhaustedError,
    JSONParsingError,
    StructuredValidationError,
)
from core.llm.structured.models import StructuredOutputRequest, StructuredOutputResponse
from core.llm.structured.parser import parse_json_payload
from core.llm.structured.prompt_builder import (
    build_correction_instruction,
    build_schema_instruction,
)
from core.llm.structured.validator import validate_against_schema

T = TypeVar("T", bound=BaseModel)


class StructuredOutputEngine:
    """
    Coordinates structured generation and validation across LLM providers.
    Supports LLMProviderManager or direct BaseLLMProvider instances.
    """

    def __init__(self, dispatcher: Any) -> None:
        """
        Initialize engine with an LLM dispatcher (LLMProviderManager or BaseLLMProvider).
        """
        self._dispatcher = dispatcher

    def _dispatch_generate(
        self,
        request: LLMRequest,
        provider_name: Optional[str] = None,
    ) -> LLMResponse:
        """
        Dispatch request to underlying provider or manager cleanly.
        """
        if isinstance(self._dispatcher, BaseLLMProvider):
            return self._dispatcher.generate(request)
        if provider_name is not None:
            return self._dispatcher.generate(request, provider_name=provider_name)
        return self._dispatcher.generate(request)

    def _inject_schema_instruction(
        self,
        request: LLMRequest,
        schema_class: Type[T],
        system_instruction_override: Optional[str] = None,
    ) -> LLMRequest:
        """
        Inject JSON schema instruction into the LLM request messages.
        """
        base_instruction = build_schema_instruction(schema_class)
        if system_instruction_override and system_instruction_override.strip():
            combined_instruction = f"{system_instruction_override.strip()}\n\n{base_instruction}"
        else:
            combined_instruction = base_instruction

        messages = list(request.messages)
        if messages and messages[0].role == MessageRole.SYSTEM:
            # Append schema instruction to existing system prompt
            existing_sys = messages[0].content.strip()
            messages[0] = LLMMessage(
                role=MessageRole.SYSTEM,
                content=f"{existing_sys}\n\n{combined_instruction}",
            )
        else:
            # Prepend new system prompt with schema instruction
            messages.insert(
                0,
                LLMMessage(role=MessageRole.SYSTEM, content=combined_instruction),
            )

        # Configure response_format for providers supporting native JSON mode
        gen_cfg = request.generation_config
        updates: dict[str, Any] = {}
        if hasattr(gen_cfg, "response_format") and gen_cfg.response_format is None:
            updates["response_format"] = {"type": "json_object"}

        new_gen_cfg = gen_cfg.model_copy(update=updates) if updates else gen_cfg

        return request.model_copy(
            update={
                "messages": messages,
                "generation_config": new_gen_cfg,
            }
        )

    def _build_correction_request(
        self,
        original_request: LLMRequest,
        failed_response_content: str,
        error_summary: str,
        schema_class: Type[T],
    ) -> LLMRequest:
        """
        Build a secondary correction request providing the validation failure context.
        """
        correction_prompt = build_correction_instruction(
            schema_class=schema_class,
            error_summary=error_summary,
        )

        messages = list(original_request.messages)
        # Append assistant turn with failed output, followed by user correction instruction
        messages.append(
            LLMMessage(
                role=MessageRole.ASSISTANT,
                content=failed_response_content,
            )
        )
        messages.append(
            LLMMessage(
                role=MessageRole.USER,
                content=correction_prompt,
            )
        )

        return original_request.model_copy(update={"messages": messages})

    def generate(
        self,
        request: Union[LLMRequest, StructuredOutputRequest[T]],
        schema_class: Optional[Type[T]] = None,
        max_attempts: int = 2,
        provider_name: Optional[str] = None,
    ) -> StructuredOutputResponse[T]:
        """
        Execute structured generation against target Pydantic schema with bounded retries.

        Args:
            request: LLMRequest or StructuredOutputRequest.
            schema_class: Target Pydantic model class (required if request is LLMRequest).
            max_attempts: Maximum attempts (strictly clamped to at most 2: 1 initial + 1 correction).
            provider_name: Optional explicit provider identifier to bypass fallback routing.

        Returns:
            StructuredOutputResponse containing validated Pydantic model and diagnostics.

        Raises:
            JSONParsingError: If LLM output cannot be parsed into valid JSON.
            StructuredValidationError: If JSON fails schema validation.
            CorrectionExhaustedError: If correction attempts reach maximum limit without passing.
        """
        if isinstance(request, StructuredOutputRequest):
            base_req = request.request
            target_schema = request.schema_class
            bounded_attempts = min(max(request.max_attempts, 1), 2)
            sys_override = request.system_instruction
        else:
            if schema_class is None:
                raise ValueError("schema_class must be provided when passing a standard LLMRequest.")
            base_req = request
            target_schema = schema_class
            bounded_attempts = min(max(max_attempts, 1), 2)
            sys_override = None

        # Build Attempt 1 Request
        attempt_1_req = self._inject_schema_instruction(
            request=base_req,
            schema_class=target_schema,
            system_instruction_override=sys_override,
        )

        # Dispatch Attempt 1
        resp_1 = self._dispatch_generate(attempt_1_req, provider_name=provider_name)

        # Attempt 1 Parsing and Validation
        try:
            parsed_1 = parse_json_payload(
                raw_text=resp_1.content,
                expected_root_type=dict,
                provider_name=resp_1.provider_name,
                model_name=resp_1.model_name,
                attempts=1,
            )
            validated_1 = validate_against_schema(
                data=parsed_1,
                schema_class=target_schema,
                provider_name=resp_1.provider_name,
                model_name=resp_1.model_name,
                raw_content=resp_1.content,
                attempts=1,
            )
            raw_meta_1 = dict(resp_1.raw_metadata or {})
            raw_meta_1["structured_attempts"] = 1
            raw_meta_1["correction_occurred"] = False
            raw_meta_1["initial_provider"] = resp_1.provider_name
            raw_meta_1["initial_model"] = resp_1.model_name
            raw_meta_1["final_provider"] = resp_1.provider_name
            raw_meta_1["final_model"] = resp_1.model_name

            if resp_1.usage is not None:
                raw_meta_1["usage_history"] = [
                    {
                        "attempt": 1,
                        "provider": resp_1.provider_name,
                        "model": resp_1.model_name,
                        "prompt_tokens": resp_1.usage.prompt_tokens,
                        "completion_tokens": resp_1.usage.completion_tokens,
                        "total_tokens": resp_1.usage.total_tokens,
                    }
                ]

            return StructuredOutputResponse[T](
                data=validated_1,
                raw_response=resp_1,
                raw_text=resp_1.content,
                attempts=1,
                correction_occurred=False,
                provider_name=resp_1.provider_name,
                model_name=resp_1.model_name,
                raw_metadata=raw_meta_1,
            )
        except (JSONParsingError, StructuredValidationError) as initial_err:
            if bounded_attempts < 2:
                # No correction retry allowed by caller configuration
                raise initial_err

            # Attempt 2: Bounded Correction on the SAME provider
            error_feedback = redact_secrets(str(initial_err))
            correction_req = self._build_correction_request(
                original_request=attempt_1_req,
                failed_response_content=resp_1.content,
                error_summary=error_feedback,
                schema_class=target_schema,
            )

            # Preserve provider stability: stay with the provider that returned HTTP 200
            target_prov = provider_name or resp_1.provider_name
            resp_2 = self._dispatch_generate(correction_req, provider_name=target_prov)

            try:
                parsed_2 = parse_json_payload(
                    raw_text=resp_2.content,
                    expected_root_type=dict,
                    provider_name=resp_2.provider_name,
                    model_name=resp_2.model_name,
                    attempts=2,
                )
                validated_2 = validate_against_schema(
                    data=parsed_2,
                    schema_class=target_schema,
                    provider_name=resp_2.provider_name,
                    model_name=resp_2.model_name,
                    raw_content=resp_2.content,
                    attempts=2,
                )
                # Preserve composite metadata across initial generation and correction
                resp_1_meta = dict(resp_1.raw_metadata or {})
                resp_2_meta = dict(resp_2.raw_metadata or {})
                merged_meta = dict(resp_2_meta)

                # Preserve infrastructure fallback history from initial generation
                fallback_occurred = resp_1_meta.get("fallback_occurred", False)
                if fallback_occurred:
                    merged_meta["fallback_occurred"] = True
                    merged_meta["primary_provider"] = resp_1_meta.get("primary_provider", "groq")
                    if "primary_error" in resp_1_meta:
                        merged_meta["primary_error"] = resp_1_meta["primary_error"]
                    if "secondary_provider" in resp_1_meta:
                        merged_meta["secondary_provider"] = resp_1_meta["secondary_provider"]

                # Structured generation metadata trace
                merged_meta["structured_attempts"] = 2
                merged_meta["correction_occurred"] = True
                merged_meta["initial_provider"] = resp_1.provider_name
                merged_meta["initial_model"] = resp_1.model_name
                merged_meta["correction_provider"] = resp_2.provider_name
                merged_meta["correction_model"] = resp_2.model_name
                merged_meta["final_provider"] = resp_2.provider_name
                merged_meta["final_model"] = resp_2.model_name

                # Preserve usage history from both attempts without fabrication
                usage_history = []
                if resp_1.usage is not None:
                    usage_history.append({
                        "attempt": 1,
                        "provider": resp_1.provider_name,
                        "model": resp_1.model_name,
                        "prompt_tokens": resp_1.usage.prompt_tokens,
                        "completion_tokens": resp_1.usage.completion_tokens,
                        "total_tokens": resp_1.usage.total_tokens,
                    })
                if resp_2.usage is not None:
                    usage_history.append({
                        "attempt": 2,
                        "provider": resp_2.provider_name,
                        "model": resp_2.model_name,
                        "prompt_tokens": resp_2.usage.prompt_tokens,
                        "completion_tokens": resp_2.usage.completion_tokens,
                        "total_tokens": resp_2.usage.total_tokens,
                    })
                if usage_history:
                    merged_meta["usage_history"] = usage_history

                return StructuredOutputResponse[T](
                    data=validated_2,
                    raw_response=resp_2,
                    raw_text=resp_2.content,
                    attempts=2,
                    correction_occurred=True,
                    provider_name=resp_2.provider_name,
                    model_name=resp_2.model_name,
                    raw_metadata=merged_meta,
                )
            except (JSONParsingError, StructuredValidationError) as second_err:
                raise CorrectionExhaustedError(
                    message=f"Structured output validation failed after 2 attempts: {second_err}",
                    last_error=second_err,
                    attempts=2,
                    provider_name=resp_2.provider_name,
                    model_name=resp_2.model_name,
                    raw_content=resp_2.content,
                ) from second_err
