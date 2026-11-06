"""
Comprehensive unit and integration tests for Phase 3.5 Structured LLM Outputs & Validation.
Tests JSON parsing, Pydantic schema validation, bounded correction retries,
provider integration (Groq, Gemini, Mock), privacy controls, and security.
Guarantees 100% mock isolation with zero network calls and zero quota consumption.
"""
from enum import Enum
import json
import pytest
from typing import Optional
from unittest.mock import MagicMock
from pydantic import BaseModel, Field

from core.llm.exceptions import LLMRateLimitError, LLMTimeoutError
from core.llm.manager import LLMProviderManager
from core.llm.mock_provider import MockLLMProvider
from core.llm.models import (
    LLMGenerationConfig,
    LLMMessage,
    LLMProviderConfig,
    LLMRequest,
    LLMResponse,
    LLMUsageMetadata,
    MessageRole,
)
from core.llm.structured import (
    CorrectionExhaustedError,
    JSONParsingError,
    StructuredOutputEngine,
    StructuredOutputRequest,
    StructuredOutputResponse,
    StructuredValidationError,
    build_correction_instruction,
    build_schema_instruction,
    extract_json_text,
    format_validation_errors,
    parse_json_payload,
    validate_against_schema,
)


# =====================================================================
# Test Schema Models
# =====================================================================

class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    EXECUTIVE = "executive"


class SkillDetail(BaseModel):
    skill_name: str
    years: int


class CandidateEvaluation(BaseModel):
    candidate_name: str
    level: ExperienceLevel
    skills: list[SkillDetail]
    summary: str
    target_salary: Optional[int] = None
    tags: list[str] = Field(default_factory=list)


# =====================================================================
# Test Group A: JSON Parsing
# =====================================================================

class TestJSONParsing:
    """Verifies strict, secure JSON extraction and root type enforcement."""

    def test_parse_normal_json(self):
        raw = '{"candidate_name": "Jane Doe", "level": "senior", "skills": []}'
        data = parse_json_payload(raw)
        assert data["candidate_name"] == "Jane Doe"
        assert data["level"] == "senior"

    def test_parse_markdown_fenced_json(self):
        raw = "```json\n{\n  \"candidate_name\": \"John Smith\",\n  \"level\": \"mid\"\n}\n```"
        data = parse_json_payload(raw)
        assert data["candidate_name"] == "John Smith"
        assert data["level"] == "mid"

    def test_parse_markdown_fenced_generic(self):
        raw = "```\n{\n  \"key\": \"value\"\n}\n```"
        data = parse_json_payload(raw)
        assert data["key"] == "value"

    def test_parse_whitespace_and_bom(self):
        raw = "\ufeff   \n\t  {\"status\": \"ok\"}  \n  "
        data = parse_json_payload(raw)
        assert data["status"] == "ok"

    def test_reject_invalid_syntax_json(self):
        raw = '{"candidate_name": "Jane", "unclosed_string: 123}'
        with pytest.raises(JSONParsingError, match="Invalid or truncated JSON"):
            parse_json_payload(raw)

    def test_reject_truncated_json(self):
        raw = '{"candidate_name": "Jane Doe", "skills": [{"skill_name": "Python"'
        with pytest.raises(JSONParsingError, match="Invalid or truncated JSON"):
            parse_json_payload(raw)

    def test_reject_multiple_markdown_fences_ambiguous(self):
        raw = "```json\n{\"object\": 1}\n```\nSome commentary\n```json\n{\"object\": 2}\n```"
        with pytest.raises(JSONParsingError, match="Ambiguous LLM output: detected 2 separate markdown code blocks"):
            parse_json_payload(raw)

    def test_reject_multiple_top_level_json_objects(self):
        raw = '{"first": 1} {"second": 2}'
        with pytest.raises(JSONParsingError, match="Invalid or truncated JSON"):
            parse_json_payload(raw)

    def test_reject_plain_text_without_json(self):
        raw = "I am an AI assistant and I cannot find a resume to process."
        with pytest.raises(JSONParsingError, match="Invalid or truncated JSON"):
            parse_json_payload(raw)

    def test_reject_empty_or_whitespace_output(self):
        with pytest.raises(JSONParsingError, match="empty or contains only whitespace"):
            parse_json_payload("   \n\t   ")

    def test_reject_incorrect_root_type_list_when_dict_expected(self):
        raw = '[{"key": "value"}]'
        with pytest.raises(JSONParsingError, match="Incorrect root JSON type: expected dict, got list"):
            parse_json_payload(raw, expected_root_type=dict)

    def test_reject_incorrect_root_type_primitive(self):
        raw = '"just a string"'
        with pytest.raises(JSONParsingError, match="Incorrect root JSON type: expected dict, got str"):
            parse_json_payload(raw, expected_root_type=dict)


# =====================================================================
# Test Group B: Pydantic Schema Validation
# =====================================================================

class TestPydanticValidation:
    """Verifies schema validation across models, nested objects, enums, and types."""

    def test_valid_schema_instantiation(self):
        data = {
            "candidate_name": "Alex Mercer",
            "level": "senior",
            "skills": [{"skill_name": "Python", "years": 8}],
            "summary": "Experienced backend architect",
            "target_salary": 185000,
            "tags": ["cloud", "fastapi"],
        }
        res = validate_against_schema(data, CandidateEvaluation)
        assert res.candidate_name == "Alex Mercer"
        assert res.level == ExperienceLevel.SENIOR
        assert len(res.skills) == 1
        assert res.skills[0].years == 8
        assert res.target_salary == 185000
        assert res.tags == ["cloud", "fastapi"]

    def test_missing_required_field_fails(self):
        data = {
            "level": "senior",
            "skills": [],
            "summary": "Missing name",
        }
        with pytest.raises(StructuredValidationError) as exc_info:
            validate_against_schema(data, CandidateEvaluation)
        err = exc_info.value
        assert "candidate_name" in str(err)
        assert any("candidate_name" in e["field"] for e in err.validation_errors)

    def test_wrong_primitive_type_fails(self):
        data = {
            "candidate_name": 12345,  # Needs to be string
            "level": "mid",
            "skills": [],
            "summary": "Wrong name type",
            "target_salary": "NOT_A_NUMBER",
        }
        with pytest.raises(StructuredValidationError) as exc_info:
            validate_against_schema(data, CandidateEvaluation)
        assert "target_salary" in str(exc_info.value)

    def test_invalid_enum_fails(self):
        data = {
            "candidate_name": "Bob",
            "level": "wizard",  # Not in ExperienceLevel enum
            "skills": [],
            "summary": "Invalid enum level",
        }
        with pytest.raises(StructuredValidationError) as exc_info:
            validate_against_schema(data, CandidateEvaluation)
        assert "level" in str(exc_info.value)

    def test_invalid_nested_object_fails(self):
        data = {
            "candidate_name": "Carol",
            "level": "entry",
            "skills": [{"skill_name": "Go", "years": "three"}],  # years must be int
            "summary": "Invalid nested field",
        }
        with pytest.raises(StructuredValidationError) as exc_info:
            validate_against_schema(data, CandidateEvaluation)
        assert "skills -> 0 -> years" in str(exc_info.value)

    def test_optional_field_omitted(self):
        data = {
            "candidate_name": "David",
            "level": "mid",
            "skills": [],
            "summary": "No target salary provided",
        }
        res = validate_against_schema(data, CandidateEvaluation)
        assert res.target_salary is None

    def test_non_base_model_schema_rejected(self):
        with pytest.raises(TypeError, match="subclass of pydantic.BaseModel"):
            validate_against_schema({}, dict)


# =====================================================================
# Test Group C & D: Retries, Correction, and Bounded Attempts
# =====================================================================

class TestControlledRetriesAndEngine:
    """Verifies bounded 2-attempt correction behavior and provider stability."""

    def test_first_attempt_succeeds_no_correction(self):
        """Initial valid JSON returns immediately with attempts=1 and correction_occurred=False."""
        valid_json = json.dumps({
            "candidate_name": "Alice",
            "level": "senior",
            "skills": [{"skill_name": "Python", "years": 5}],
            "summary": "Solid engineer",
        })
        mock_provider = MockLLMProvider(canned_response=valid_json)
        engine = StructuredOutputEngine(dispatcher=mock_provider)

        req = LLMRequest.from_prompt("Evaluate Alice")
        res = engine.generate(req, CandidateEvaluation)

        assert res.data.candidate_name == "Alice"
        assert res.data.level == ExperienceLevel.SENIOR
        assert res.attempts == 1
        assert res.correction_occurred is False
        assert res.provider_name == "mock"
        assert res.model_name == "mock-career-v1"
        assert len(mock_provider.call_history) == 1

    def test_first_attempt_fails_second_attempt_correction_succeeds(self):
        """First attempt returns malformed JSON, second returns corrected valid JSON -> attempts=2."""
        invalid_first = '{"candidate_name": "Alice", "level": "INVALID_LEVEL", "skills": []}'
        valid_second = json.dumps({
            "candidate_name": "Alice",
            "level": "mid",
            "skills": [{"skill_name": "SQL", "years": 3}],
            "summary": "Corrected profile",
        })

        mock_dispatcher = MagicMock()
        mock_dispatcher.generate.side_effect = [
            LLMResponse(content=invalid_first, provider_name="groq", model_name="openai/gpt-oss-120b"),
            LLMResponse(content=valid_second, provider_name="groq", model_name="openai/gpt-oss-120b"),
        ]

        engine = StructuredOutputEngine(dispatcher=mock_dispatcher)
        req = LLMRequest.from_prompt("Evaluate Alice")
        res = engine.generate(req, CandidateEvaluation, max_attempts=2)

        assert res.data.candidate_name == "Alice"
        assert res.data.level == ExperienceLevel.MID
        assert res.attempts == 2
        assert res.correction_occurred is True
        assert res.provider_name == "groq"
        assert mock_dispatcher.generate.call_count == 2

        # Verify second call included the assistant failed message and user correction feedback
        second_call_req = mock_dispatcher.generate.call_args_list[1][0][0]
        assert len(second_call_req.messages) >= 3
        assistant_turn = second_call_req.messages[-2]
        user_correction_turn = second_call_req.messages[-1]
        assert assistant_turn.role == MessageRole.ASSISTANT
        assert "INVALID_LEVEL" in assistant_turn.content
        assert user_correction_turn.role == MessageRole.USER
        assert "rejected due to JSON or schema validation failures" in user_correction_turn.content

    def test_both_attempts_fail_raises_correction_exhausted(self):
        """When both attempts fail schema validation, CorrectionExhaustedError is raised."""
        bad_json_1 = '{"candidate_name": "Bob"}'  # missing level, skills, summary
        bad_json_2 = '{"candidate_name": "Bob", "level": "bad_enum"}'

        mock_dispatcher = MagicMock()
        mock_dispatcher.generate.side_effect = [
            LLMResponse(content=bad_json_1, provider_name="groq", model_name="openai/gpt-oss-120b"),
            LLMResponse(content=bad_json_2, provider_name="groq", model_name="openai/gpt-oss-120b"),
        ]

        engine = StructuredOutputEngine(dispatcher=mock_dispatcher)
        req = LLMRequest.from_prompt("Evaluate Bob")

        with pytest.raises(CorrectionExhaustedError) as exc_info:
            engine.generate(req, CandidateEvaluation, max_attempts=2)

        err = exc_info.value
        assert err.attempts == 2
        assert err.provider_name == "groq"
        assert isinstance(err.last_error, StructuredValidationError)
        assert mock_dispatcher.generate.call_count == 2

    def test_max_attempts_one_disables_correction(self):
        """When max_attempts=1, an invalid first response fails immediately without second attempt."""
        invalid_json = '{"broken": json}'
        mock_dispatcher = MagicMock()
        mock_dispatcher.generate.return_value = LLMResponse(
            content=invalid_json, provider_name="mock", model_name="m1"
        )

        engine = StructuredOutputEngine(dispatcher=mock_dispatcher)
        req = LLMRequest.from_prompt("Evaluate")

        with pytest.raises(JSONParsingError):
            engine.generate(req, CandidateEvaluation, max_attempts=1)

        assert mock_dispatcher.generate.call_count == 1

    def test_no_provider_switching_on_validation_failure(self):
        """
        Validates that schema/parsing failures do NOT switch providers.
        The correction request is dispatched to the SAME provider that returned the first response.
        """
        invalid_first = '{"candidate_name": "Eve"}'
        valid_second = json.dumps({
            "candidate_name": "Eve",
            "level": "entry",
            "skills": [],
            "summary": "Fixed",
        })

        mock_dispatcher = MagicMock()
        mock_dispatcher.generate.side_effect = [
            LLMResponse(content=invalid_first, provider_name="groq", model_name="openai/gpt-oss-120b"),
            LLMResponse(content=valid_second, provider_name="groq", model_name="openai/gpt-oss-120b"),
        ]

        engine = StructuredOutputEngine(dispatcher=mock_dispatcher)
        req = LLMRequest.from_prompt("Evaluate Eve")
        res = engine.generate(req, CandidateEvaluation, max_attempts=2)

        assert res.attempts == 2
        # Check provider_name passed to second generate call
        second_call_kwargs = mock_dispatcher.generate.call_args_list[1][1]
        assert second_call_kwargs.get("provider_name") == "groq"


# =====================================================================
# Test Group E: Provider Integration via Manager
# =====================================================================

class TestManagerStructuredIntegration:
    """Verifies manager.generate_structured() and BaseLLMProvider.generate_structured()."""

    def test_manager_structured_dispatch_success(self):
        valid_json = json.dumps({
            "candidate_name": "Charlie",
            "level": "executive",
            "skills": [{"skill_name": "Leadership", "years": 15}],
            "summary": "C-level executive",
        })
        mgr = LLMProviderManager()
        mock_prov = MockLLMProvider(canned_response=valid_json)
        mgr.register_provider(mock_prov, set_active=True)

        req = LLMRequest.from_prompt("Evaluate Charlie")
        res = mgr.generate_structured(req, CandidateEvaluation)

        assert isinstance(res, StructuredOutputResponse)
        assert res.data.candidate_name == "Charlie"
        assert res.data.level == ExperienceLevel.EXECUTIVE
        assert res.attempts == 1

    def test_manager_structured_with_explicit_provider_routing(self):
        valid_json = json.dumps({
            "candidate_name": "Diana",
            "level": "mid",
            "skills": [],
            "summary": "Direct Gemini call",
        })
        gemini_mock = MagicMock()
        gemini_mock.provider_name = "gemini"
        gemini_mock.model_name = "gemini-3.8-flash"
        gemini_mock.is_available.return_value = True
        gemini_mock.generate.return_value = LLMResponse(
            content=valid_json, provider_name="gemini", model_name="gemini-3.8-flash"
        )
        gemini_mock.config = LLMProviderConfig(provider_name="gemini", model_name="gemini-3.8-flash")

        mgr = LLMProviderManager()
        mgr.register_provider(gemini_mock)

        req = LLMRequest.from_prompt("Direct route test")
        res = mgr.generate_structured(req, CandidateEvaluation, provider_name="gemini")

        assert res.provider_name == "gemini"
        assert res.model_name == "gemini-3.8-flash"
        assert res.data.candidate_name == "Diana"
        gemini_mock.generate.assert_called_once()

    def test_transient_infrastructure_error_triggers_provider_fallback_during_structured(self):
        """
        If primary (Groq) throws a transient network 503, the underlying manager falls back
        to secondary (Gemini), which generates valid JSON parsed by the structured engine.
        """
        valid_json_gemini = json.dumps({
            "candidate_name": "Frank",
            "level": "senior",
            "skills": [{"skill_name": "Go", "years": 4}],
            "summary": "Recovered via Gemini fallback",
        })

        groq_mock = MagicMock()
        groq_mock.provider_name = "groq"
        groq_mock.model_name = "openai/gpt-oss-120b"
        groq_mock.is_available.return_value = True
        groq_mock.config = LLMProviderConfig(provider_name="groq", model_name="openai/gpt-oss-120b")
        groq_mock.generate.side_effect = LLMTimeoutError("Request timed out", provider_name="groq")

        gemini_mock = MagicMock()
        gemini_mock.provider_name = "gemini"
        gemini_mock.model_name = "gemini-3.8-flash"
        gemini_mock.is_available.return_value = True
        gemini_mock.config = LLMProviderConfig(provider_name="gemini", model_name="gemini-3.8-flash")
        gemini_mock.generate.return_value = LLMResponse(
            content=valid_json_gemini,
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            raw_metadata={"fallback_trace": True},
        )

        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = LLMRequest.from_prompt("Fallback structured test")
        res = mgr.generate_structured(req, CandidateEvaluation)

        assert res.data.candidate_name == "Frank"
        assert res.provider_name == "gemini"
        assert res.raw_metadata["fallback_occurred"] is True
        assert res.raw_metadata["primary_provider"] == "groq"
        groq_mock.generate.assert_called_once()
        gemini_mock.generate.assert_called_once()

    def test_exact_three_call_composite_fallback_and_correction_sequence(self):
        """
        GAP 3 Regression: Tests exact composite sequence across fallback and correction:
        1. Groq primary provider fails with a transient 503 error.
        2. Gemini secondary fallback provider succeeds at transport level, but returns malformed JSON.
        3. Structured output engine triggers an automatic correction turn pinned to Gemini.
        4. Gemini correction request succeeds and returns valid JSON matching the schema.

        Asserts:
        - Groq mock called exactly once.
        - Gemini mock called exactly twice.
        - Total provider calls across the sequence: exactly 3.
        - Response provider is Gemini.
        - Response model is Gemini model.
        - metadata['fallback_occurred'] is True.
        - metadata['primary_provider'] == 'groq'.
        - metadata['correction_provider'] == 'gemini'.
        - metadata['correction_occurred'] is True.
        - Usage metadata accurately aggregates or preserves usage across attempts without missing records.
        """
        groq_mock = MagicMock()
        groq_mock.provider_name = "groq"
        groq_mock.model_name = "openai/gpt-oss-120b"
        groq_mock.is_available.return_value = True
        groq_mock.config = LLMProviderConfig(provider_name="groq", model_name="openai/gpt-oss-120b")
        groq_mock.generate.side_effect = LLMTimeoutError("Groq service unavailable 503", provider_name="groq")

        malformed_json_gemini = '{"candidate_name": "Carol", "level": "senior", unclosed_syntax: '
        valid_json_gemini = json.dumps({
            "candidate_name": "Carol",
            "level": "senior",
            "skills": [{"skill_name": "Python", "years": 7}],
            "summary": "Recovered via Gemini fallback and correction",
        })

        gemini_mock = MagicMock()
        gemini_mock.provider_name = "gemini"
        gemini_mock.model_name = "gemini-3.8-flash"
        gemini_mock.is_available.return_value = True
        gemini_mock.config = LLMProviderConfig(provider_name="gemini", model_name="gemini-3.8-flash")

        # Turn 1: Gemini returns malformed JSON
        resp_attempt_1 = LLMResponse(
            content=malformed_json_gemini,
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            usage=LLMUsageMetadata(prompt_tokens=150, completion_tokens=40, total_tokens=190),
            raw_metadata={"turn": 1},
        )
        # Turn 2: Gemini returns valid JSON
        resp_attempt_2 = LLMResponse(
            content=valid_json_gemini,
            provider_name="gemini",
            model_name="gemini-3.8-flash",
            usage=LLMUsageMetadata(prompt_tokens=220, completion_tokens=60, total_tokens=280),
            raw_metadata={"turn": 2},
        )
        gemini_mock.generate.side_effect = [resp_attempt_1, resp_attempt_2]

        mgr = LLMProviderManager(primary_provider="groq", secondary_provider="gemini", fallback_enabled=True)
        mgr.register_provider(groq_mock)
        mgr.register_provider(gemini_mock)

        req = LLMRequest.from_prompt("Evaluate Carol for Lead Architect")
        res = mgr.generate_structured(req, CandidateEvaluation, max_attempts=2)

        # Assert call counts
        assert groq_mock.generate.call_count == 1
        assert gemini_mock.generate.call_count == 2
        total_calls = groq_mock.generate.call_count + gemini_mock.generate.call_count
        assert total_calls == 3

        # Assert data and identity
        assert res.data.candidate_name == "Carol"
        assert res.data.level == ExperienceLevel.SENIOR
        assert res.provider_name == "gemini"
        assert res.model_name == "gemini-3.8-flash"
        assert res.attempts == 2
        assert res.correction_occurred is True

        # Assert metadata
        meta = res.raw_metadata
        assert meta.get("fallback_occurred") is True
        assert meta.get("primary_provider") == "groq"
        assert meta.get("correction_provider") == "gemini"
        assert meta.get("correction_occurred") is True
        assert meta.get("final_provider") == "gemini"
        assert meta.get("final_model") == "gemini-3.8-flash"

        # Assert usage history preservation across attempts
        assert "usage_history" in meta
        history = meta["usage_history"]
        assert len(history) == 2
        assert history[0]["attempt"] == 1
        assert history[0]["provider"] == "gemini"
        assert history[0]["total_tokens"] == 190
        assert history[1]["attempt"] == 2
        assert history[1]["provider"] == "gemini"
        assert history[1]["total_tokens"] == 280

    def test_base_provider_generate_structured_method(self):
        valid_json = json.dumps({
            "candidate_name": "Grace",
            "level": "mid",
            "skills": [],
            "summary": "Direct provider call",
        })
        mock_prov = MockLLMProvider(canned_response=valid_json)
        req = LLMRequest.from_prompt("Direct BaseLLMProvider method test")
        res = mock_prov.generate_structured(req, CandidateEvaluation)

        assert res.data.candidate_name == "Grace"
        assert res.provider_name == "mock"


# =====================================================================
# Test Group F: Privacy, Security & Secrets
# =====================================================================

class TestPrivacyAndSecurity:
    """Verifies candidate PII protection, secret redaction, and error scrubbing."""

    def test_sensitive_request_permissions_preserved_on_correction(self):
        """Correction requests inherit is_sensitive and allow_cross_provider_fallback."""
        invalid_first = '{"candidate_name": "Secret PII", "level": "invalid"}'
        valid_second = json.dumps({
            "candidate_name": "Secret PII",
            "level": "senior",
            "skills": [],
            "summary": "Sanitized",
        })

        mock_dispatcher = MagicMock()
        mock_dispatcher.generate.side_effect = [
            LLMResponse(content=invalid_first, provider_name="mock", model_name="m1"),
            LLMResponse(content=valid_second, provider_name="mock", model_name="m1"),
        ]

        engine = StructuredOutputEngine(dispatcher=mock_dispatcher)
        req = LLMRequest.from_prompt(
            prompt="Confidential candidate data",
            is_sensitive=True,
            allow_cross_provider_fallback=False,
        )

        res = engine.generate(req, CandidateEvaluation, max_attempts=2)
        assert res.attempts == 2

        # Verify second call preserved privacy flags
        second_call_req = mock_dispatcher.generate.call_args_list[1][0][0]
        assert second_call_req.is_sensitive is True
        assert second_call_req.allow_cross_provider_fallback is False
        assert second_call_req.is_fallback_permitted() is False

    def test_secret_redaction_in_parsing_error(self):
        """API keys inside error messages and raw payloads are scrubbed or excluded."""
        # 1. Direct message secret redaction
        err = JSONParsingError(message="Failed at token gsk_SecretToken12345678901234567890")
        assert "gsk_" not in str(err)
        assert "[REDACTED_API_KEY]" in str(err)
        assert "[REDACTED_API_KEY]" in err.message

        # 2. Raw content containing secret is never exposed in error text or repr
        raw_with_secret = '{"key": "gsk_SecretToken12345678901234567890", unclosed: '
        with pytest.raises(JSONParsingError) as exc_info:
            parse_json_payload(raw_with_secret)

        err_str = str(exc_info.value)
        err_repr = repr(exc_info.value)
        assert "gsk_" not in err_str
        assert "gsk_" not in err_repr
        assert "SecretToken" not in err_str
        assert "line" in err_str
        assert "col" in err_str

    def test_candidate_pii_not_exposed_in_json_parsing_errors(self):
        """
        GAP 1 Regression: Verifies synthetic candidate PII does not leak into:
        - str(error)
        - repr(error)
        - error.message
        - Correction prompt instructions sent to the model.
        """
        candidate_name = "Jane Candidate-Doe"
        candidate_email = "jane.doe@confidential-career.com"
        candidate_phone = "+1-555-867-5309"
        candidate_ssn = "000-12-3456"

        # Case 1: Malformed JSON syntax with candidate PII
        malformed_raw_payload = (
            f'{{"candidate_name": "{candidate_name}", '
            f'"email": "{candidate_email}", '
            f'"phone": "{candidate_phone}", '
            f'"ssn": "{candidate_ssn}", '
            f'broken_syntax: '
        )

        with pytest.raises(JSONParsingError) as exc_info:
            parse_json_payload(malformed_raw_payload)

        err = exc_info.value
        err_str = str(err)
        err_repr = repr(err)
        err_msg = err.message

        for pii in (candidate_name, candidate_email, candidate_phone, candidate_ssn):
            assert pii not in err_str, f"PII '{pii}' leaked in str(err): {err_str}"
            assert pii not in err_repr, f"PII '{pii}' leaked in repr(err): {err_repr}"
            assert pii not in err_msg, f"PII '{pii}' leaked in err.message: {err_msg}"

        # Preserves technical line and column diagnostics
        assert "line" in err_str
        assert "col" in err_str

        # Verify correction prompt instruction built with this error does not leak PII
        correction_instruction = build_correction_instruction(
            schema_class=CandidateEvaluation,
            error_summary=err_str,
        )
        for pii in (candidate_name, candidate_email, candidate_phone, candidate_ssn):
            assert pii not in correction_instruction, f"PII '{pii}' leaked in correction instruction prompt"

        # Case 2: Wrong root type containing candidate PII (e.g. list instead of dict)
        list_payload = json.dumps([{"name": candidate_name, "ssn": candidate_ssn}])
        with pytest.raises(JSONParsingError) as exc_info_type:
            parse_json_payload(list_payload, expected_root_type=dict)

        err_type_str = str(exc_info_type.value)
        assert candidate_name not in err_type_str
        assert candidate_ssn not in err_type_str
        assert "Incorrect root JSON type: expected dict, got list" in err_type_str

        # Case 3: Empty payload raises clean parsing error with no leaked artifacts
        with pytest.raises(JSONParsingError) as exc_info_empty:
            parse_json_payload("   ")
        assert "empty or contains only whitespace" in str(exc_info_empty.value)

    def test_structured_output_request_wrapper(self):
        """Verifies StructuredOutputRequest model wrapper passes parameters cleanly."""
        valid_json = json.dumps({
            "candidate_name": "Henry",
            "level": "entry",
            "skills": [],
            "summary": "Typed request wrapper",
        })
        mock_prov = MockLLMProvider(canned_response=valid_json)
        engine = StructuredOutputEngine(dispatcher=mock_prov)

        struct_req = StructuredOutputRequest(
            request=LLMRequest.from_prompt("Evaluate Henry"),
            schema_class=CandidateEvaluation,
            max_attempts=2,
            system_instruction="Custom evaluation system prompt.",
        )
        res = engine.generate(struct_req)
        assert res.data.candidate_name == "Henry"
        assert res.attempts == 1

