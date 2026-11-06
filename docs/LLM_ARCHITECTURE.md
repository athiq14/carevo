# LLM Provider Abstraction Architecture (Phase 3.1)

## 1. Architectural Motivation & Overview

The **AI Career Intelligence Agent** requires integration with state-of-the-art Large Language Models (LLMs) to perform resume analysis, semantic job matching, career trajectory roadmapping, and interview coaching. Rather than coupling downstream business logic directly to vendor-specific SDKs (such as Google GenAI or Groq SDKs), Phase 3.1 introduces a **provider-independent abstraction layer**.

### Key Architectural Tenets:
- **Vendor Agnosticism:** Core application logic and reasoning agents interact solely with abstract interfaces and normalized data models. Downstream code never imports vendor SDKs directly.
- **Fail-Safe & Predictable:** In Phase 3.1, no silent or automatic fallback is performed. Provider failures produce clear, typed exceptions.
- **Zero-Trust Credential Security:** API keys and credentials are automatically redacted from error messages, debug representations (`__repr__`), and string serializations (`__str__`).
- **Air-Gapped Deterministic Testing:** A fully offline `MockLLMProvider` enables rigorous unit and integration testing with 0 API keys and 0 network calls.

---

## 2. Architecture & Data Flow

```mermaid
flowchart TD
    subgraph ClientLayer["Application & Agent Layer"]
        StreamlitUI["Streamlit Web UI / CLI"]
        CareerAgents["Downstream Agents (Phase 3.4+)"]
    end

    subgraph AbstractionLayer["Core LLM Abstraction Layer (core/llm)"]
        Manager["LLMProviderManager"]
        Models["Normalized Models\n(LLMRequest / LLMResponse)"]
        BaseProvider["<<abstract>>\nBaseLLMProvider"]
        Exceptions["Structured Exceptions\n(Auto-Redacting)"]
    end

    subgraph ConcreteProviders["Provider Implementations"]
        MockProvider["MockLLMProvider\n(Deterministic / Offline)"]
        GeminiProvider["GeminiProvider\n(Phase 3.2)"]
        GroqProvider["GroqProvider\n(Phase 3.3)"]
    end

    subgraph ExternalAPIs["External Provider Services"]
        GoogleGenAI["Google Gemini API"]
        GroqCloud["Groq Cloud API"]
    end

    ClientLayer -->|generate_text() / generate()| Manager
    Manager -->|delegates to active provider| BaseProvider
    BaseProvider <|-- MockProvider
    BaseProvider <|-- GeminiProvider
    BaseProvider <|-- GroqProvider
    
    MockProvider -->|Zero Network Calls| Models
    GeminiProvider -.->|HTTPS / API Key| GoogleGenAI
    GroqProvider -.->|HTTPS / API Key| GroqCloud
```

---

## 3. Core Component Reference

### 3.1 Data Models (`core/llm/models.py`)

All models are immutable Pydantic V2 schemas (`frozen=True`):

| Model | Purpose | Key Attributes |
|---|---|---|
| `MessageRole` | Standardized role enum | `system`, `user`, `assistant` |
| `LLMMessage` | Single conversational message unit | `role`, `content` |
| `LLMGenerationConfig` | Generation hyperparameter controls | `temperature` (0.0–2.0), `max_output_tokens`, `top_p` (0.0–1.0), `timeout`, `stop_sequences` |
| `LLMUsageMetadata` | Token consumption accounting | `prompt_tokens`, `completion_tokens`, `total_tokens` (nullable) |
| `LLMRequest` | Request payload sent to any provider | `messages: list[LLMMessage]`, `generation_config`, factory: `from_prompt()` |
| `LLMResponse` | Normalized provider completion | `content: str`, `provider_name`, `model_name`, `usage`, `finish_reason`, `raw_metadata` |
| `LLMProviderConfig` | Provider runtime configuration | `provider_name`, `model_name`, `api_key`, `temperature`, `max_output_tokens`, `timeout`, `is_api_key_configured` (masked `__repr__`, `__str__`, `model_dump`, and `model_dump_json`) |

### 3.2 Abstract Provider Contract (`core/llm/base.py`)

Every concrete provider must inherit from `BaseLLMProvider` and implement:
- `@property def provider_name(self) -> str`
- `@property def model_name(self) -> str`
- `@property def config(self) -> LLMProviderConfig`
- `def is_available(self) -> bool`
- `def validate_configuration(self) -> None`
- `def generate(self, request: LLMRequest) -> LLMResponse`
- Concrete utility: `generate_text(prompt, system_prompt, generation_config) -> LLMResponse`

### 3.3 Exception Hierarchy & Automatic Secret Redaction (`core/llm/exceptions.py`)

All provider errors derive from `LLMProviderError`:
```
LLMProviderError
  ├── LLMConfigurationError    (invalid settings, malformed parameters)
  ├── LLMAuthenticationError   (missing or invalid API keys)
  ├── LLMRateLimitError        (HTTP 429 quota exhaustion)
  ├── LLMTimeoutError          (network deadline exceeded)
  ├── LLMResponseError         (empty, truncated, or unparseable payload)
  └── LLMFallbackError         (both primary and secondary providers failed in sequence)
```
- **Secret Redaction:** The `redact_secrets()` utility cleans Google API keys (`AIzaSy...`), Groq keys (`gsk_...`), and OpenAI/Anthropic-style tokens (`sk-...`) before storing or formatting error messages.
- **Error Classifier:** `is_transient_error()` distinguishes transient failures (429, 503, 502/504, timeouts, connection errors) from permanent failures (401/403, 400, 404, safety blocks).

### 3.4 Provider Manager (`core/llm/manager.py`)

The `LLMProviderManager` serves as the centralized orchestrator and two-stage fallback router:
- **Priority Strategy:** Primary = Groq (`openai/gpt-oss-120b`), Secondary = Gemini (`GEMINI_MODEL`), Development = `MockLLMProvider`.
- **Fallback Policy:** Exactly TWO attempts max (1 primary + 1 secondary). Zero retry loops.
- **Registration:** `register_provider(provider, set_active=False)`
- **Retrieval:** `get_provider(name=None)` (defaults to active provider)
- **Activation:** `set_active_provider(name)`
- **Fallback Configuration:** `set_fallback_chain(primary, secondary, enabled)`, `set_fallback_enabled(bool)`.
- **Diagnostics:** `get_status()` returns sanitized status dictionaries including `primary_provider`, `secondary_provider`, `fallback_enabled`, and `fallback_status` (`Active`, `Degraded`, `Disabled`, `Unavailable`).
- **Privacy Gating:** Requests carrying sensitive candidate PII (`is_sensitive=True`) have cross-provider fallback disabled by default unless explicitly opted in via `allow_cross_provider_fallback=True`.

### 3.5 Deterministic Mock Provider (`core/llm/mock_provider.py`)

Designed for automated CI/CD and offline development:
- Requires **0 API keys** and triggers **0 network connections**.
- Configurable canned responses via `canned_response`.
- Configurable error simulation via `simulate_error` (for testing downstream resilience against rate-limiting or authentication failures).
- Accurately tracks request count and prompt history via `call_history`.

### 3.6 Google Gemini Provider (`core/llm/gemini_provider.py` - Phase 3.2)

Production adapter integrating Google Gemini via the official `google-genai` SDK:
- **Model Configurable:** Resolves target model via `config.model_name`, `GEMINI_MODEL` environment variable, or defaults to `gemini-3.8-flash` (with support for `gemini-2.5-flash`).
- **Modern Interactions API:** Primary engine uses Google's recommended `client.interactions.create` with `store=False` (enforcing stateless privacy with zero cloud retention), and retains `client.models.generate_content` as fallback.
- **Message Mapping:** Automatically separates system instructions into `types.GenerateContentConfig(system_instruction=...)` or direct `system_instruction` in Interactions API, translating user and assistant roles to Gemini's native `user` and `model` Content parts.
- **Hyperparameters:** Maps `temperature`, `top_p`, `max_output_tokens`, `stop_sequences`, and corrected millisecond `http_options.timeout`.
- **Usage Accounting:** Extracts `prompt_token_count`, `candidates_token_count`, and `total_token_count` into `LLMUsageMetadata`.
- **Safety & Block Filtering:** Detects prompt blocks (`prompt_feedback.block_reason`) and candidate blocks (`finish_reason in ('SAFETY', 'RECITATION', ...)`), raising descriptive `LLMResponseError` exceptions.
- **Error Mapping:** Translates `errors.APIError` (401/403 -> Authentication, 429 -> RateLimit, 408/504 -> Timeout, 400 -> Configuration) and `httpx.TimeoutException` with complete secret redaction.

### 3.7 Groq Provider (`core/llm/groq_provider.py` - Phase 3.3)

High-performance adapter integrating Groq LPU inference via the official `groq` SDK:
- **Model Configurable:** Resolves target model via `config.model_name`, `GROQ_MODEL` environment variable, or defaults to `openai/gpt-oss-120b` (120B open-weight flagship, 131k context window) with verified fallback to `llama-3.1-8b-instant`.
- **Stateless Chat Completions:** Invokes `client.chat.completions.create(...)` for rapid, deterministic conversational inference.
- **Message Mapping:** Translates canonical `MessageRole` values (`system`, `user`, `assistant`) directly into OpenAI-compatible message dictionaries.
- **Hyperparameters:** Maps `temperature`, `top_p`, `max_tokens` (for `max_output_tokens`), `stop` (for `stop_sequences`), and `timeout` in seconds.
- **Usage Accounting:** Extracts `prompt_tokens`, `completion_tokens`, and `total_tokens` from `response.usage` into normalized `LLMUsageMetadata`.
- **Finish Reason:** Normalizes finish reasons (`stop`, `length`) to lowercase strings.
- **Error Mapping:** Maps `groq.AuthenticationError` (401/403), `groq.RateLimitError` (429), `groq.APITimeoutError` (408/504), `groq.BadRequestError` (400), `groq.APIResponseValidationError`, and `groq.APIConnectionError` with full `gsk_...` secret scrubbing.

### 3.8 Intelligent Provider Fallback & Privacy Policy (`core/llm/manager.py` - Phase 3.4)

Provides production resilience against transient outages without risking candidate data leaks:
- **Provider Priority:** Primary is Groq (`openai/gpt-oss-120b`). Secondary is Google Gemini (`gemini-3.8-flash` / configured `GEMINI_MODEL`).
- **Two-Attempt Limit:** Maximum of 2 provider attempts total (1 primary + 1 secondary). Retries and loops are strictly prohibited.
- **Transient vs Permanent Classification:** Only transient errors (429, 503, 502/504, network breaks, timeouts) qualify for fallback. Permanent errors (401/403 auth, 400 bad request, 404 model not found, safety blocks) raise immediately.
- **Privacy Gating for Resume Data:** Requests with `is_sensitive=True` (containing candidate PII or confidential resume data) default to cross-provider fallback DISABLED unless `allow_cross_provider_fallback=True` is explicitly specified by the caller.
- **Audit & Observability:** Responses returning from a fallback execution carry full diagnostic traces in `raw_metadata`: `fallback_occurred=True`, `primary_provider`, `primary_error` (sanitized), and `secondary_provider`.
- **Composite Fallback Error:** If both providers fail in sequence, `LLMFallbackError` is raised, capturing sanitized traces for both attempts without leaking credentials.

### 3.9 Structured Outputs & Validation (`core/llm/structured/` - Phase 3.5)

Provides schema-enforced, type-safe JSON extraction and Pydantic validation across all LLM providers:
- **Generic Schema Support:** Accepts arbitrary caller-provided Pydantic `BaseModel` schemas without hardcoding resume-specific schemas. Supports nested models, lists, dictionaries, enums, and optional fields.
- **Strict JSON Parsing (`parser.py`):** Handles raw JSON objects, single markdown code fences (````json ... ````), and whitespace/BOM. Rejects truncated JSON, multiple ambiguous code fences, multiple top-level objects, and non-object root types. Zero unsafe `eval()` execution. Error diagnostics include JSON error message, line, and column numbers while strictly omitting raw response text previews to prevent candidate PII leakage.
- **PII-Safe Validation (`validator.py`):** Validates parsed data via `schema.model_validate(data)`. Produces sanitized diagnostic error messages stripping sensitive candidate personal data and credentials. Never fabricates missing information.
- **Bounded Error Correction (`engine.py`):** Strictly bounded to at most TWO attempts total (1 initial + at most 1 correction). If the initial output fails validation, an automated correction request is dispatched to the **same** provider containing exact validation feedback. Provider switching is never triggered by validation errors.
- **Convenience API:** Integrated into `LLMProviderManager.generate_structured()` and `BaseLLMProvider.generate_structured()`.

#### 3.9.1 Attempt Boundaries & Execution Limits

To ensure deterministic system behavior and prevent runaway API loops or quota exhaustion, Phase 3.5 establishes strict, unambiguous attempt boundaries across structured generation and provider fallback:

1. **Structured Generation Rounds (Max 2 Rounds):**
   - **Round 1 (Initial Generation):** Submits the caller's prompt augmented with the target Pydantic schema and strict JSON instructions. If parsed and validated successfully, execution terminates immediately.
   - **Round 2 (Error Correction):** If Round 1 fails JSON parsing or Pydantic schema validation, a single correction request is constructed with exact technical error feedback (field names, validation constraints, syntax error line/col) appended to the conversation history. If Round 2 fails, a `CorrectionExhaustedError` is raised. No third round is ever permitted.

2. **Initial Generation Provider Attempts (Max 2 via Fallback):**
   - When Round 1 is dispatched through `LLMProviderManager`, the manager applies Phase 3.4 intelligent fallback:
     - Dispatches to **Primary Provider (Groq)**.
     - If Groq fails with a transient error (e.g. 503, 429, timeout) and fallback is permitted, the manager falls back to **Secondary Provider (Gemini)**.
     - Permanent errors (e.g. 401 auth, 400 bad request) abort immediately without fallback.

3. **Correction Attempts (Max 1, Pinned to Same Provider):**
   - When Round 2 is triggered, the correction request is **pinned** directly to the provider that returned the HTTP 200 payload in Round 1 (`provider_name=resp_1.provider_name`).
   - Direct provider routing bypasses the manager's fallback logic. Under no circumstances does a parsing or validation error trigger fallback to another provider.

4. **Combined Provider Call Upper Bound (Strictly 3 Calls Max at Application Layer):**
   - **Scenario A (Happy Path):** Groq succeeds on Round 1 $\rightarrow$ **1 provider call**.
   - **Scenario B (Groq Correction):** Groq returns invalid JSON on Round 1; Groq corrects on Round 2 $\rightarrow$ **2 provider calls**.
   - **Scenario C (Infrastructure Fallback, Instant Success):** Groq fails 503; Gemini fallback returns valid JSON on Round 1 $\rightarrow$ **2 provider calls** (1 Groq + 1 Gemini).
   - **Scenario D (Worst-Case Infrastructure Fallback + Correction):**
     1. Groq fails with a transient 503 error $\rightarrow$ Call 1 (Groq).
     2. Gemini fallback succeeds at transport level but returns malformed JSON $\rightarrow$ Call 2 (Gemini).
     3. Correction request is dispatched pinned directly to Gemini $\rightarrow$ Call 3 (Gemini).
     - **Total provider calls across the sequence: exactly 3 calls.**
   - The sequence can **never exceed 3 application-level provider calls**.

#### 3.9.2 Groq Reasoning Token Budget Considerations (`openai/gpt-oss-120b`)

1. **Reasoning Token Consumption:**
   - Groq's flagship model `openai/gpt-oss-120b` is an advanced reasoning model. Before emitting visible output tokens (such as JSON syntax), the model performs internal chain-of-thought reasoning that consumes tokens from the `max_tokens` allocation.
2. **Risk of Low Output Token Budgets:**
   - If `max_output_tokens` is configured too low (e.g., 10 to 100 tokens), the model exhausts its entire token quota inside the reasoning block before emitting any visible response text. This results in an empty response string with `finish_reason="length"`, which causes JSON parsing to fail.
3. **Recommended Token Budgets:**
   - For structured requests utilizing `openai/gpt-oss-120b`, configure `max_output_tokens >= 512` (typically 1024–2048 for deep career evaluations and complex resume schemas) to allow sufficient headroom for reasoning followed by complete JSON serialization.
4. **Diagnostic Patterns for Token Exhaustion:**
   - When inspecting provider logs or metadata:
     - Check `finish_reason == "length"`.
     - Check `usage.completion_tokens == max_output_tokens`.
     - If exposed by the SDK, check `usage.reasoning_tokens` to observe reasoning token consumption relative to visible tokens.
     - If `content` is empty or truncated mid-object while `finish_reason` is `"length"`, increase the token limit rather than altering the schema.

---

## 4. Extension Blueprint for Future Phases

### Multi-Provider Orchestration & Routing (Phase 4+)
- **Dynamic Routing:** Route high-reasoning tasks (e.g. strategic career trajectory, gap analysis) to `gemini` or `groq/openai/gpt-oss-120b`, and latency-sensitive tasks to `groq/llama-3.1-8b-instant`.
- **Resume-to-LLM Integration:** Feed validated `ResumeProfile` models into structured career intelligence prompt templates.

