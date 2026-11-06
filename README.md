# AI Career Intelligence Agent

An enterprise-ready AI career intelligence and matching platform designed to provide rigorous candidate-job compatibility evaluation, deterministic multi-dimensional scoring, ATS compliance auditing, and evidence-grounded career growth explanations.

---

## 🏛️ System Architecture

The platform separates **authoritative mathematical evaluation** from **qualitative narrative explanation**, guaranteeing 100% reproducible, un-inflated scoring while delivering intuitive career insights.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. Candidate Ingestion                          │
│  Uploaded PDF Stream ──► In-Memory Validation (PyMuPDF)               │
│                      ──► Text Cleaner & Section Detector               │
│                      ──► Entity Extractor (Skills, Exp, Edu, Proj)     │
│                      ──► Immutable ResumeProfile                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                        2. Employer JD Ingestion                        │
│  Raw Job Description ──► Input Sanitizer & Length Boundary             │
│                      ──► Canonical Skill Taxonomy Lookup               │
│                      ──► Threshold Counter (Full vs Provisional)       │
│                      ──► Immutable JobDescriptionProfile               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  3. Deterministic Matching Engine                      │
│  • Dimension 1: Technical Skills (Exact, Synonym, Related Tech)        │
│  • Dimension 2: Work Experience (Tenure, Alignment Tiers, Recency)     │
│  • Dimension 3: Academic Education (Degree Level, Core/Adjacent STEM)  │
│  • Dimension 4: Engineering Projects (Code, Metrics, Offline URLs)     │
│  • Weight Manager: Anti-inflation 60% caps & unassessed reserve        │
│  • Output: Authoritative JobMatchResult (100% Deterministic)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    4. Qualitative Explanation Layer                    │
│  • Allowlist Payload Builder (Zero raw resume text, PII scrubbed)      │
│  • XML Closing-Tag Sanitizer (Prevents prompt-injection escape)        │
│  • StructuredOutputEngine (Enforces MatchExplanation Pydantic schema)  │
│  • Provider Manager (Groq Primary ──► Gemini Secondary Failover)       │
│  • Rule-Based Fallback Generator (Offline deterministic resilience)    │
│  • Immutability Verification (Guarantees zero score mutation under -O) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                     5. Interactive Streamlit UI                        │
│  • Real-time st.status pipeline progress tracking                      │
│  • Hero Compatibility Metric Cards & Fit Classification Badges         │
│  • Dimensional Scoring Cards & Neutralization Transparency             │
│  • Skills Audit: Exact Matches Supported by Evidence vs Missing Gaps   │
│  • Deep Dive Qualitative Assessments & Interview Preparation           │
│  • Allowlist-Filtered Downloadable JSON Evaluation Report              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔒 Core Invariants & Architectural Guarantees

1. **Deterministic Authority**:
   - The LLM **never** calculates, recalculates, modifies, or inflates compatibility scores, weights, or fit classifications.
   - All numerical scores and dimension weights originate exclusively from the authoritative offline Python matching engine.
2. **Score Immutability**:
   - Explicit non-assert checks (`verify_result_immutability`) verify all 16 evaluation fields before and after explanation attachment.
   - Enforcement remains 100% active even under optimized execution (`python -O` and `PYTHONOPTIMIZE=1`).
3. **Prompt Injection Boundary**:
   - Candidate evidence is strictly quarantined within `<untrusted_evaluation_evidence>` envelopes.
   - All XML closing tags (such as `</untrusted_evaluation_evidence>` or `</instructions>`) are safely escaped in JSON (`<\/`) before prompt interpolation.
4. **Candidate Privacy**:
   - Resumes are processed strictly in memory (`BytesIO`); no files are persisted to disk.
   - Candidate names, email addresses, phone numbers, and physical addresses are scrubbed from LLM payloads and JSON exports.
5. **Cross-Provider Resilience**:
   - Production requests route through Groq (primary) with automatic failover to Google Gemini (secondary).
   - If all external providers fail or are unconfigured, the system synthesizes a deterministic, fact-grounded explanation with zero pipeline errors.

---

## 🛠️ Tech Stack

- **Runtime**: Python 3.11+
- **Frontend**: Streamlit
- **Document Processing**: PyMuPDF (`fitz`)
- **Domain Modeling**: Pydantic V2 (Immutable `frozen=True` schemas)
- **LLM Infrastructure**: Groq API, Google Gemini API, MockLLMProvider (Offline dev/testing)
- **Testing**: Pytest (392 comprehensive unit, integration, and regression tests)

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14 installed on your system.
- Git installed.

### 2. Virtual Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai-career-intelligence-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (cmd.exe):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the `.env.example` file to create your local `.env`:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` to configure your API keys:
```ini
# Application Configuration
APP_NAME="AI Career Intelligence Agent"
APP_ENV="development"
APP_DEBUG="true"

# Primary Provider (Groq)
GROQ_API_KEY="gsk_your_groq_api_key_here"
GROQ_MODEL="openai/gpt-oss-120b"

# Secondary Provider (Google Gemini Fallback)
GEMINI_API_KEY="AIzaSy_your_gemini_api_key_here"
GEMINI_MODEL="gemini-3.8-flash"
```

> **Note:** If no API keys are configured, the application automatically uses the offline rule-based fallback generator, allowing full functional evaluation without external network access.

---

## 🚀 Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`:
1. **Resume Analysis Tab**: Upload a PDF resume to audit extractable sections, identify canonical skills, and review structured entities.
2. **Job Description Matching Tab**:
   - Choose whether to evaluate the resume from Tab 1 or upload a new PDF resume directly.
   - Paste a job description (or click one of the quick-fill sample buttons: Senior Backend, Sparse/Provisional, or Culture/Insufficient Evidence).
   - Click **⚡ Run Intelligent Career Match** to execute the pipeline with live `st.status` progress tracking.
   - Inspect the multi-dimensional compatibility dashboard and download the allowlist-sanitized JSON report.

---

## 🧪 Running the Test Suite

Run the complete automated test suite:

```bash
pytest -v
```

### Key Test Suites:
- `tests/test_end_to_end_matching.py`: Complete pipeline integration (dual input modes, provisional JDs, score immutability, zero-network socket isolation).
- `tests/test_ui_integration.py`: UI session-state lifecycle, reset isolation, exact match wording, and export filtering.
- `tests/test_explanation_generator.py`: Prompt-injection defense, XML tag escaping, score immutability under `python -O`, and deterministic fallback synthesis.
- `tests/test_matching_engine.py`: Authoritative deterministic multi-dimensional compatibility calculations.
- `tests/test_jd_processor.py`: Job description parsing, taxonomy normalization, and requirement-counting thresholds.
- `tests/test_resume_pipeline.py`: In-memory PDF document extraction, text normalization, and entity extraction.

---

## ⚠️ Known Limitations

1. **Scanned Documents**: Resumes that are purely image-based scans without text layers are detected and flagged as `requires_ocr`. Standard processing handles text-based PDFs.
2. **Repository Inspection**: Candidate GitHub and GitLab portfolio URLs are structurally validated offline for HTTPS syntax and accepted domains; remote repository code quality and accessibility are not inspected over the network.
3. **Self-Reported Metrics**: Performance improvements (e.g. latency reductions, throughput gains) extracted from candidate project descriptions are identified and disclosed as candidate self-reported claims, not independently verified measurements.
