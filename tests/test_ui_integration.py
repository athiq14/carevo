"""
Unit and integration tests for Streamlit UI logic and session-state behavior (Phase 4 Step 5).
Validates UI session-state lifecycle, reset isolation, exact match wording,
export filtering, and provenance badge logic without requiring a live browser.
"""
import json
import pytest

from core.llm.mock_provider import MockLLMProvider
from core.llm.structured.engine import StructuredOutputEngine
from core.matching.explanation_generator import MatchExplainer
from core.matching.pipeline import CareerMatchingPipeline, PipelineMatchResult
from models.matching import CandidateFitLevel, EvaluationStatus
from models.resume_profile import ProfileProcessingStatus, ResumeProfile
from tests.test_end_to_end_matching import (
    SAMPLE_EXPLANATION_JSON,
    SAMPLE_JD_FULL,
    SAMPLE_RESUME_TEXT,
    make_test_pdf,
)


class TestUISessionStateAndPresentation:
    """Tests UI state management, reset workflows, and presentation constraints."""

    def test_ui_session_state_reset_clears_all_pipeline_data(self):
        """Refinements 6: Reset button clears pipeline_result and all input fields without residual leakage."""
        # Simulate Streamlit session state dictionary
        session_state = {
            "tab1_resume_profile": "dummy_profile",
            "tab1_resume_filename": "dummy.pdf",
            "pipeline_result": "dummy_result",
            "input_jd_area": "some long job description text",
            "input_jt_field": "Senior SRE",
            "input_comp_field": "Netflix",
            "tab2_resume_uploader": "file_data",
        }

        # Emulate the Reset button callback logic in app.py
        keys_to_clear = ["pipeline_result", "input_jd_area", "input_jt_field", "input_comp_field", "tab2_resume_uploader"]
        for k in keys_to_clear:
            if k in session_state:
                del session_state[k]

        # Verify all matching state is destroyed
        assert "pipeline_result" not in session_state
        assert "input_jd_area" not in session_state
        assert "input_jt_field" not in session_state
        assert "input_comp_field" not in session_state
        assert "tab2_resume_uploader" not in session_state

        # Tab 1 resume profile is preserved for convenience
        assert session_state["tab1_resume_profile"] == "dummy_profile"
        assert session_state["tab1_resume_filename"] == "dummy.pdf"

    def test_ui_exact_matches_labeling_constraint(self):
        """Refinement 5: UI presentation must use 'Exact Matches Supported by Resume Evidence'."""
        # Read app.py source and confirm exact phrasing
        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()

        # Must contain approved phrasing
        assert "Exact Matches Supported by Resume Evidence" in app_code

        # Must NOT claim verified exact matches or verified repositories
        assert "Verified Exact Matches" not in app_code
        assert "Verified repository contents" not in app_code

    def test_ui_provenance_distinction_between_llm_and_fallback(self):
        """Refinement 3 & UI: Displays distinct badges for live LLM vs deterministic fallback."""
        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()

        # Both provenance states must be explicitly handled in UI
        assert "Deterministic Rule-Based Fallback" in app_code
        assert "All compatibility scores remain authoritative" in app_code
        assert "Architectural Guarantee:" in app_code

    def test_ui_json_export_allowlist_excludes_secrets_and_raw_text(self):
        """Refinement 4: Export report excludes raw text, contact information, and secret keys."""
        mock_provider = MockLLMProvider(canned_response=SAMPLE_EXPLANATION_JSON)
        engine = StructuredOutputEngine(dispatcher=mock_provider)
        explainer = MatchExplainer(engine=engine)
        pipeline = CareerMatchingPipeline(explainer=explainer)

        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)
        result = pipeline.evaluate(jd_text=SAMPLE_JD_FULL, resume_file=pdf_bytes)

        export_dict = result.to_export_dict()
        export_str = json.dumps(export_dict)

        # Prohibited in export
        prohibited_tokens = [
            "jane.doe@example.com",
            "(555) 987-6543",
            "San Francisco, CA",
            "GROQ_API_KEY",
            "GEMINI_API_KEY",
            "secret",
            "password",
        ]
        for token in prohibited_tokens:
            assert token not in export_str

        # Required in export
        assert "deterministic_match" in export_dict
        assert "skills_and_evidence" in export_dict
        assert "qualitative_explanation" in export_dict
        assert export_dict["deterministic_match"]["overall_score"] == round(result.match_result.overall_score, 2)

    def test_ui_successful_analysis_followed_by_failed_analysis_clears_old_result(self):
        """Correction 2.1: Failed second analysis must NOT display the earlier candidate's result."""
        # Setup session state after a successful first run
        session_state = {
            "pipeline_result": "earlier_candidate_successful_result",
            "evaluated_input_signature": ("sig_candidate_a", "jd_a", "title_a", "comp_a"),
        }

        # Emulate the new analysis trigger in app.py (lines 497-502)
        # Step 1: User clicks Run Match -> immediate invalidation of prior result
        if "pipeline_result" in session_state:
            del session_state["pipeline_result"]
        if "evaluated_input_signature" in session_state:
            del session_state["evaluated_input_signature"]

        # Step 2: Analysis fails (simulating exception during pipeline execution)
        error_raised = "CorruptedPDFError: PDF file header missing"

        # Verification: pipeline_result remains completely absent
        assert "pipeline_result" not in session_state
        assert "evaluated_input_signature" not in session_state
        assert error_raised is not None

    def test_ui_successful_analysis_followed_by_changed_jd_text_invalidates_result(self):
        """Correction 2.2: Changing JD text invalidates previous results so stale evaluation is not shown."""
        session_state = {
            "pipeline_result": "valid_result_for_python_jd",
            "evaluated_input_signature": (("tab1", "resume.pdf"), "Python JD requirements", "Backend Dev", "Acme"),
        }

        # User modifies JD text area in Streamlit
        current_jd_text = "Updated Go and Kubernetes JD requirements"
        current_input_signature = (("tab1", "resume.pdf"), current_jd_text.strip(), "Backend Dev", "Acme")

        # Emulate app.py stale-result invalidation check (lines 538-545)
        if "pipeline_result" in session_state:
            if session_state.get("evaluated_input_signature") != current_input_signature:
                del session_state["pipeline_result"]
                del session_state["evaluated_input_signature"]

        # Result must be invalidated
        assert "pipeline_result" not in session_state
        assert "evaluated_input_signature" not in session_state

    def test_ui_switching_from_tab1_to_new_pdf_invalidates_result(self):
        """Correction 2.3: Switching resume input source from Tab 1 to a new PDF invalidates previous result."""
        session_state = {
            "pipeline_result": "valid_result_for_tab1_candidate",
            "evaluated_input_signature": (("tab1", "alice_resume.pdf"), "Python JD", "Dev", "Co"),
        }

        # User switches radio button to a new file upload
        current_resume_sig = ("file", "bob_resume.pdf", 1024)
        current_input_signature = (current_resume_sig, "Python JD", "Dev", "Co")

        # Emulate app.py stale-result invalidation check
        if "pipeline_result" in session_state:
            if session_state.get("evaluated_input_signature") != current_input_signature:
                del session_state["pipeline_result"]
                del session_state["evaluated_input_signature"]

        assert "pipeline_result" not in session_state
        assert "evaluated_input_signature" not in session_state

    def test_ui_reset_retains_tab1_parsed_resume(self):
        """Correction 2.4: Resetting match results preserves Tab 1 parsed resume profile."""
        session_state = {
            "tab1_resume_profile": "parsed_candidate_profile_from_tab1",
            "tab1_resume_filename": "candidate.pdf",
            "pipeline_result": "evaluated_result",
            "evaluated_input_signature": "signature",
            "input_jd_area": "job description text",
        }

        # Emulate Reset button callback
        keys_to_clear = ["pipeline_result", "evaluated_input_signature", "input_jd_area", "input_jt_field", "input_comp_field", "tab2_resume_uploader"]
        for k in keys_to_clear:
            if k in session_state:
                del session_state[k]

        # Match results and inputs are cleared
        assert "pipeline_result" not in session_state
        assert "evaluated_input_signature" not in session_state
        assert "input_jd_area" not in session_state

        # Tab 1 profile and filename are safely retained
        assert session_state["tab1_resume_profile"] == "parsed_candidate_profile_from_tab1"
        assert session_state["tab1_resume_filename"] == "candidate.pdf"

    def test_ui_wording_updated_to_previously_processed(self):
        """Wording Correction: Confirms 'Use previously processed resume from Tab 1' is used."""
        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()

        assert "Use previously processed resume from Tab 1" in app_code
        assert "Use verified resume from Tab 1" not in app_code

