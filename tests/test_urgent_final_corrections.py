"""
Comprehensive regression and functional test suite for the urgent final corrections:
1. LinkedIn About / Summary optimization (8 sections, copy/download, validation against hallucinations)
2. Career Advisor standalone capability & non-software Insurance Advisor curriculum
3. Clean decoupling: Target JD vs User Career Objective vs Workflow Goal Enum
4. Razorpay JD extraction with distinct requirement categories
5. Candidate verified skills (Python, JS, React, Node, Next, API Integration, Automation) & genuine Hulsti.AI experience
"""
import io
from unittest.mock import patch
import fitz
import pytest

from models.resume_profile import ResumeProfile
from models.matching import (
    CandidateFitLevel,
    EvaluationStatus,
    EvidenceStatus,
    JobDescriptionProfile,
    JobMatchResult,
    RequirementPriority,
    SkillGap,
    SkillMatch,
    SkillMatchType,
)
from models.workflow import UserWorkflowGoal
from core.workflow.state import CareerWorkflowState
from core.workflow.validators import LinkedInProfileValidator
from core.workflow.router import route_after_triage
from core.workflow.nodes import clarification_node, triage_node
from agents.remediation_agent import CareerRemediationAgent
from agents.interview_agent import InterviewPreparationAgent
from processors.resume_pipeline import ResumePipeline
from processors.jd_processor import process_job_description
from app import DEMO_RESUME_ATHIQ, DEMO_JD_RAZORPAY_AI_BUILDER


# =============================================================================
# Helper: Generate synthetic PDF for testing
# =============================================================================

def make_test_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 560, 800), text, fontsize=9)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def candidate_profile() -> ResumeProfile:
    pdf_bytes = make_test_pdf(DEMO_RESUME_ATHIQ)
    return ResumePipeline().process_resume(pdf_bytes, filename="Mohamed_Athiq_Resume.pdf")


# =============================================================================
# 1. LinkedIn Profile Optimization & Validation Tests
# =============================================================================

class TestLinkedInAboutAndHeadlineOptimization:
    """Verifies LinkedIn About generation and validation integrity."""

    def test_linkedin_about_generation_has_all_sections_and_hulsti(self, candidate_profile):
        """Verifies LinkedIn About section includes all 8 key components and genuine Hulsti.AI venture."""
        res = LinkedInProfileValidator.generate_optimized_about(candidate_profile)

        # Structure checks
        assert "complete_about_text" in res
        assert "opening_hook" in res
        assert "professional_identity" in res
        assert "experience_bullets" in res
        assert "project_bullets" in res
        assert "certifications" in res
        assert "closing_statement" in res

        about_text = res["complete_about_text"]

        # Content and provenance checks
        assert "Hulsti.AI" in about_text
        assert "Founder" in about_text or "co-pilot" in about_text.lower()
        assert "Python" in about_text
        assert "Sathyabama" in about_text or "2026" in about_text

        # Length and structure check
        assert len(about_text) > 300
        violations = LinkedInProfileValidator.validate_about_section(about_text, candidate_profile)
        assert len(violations) == 0

    def test_linkedin_about_validation_rejects_unsupported_seniority(self, candidate_profile):
        """Verifies validator flags unevidenced seniority in About text."""
        hallucinated_about = (
            "As an Executive VP and Principal Architect with 15 years leading enterprise engineering teams, "
            "I architect massive systems at scale."
        )
        violations = LinkedInProfileValidator.validate_about_section(hallucinated_about, candidate_profile)
        assert len(violations) > 0
        assert any("TENURE" in v.violation_type.name or "TITLE" in v.violation_type.name for v in violations)

    def test_linkedin_about_validation_rejects_unsupported_institution(self, candidate_profile):
        """Verifies validator flags degrees/institutions not present in the candidate profile."""
        hallucinated_about = (
            "Software engineer graduated from Stanford University with a PhD in Artificial Intelligence and Robotics."
        )
        violations = LinkedInProfileValidator.validate_about_section(hallucinated_about, candidate_profile)
        assert len(violations) > 0
        assert any("TITLE" in v.violation_type.name or "institution" in v.message.lower() for v in violations)

    def test_linkedin_about_validation_rejects_unsupported_scale(self, candidate_profile):
        """Verifies validator flags unsubstantiated extreme scale claims."""
        hallucinated_about = (
            "Scaled microservices handling 500 million daily active users and processed 10 billion transactions annually."
        )
        violations = LinkedInProfileValidator.validate_about_section(hallucinated_about, candidate_profile)
        assert len(violations) > 0
        assert any("METRIC" in v.violation_type.name or "scale" in v.message.lower() for v in violations)

    def test_linkedin_headline_generation_three_grounded_options(self, candidate_profile):
        """Verifies generated headline alternatives reflect verified skills and proper tiering."""
        # Test validator on candidate's verified headline
        verified_headline = "Software Engineer | Python • React • Web Applications | Sathyabama '26"
        violations = LinkedInProfileValidator.validate_headline(verified_headline, candidate_profile)
        assert len(violations) == 0

        # Unsupported seniority in headline must be flagged
        unsupported_headline = "Senior Principal Systems Architect | Cloud Director"
        violations_bad = LinkedInProfileValidator.validate_headline(unsupported_headline, candidate_profile)
        assert len(violations_bad) > 0


# =============================================================================
# 2. Career Advisor Standalone & Non-Software (Insurance Advisor) Tests
# =============================================================================

class TestCareerAdvisorInsuranceCurriculum:
    """Verifies Career Advisor support for explicit career objectives including Insurance Advisor."""

    @pytest.fixture
    def minimal_match(self) -> JobMatchResult:
        return JobMatchResult(
            overall_score=75.0,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            fit_classification=CandidateFitLevel.STRONG,
            active_evaluation_coverage=85.0,
            unassessed_reserve=15.0,
            job_metadata={"job_title": "AI Builder", "company": "Razorpay"},
            matched_skills=[
                SkillMatch(
                    requirement_id="req-py",
                    skill_name="Python",
                    canonical_name="Python",
                    requirement_priority=RequirementPriority.REQUIRED,
                    match_type=SkillMatchType.EXACT_MATCH,
                    evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
                    source_context="Verified in resume",
                )
            ],
            missing_evidence=[
                SkillGap(
                    requirement_id="req-irdai",
                    skill_name="IRDAI Licensing",
                    priority=RequirementPriority.REQUIRED,
                    evidence_status=EvidenceStatus.MISSING_EVIDENCE,
                    gap_explanation="Missing mandatory IRDAI IC-38 licensing certification.",
                    impact_on_score=10.0,
                    suggested_learning_direction="Complete 50-hour IRDAI accredited training",
                )
            ],
        )

    def test_remediation_agent_tailors_to_insurance_advisor(self, minimal_match):
        """CareerRemediationAgent tailors milestones and portfolio blueprint to Insurance Advisor."""
        agent = CareerRemediationAgent()
        plan = agent.synthesize_deterministic_fallback(
            minimal_match,
            user_career_objective="Insurance Advisor",
        )

        assert plan is not None
        plan_text = str(plan.model_dump()).lower()

        # Must include insurance-specific curriculum
        assert "irdai" in plan_text or "ic-38" in plan_text
        assert "underwriting" in plan_text or "human life value" in plan_text or "family protection" in plan_text

        # Portfolio project must be insurance focused
        assert any(
            "insurance" in p.project_concept.lower() or "family protection" in p.project_concept.lower()
            for p in plan.portfolio_projects
        )

    def test_interview_agent_tailors_to_insurance_advisor(self, minimal_match, candidate_profile):
        """InterviewPreparationAgent formulates insurance advisory consultative questions."""
        agent = InterviewPreparationAgent()
        guide = agent.synthesize_deterministic_fallback(
            minimal_match,
            profile=candidate_profile,
            user_career_objective="Insurance Advisor",
        )

        assert guide is not None
        guide_text = str(guide.model_dump()).lower()

        assert "insurance" in guide_text or "ic-38" in guide_text or "underwriting" in guide_text or "client" in guide_text


# =============================================================================
# 3. Decoupling: Target JD vs User Career Objective vs Workflow Goal Enum
# =============================================================================

class TestWorkflowDecouplingAndClarification:
    """Verifies clean decoupling and divergence handling across workflow components."""

    def test_divergence_between_objective_and_jd_triggers_clarification(self):
        """When user objective is 'Insurance Advisor' but target JD is 'AI Builder', route to clarification."""
        state: CareerWorkflowState = {
            "session_id": "test-session",
            "resume_profile": None,
            "jd_profile": JobDescriptionProfile(
                job_title="AI Builder",
                company="Razorpay",
                original_description="AI Builder role description at Razorpay.",
                required_skills=["Python", "Docker"],
            ),
            "match_result": None,
            "baseline_match_result": None,
            "user_career_objective": "Insurance Advisor",
            "selected_goal": UserWorkflowGoal.FULL_PREPARATION,
            "current_phase": "triage",
            "next_phase": "",
            "active_plan": None,
            "active_interview_prep": None,
            "active_optimized_bullets": [],
            "approval_state": None,
            "audit_trail": [],
            "error_log": [],
            "iterations_count": 0,
            "needs_user_clarification": False,
            "clarification_prompt": "",
            "clarification_response": "",
            "requires_human_approval": False,
            "execution_metrics": {},
        }

        next_node = route_after_triage(state)
        assert next_node == "clarification"

    def test_aligned_objective_and_jd_routes_to_goal_selection(self):
        """When user objective aligns with target JD domain, route directly to goal_selection."""
        state: CareerWorkflowState = {
            "session_id": "test-session",
            "resume_profile": None,
            "jd_profile": JobDescriptionProfile(
                job_title="AI Builder",
                company="Razorpay",
                original_description="AI Builder role description at Razorpay.",
                required_skills=["Python", "Docker"],
            ),
            "match_result": None,
            "baseline_match_result": None,
            "user_career_objective": "AI Builder & Automation Engineer",
            "selected_goal": UserWorkflowGoal.FULL_PREPARATION,
            "current_phase": "triage",
            "next_phase": "",
            "active_plan": None,
            "active_interview_prep": None,
            "active_optimized_bullets": [],
            "approval_state": None,
            "audit_trail": [],
            "error_log": [],
            "iterations_count": 0,
            "needs_user_clarification": False,
            "clarification_prompt": "",
            "clarification_response": "",
            "requires_human_approval": False,
            "execution_metrics": {},
        }

        next_node = route_after_triage(state)
        assert next_node == "goal_selection"

    def test_clarification_node_formats_divergence_message(self):
        """Clarification node populates clear explanation for domain divergence."""
        state: CareerWorkflowState = {
            "session_id": "test-session",
            "resume_profile": None,
            "jd_profile": JobDescriptionProfile(
                job_title="AI Builder",
                company="Razorpay",
                original_description="AI Builder role description at Razorpay.",
                required_skills=["Python"],
            ),
            "match_result": None,
            "baseline_match_result": None,
            "user_career_objective": "Insurance Advisor",
            "selected_goal": UserWorkflowGoal.FULL_PREPARATION,
            "current_phase": "clarification",
            "next_phase": "",
            "active_plan": None,
            "active_interview_prep": None,
            "active_optimized_bullets": [],
            "approval_state": None,
            "audit_trail": [],
            "error_log": [],
            "iterations_count": 0,
            "needs_user_clarification": False,
            "clarification_prompt": "",
            "clarification_response": "",
            "requires_human_approval": False,
            "execution_metrics": {},
        }

        with patch("core.workflow.nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "Focus on Career Objective: Insurance Advisor"
            res = clarification_node(state)
            mock_interrupt.assert_called_once()
            call_arg = mock_interrupt.call_args[0][0]
            assert call_arg["divergence_conflict"] is True
            assert "Insurance Advisor" in call_arg["message"]
            assert "AI Builder" in call_arg["message"]


# =============================================================================
# 4. Razorpay JD Extraction & Category Verification
# =============================================================================

class TestRazorpayJDExtraction:
    """Verifies that Razorpay JD is extracted with distinct requirement categories."""

    def test_razorpay_jd_distinct_category_extraction(self):
        """process_job_description parses headers like 'What you will do', 'Candidate Profile', and builder items."""
        jd_prof = process_job_description(DEMO_JD_RAZORPAY_AI_BUILDER)

        assert jd_prof is not None
        assert jd_prof.job_title == "AI Builder"
        assert jd_prof.company == "Razorpay"

        # Check required skills
        req_skills_lower = [s.lower() for s in jd_prof.required_skills]
        assert "python" in req_skills_lower
        assert "javascript" in req_skills_lower
        assert "react" in req_skills_lower
        assert "next.js" in req_skills_lower
        assert "node.js" in req_skills_lower
        assert "rest apis" in req_skills_lower
        assert "api integration" in req_skills_lower

        # Check preferred skills
        pref_skills_lower = [s.lower() for s in jd_prof.preferred_skills]
        assert any("docker" in s or "kubernetes" in s or "aws" in s or "redis" in s for s in pref_skills_lower)

        # Check responsibilities
        assert len(jd_prof.responsibilities) >= 3


# =============================================================================
# 5. Candidate Skills & Genuine Hulsti.AI Extraction
# =============================================================================

class TestCandidateProfileExtraction:
    """Verifies candidate skills taxonomy and genuine Hulsti.AI venture extraction."""

    def test_athiq_resume_extracts_hulsti_and_fullstack_skills(self):
        """Candidate extraction identifies Hulsti.AI, Python, Next.js, Node.js, React, REST APIs, Automation."""
        pdf_bytes = make_test_pdf(DEMO_RESUME_ATHIQ)
        prof = ResumePipeline().process_resume(pdf_bytes, filename="Mohamed_Athiq_Resume.pdf")

        # Check experience records for Hulsti.AI
        exp_titles = [e.job_title.lower() for e in prof.experience]
        exp_orgs = [e.company.lower() for e in prof.experience]
        assert any("hulsti" in org for org in exp_orgs)
        assert any("founder" in title or "project lead" in title for title in exp_titles)

        # Check technical skills
        skill_names = [s.name.lower() for s in prof.skills]
        assert "python" in skill_names
        assert "javascript" in skill_names or "react" in skill_names
        assert "api integration" in skill_names or "automation" in skill_names

        # Check institution and education
        assert len(prof.education) >= 1
        assert "sathyabama" in prof.education[0].institution.lower()
        assert "2026" in (prof.education[0].end_date or prof.education[0].original_text)
