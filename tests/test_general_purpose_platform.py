"""
Comprehensive multi-user regression scenarios for the General-Purpose AI Career Intelligence Platform:
- Scenario A: Athiq demo + Razorpay AI Builder + Insurance Advisor objective.
- Scenario B: Data Analyst candidate + Data Analyst JD + Data Analyst objective.
- Scenario C: Software Engineer candidate + Software Engineer JD + Software Engineer objective.
- Scenario D: Non-technical candidate + no JD + non-technical objective.
- Scenario E: Same candidate + 2 different JDs (matching changes, candidate profile invariant).
- Scenario F: 2 different resumes + same JD (results reflect differing candidate evidence).
- Scenario G: New resume uploaded -> verifies no previous candidate info leaks into new analysis.
- Scenario H: No career objective -> verifies recommendations label and functioning Career Advisor.
"""
import io
import json
import re
from unittest.mock import patch
import fitz
import pytest

from models.resume_profile import ResumeProfile
from models.resume_entities import ExtractedSkill, SkillCategory, WorkExperience, ProjectRecord, EducationRecord
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
from core.workflow.validators import LinkedInProfileValidator, DeterministicWorkflowValidator
from core.workflow.router import route_after_triage
from core.workflow.nodes import clarification_node, triage_node
from agents.remediation_agent import CareerRemediationAgent
from agents.interview_agent import InterviewPreparationAgent
from processors.resume_pipeline import ResumePipeline
from processors.jd_processor import process_job_description
from core.matching.engine import DeterministicMatchingEngine as MatchingEngine
from app import (
    DEMO_RESUME_ATHIQ,
    DEMO_JD_RAZORPAY_AI_BUILDER,
    clear_candidate_session_state,
)


# =============================================================================
# Helper: PDF Generation
# =============================================================================

def make_test_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 560, 800), text, fontsize=9)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# =============================================================================
# Synthetic Resumes & Job Descriptions
# =============================================================================

RESUME_DATA_ANALYST_PRIYA = """PRIYA SHARMA
Chicago, IL | priya.sharma@example.com | https://linkedin.com/in/priyasharma

EDUCATION:
University of Illinois Urbana-Champaign
Bachelor of Science in Statistics | 2020 - 2024
CGPA: 3.85 / 4.0

TECHNICAL SKILLS:
SQL, Python, Tableau, Excel, PowerBI, Pandas, ETL, Data Modeling

EXPERIENCE:
Data Analyst Intern — Midwest Analytics Corp (Jan 2024 - May 2024)
- Analyzed enterprise customer datasets using SQL and Python to extract churn metrics.
- Developed Tableau interactive reporting dashboards for business leadership.

PROJECTS:
E-Commerce Churn Prediction Model (Python, Pandas, SQL)
- Engineered automated data extraction and predictive modeling pipelines.
"""

RESUME_SOFTWARE_ENG_MARCUS = """MARCUS CHEN
Seattle, WA | marcus.chen@example.com | https://linkedin.com/in/marcuschen

EDUCATION:
Georgia Institute of Technology
Master of Science in Computer Science | 2019 - 2021

TECHNICAL SKILLS:
Go, Kubernetes, Docker, PostgreSQL, Redis, gRPC, Distributed Systems

EXPERIENCE:
Senior Software Engineer — Apex Cloud Systems (Jun 2021 - Present)
- Architected high-throughput microservices using Go and gRPC handling 20,000 requests per second.
- Managed Kubernetes clusters, containerized deployments, and Redis caching layers.

PROJECTS:
Distributed Event Ingestion Engine (Go, Kubernetes, Redis)
- Built fault-tolerant streaming ingestion workers with automated failover.
"""

RESUME_NON_TECH_ELENA = """ELENA ROSTOVA
Washington, DC | elena.rostova@example.com | https://linkedin.com/in/elenarostova

EDUCATION:
Georgetown University
Bachelor of Arts in Communications | 2018 - 2022

SKILLS:
Public Relations, Media Relations, Strategic Communications, Crisis Management, Corporate Writing

EXPERIENCE:
Communications Manager — Metro Media Relations (Jul 2022 - Present)
- Led corporate messaging and media outreach across national press outlets.
- Formulated stakeholder communication strategies and executive briefing memos.
"""

JD_DATA_ANALYST = """Target Position: Senior Data Analyst
Company: FinTech Insights
Location: Chicago, IL

Required Qualifications:
- Advanced SQL proficiency for complex multi-table queries and window functions.
- Hands-on Python data analysis experience with Pandas.
- Proven experience building enterprise business dashboards in Tableau.
- Practical experience with Snowflake data warehousing and dbt modeling.

Preferred Qualifications:
- Familiarity with AWS data pipelines and Docker."""

JD_ML_RESEARCHER = """Target Position: Machine Learning Researcher
Company: DeepMind Technologies
Location: London, UK

Required Qualifications:
- Deep expertise in PyTorch and TensorFlow for neural network architectures.
- Experience with Computer Vision, transformers, and CUDA kernel optimization.
- Publications in top-tier machine learning conferences (NeurIPS, ICML)."""


# =============================================================================
# SCENARIO A: Athiq Demo + Razorpay AI Builder + Insurance Advisor Objective
# =============================================================================

class TestScenarioA_AthiqRazorpayInsurance:
    """Scenario A: Athiq demo + Razorpay AI Builder + Insurance Advisor objective."""

    @pytest.fixture
    def athiq_profile(self) -> ResumeProfile:
        pdf_bytes = make_test_pdf(DEMO_RESUME_ATHIQ)
        return ResumePipeline().process_resume(pdf_bytes, filename="Mohamed_Athiq_Resume.pdf")

    @pytest.fixture
    def razorpay_jd(self) -> JobDescriptionProfile:
        return process_job_description(DEMO_JD_RAZORPAY_AI_BUILDER)

    def test_candidate_profile_extracted_authentically(self, athiq_profile):
        """Candidate profile extracts Athiq's genuine credentials without hardcoding."""
        assert "Mohamed Athiq" in (athiq_profile.cleaned_text or "") or "Athiq" in athiq_profile.filename
        assert len(athiq_profile.education) > 0
        edu0 = athiq_profile.education[0]
        assert "Sathyabama" in (edu0.institution or "")
        skills = {s.name.lower() for s in athiq_profile.skills}
        assert "python" in skills
        assert "rest apis" in skills

    def test_matching_evaluates_independently(self, athiq_profile, razorpay_jd):
        """Phase 4 matching evaluates Razorpay JD against Athiq's evidence without score drift."""
        match_result = MatchingEngine().match(athiq_profile, razorpay_jd)

        assert 0.0 <= match_result.overall_score <= 100.0
        # Athiq has Python and REST APIs demonstrated
        matched_names = {s.canonical_name.lower() for s in match_result.matched_skills}
        assert "python" in matched_names

    def test_career_advisor_tailors_to_insurance_without_modifying_jd_match(self, athiq_profile, razorpay_jd):
        """Explicit objective 'Insurance Advisor' generates insurance blueprint without mutating JD match."""
        match_result = MatchingEngine().match(athiq_profile, razorpay_jd)
        orig_score = match_result.overall_score

        agent = CareerRemediationAgent()
        plan, _ = agent.run(
            match_result=match_result,
            profile=athiq_profile,
            jd_profile=razorpay_jd,
            user_career_objective="Insurance Advisor",
        )

        assert "insurance" in plan.executive_summary.lower()
        # Phase 4 match result score was NOT overwritten
        assert match_result.overall_score == orig_score

    def test_workflow_divergence_routes_to_clarification(self, athiq_profile, razorpay_jd):
        """LangGraph workflow triggers clarification when career objective diverges from target JD."""
        match_result = MatchingEngine().match(athiq_profile, razorpay_jd)

        initial_state: CareerWorkflowState = {
            "session_id": "test_scenario_a",
            "candidate_profile": athiq_profile,
            "resume_profile": athiq_profile,
            "jd_profile": razorpay_jd,
            "match_result": match_result,
            "baseline_match_snapshot": None,
            "workflow_goal": UserWorkflowGoal.FULL_PREPARATION,
            "user_career_objective": "Insurance Advisor",
            "execution_trace": [],
            "current_step": "triage",
            "next_step": "",
            "clarification_needed": False,
            "clarification_message": None,
            "clarification_options": [],
            "user_clarification_response": None,
            "remediation_plan": None,
            "resume_optimization_report": None,
            "interview_prep_kit": None,
            "final_synthesis": None,
            "audit_trail": [],
            "errors": [],
            "telemetry": {},
            "requires_human_approval": False,
            "approval_status": "PENDING",
            "validation_failures": [],
        }

        # Verify triage routing detects domain conflict between Insurance Advisor and AI Builder
        decision = route_after_triage(initial_state)
        assert decision == "clarification"

        # Verify clarification node generates divergence prompt
        with patch("core.workflow.nodes.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "Focus on Career Objective: Insurance Advisor"
            clarif_res = clarification_node(initial_state)
            mock_interrupt.assert_called_once()
            call_arg = mock_interrupt.call_args[0][0]
            assert call_arg["divergence_conflict"] is True
            assert "Insurance Advisor" in call_arg["message"]
            assert "AI Builder" in call_arg["message"]


# =============================================================================
# SCENARIO B: Data Analyst Candidate + Data Analyst JD + Data Analyst Objective
# =============================================================================

class TestScenarioB_DataAnalystCandidate:
    """Scenario B: Data Analyst candidate + Data Analyst JD + Data Analyst objective."""

    @pytest.fixture
    def priya_profile(self) -> ResumeProfile:
        pdf_bytes = make_test_pdf(RESUME_DATA_ANALYST_PRIYA)
        return ResumePipeline().process_resume(pdf_bytes, filename="Priya_Sharma_Resume.pdf")

    @pytest.fixture
    def data_analyst_jd(self) -> JobDescriptionProfile:
        return process_job_description(JD_DATA_ANALYST)

    def test_candidate_profile_has_zero_athiq_leakage(self, priya_profile):
        """Priya Sharma's profile has authentic name, education, and zero Mohamed Athiq / Hulsti references."""
        assert "Priya Sharma" in (priya_profile.cleaned_text or "") or "Priya" in priya_profile.filename
        assert len(priya_profile.education) > 0
        assert "Illinois" in (priya_profile.education[0].institution or "")
        corpus = (priya_profile.cleaned_text or "").lower()
        assert "athiq" not in corpus
        assert "sathyabama" not in corpus
        assert "hulsti" not in corpus

    def test_data_analyst_jd_matching(self, priya_profile, data_analyst_jd):
        """Matches SQL, Python, Tableau and flags Snowflake, dbt as missing gaps."""
        match_result = MatchingEngine().match(priya_profile, data_analyst_jd)

        matched_names = {s.canonical_name.lower() for s in match_result.matched_skills}
        assert "sql" in matched_names or "python" in matched_names or "tableau" in matched_names

        missing_names = {g.skill_name.lower() for g in match_result.missing_evidence}
        # Snowflake and dbt were not in Priya's resume
        assert any("snowflake" in m or "dbt" in m or "data warehous" in m for m in missing_names)

    def test_remediation_plan_tailored_to_data_analyst(self, priya_profile, data_analyst_jd):
        """Remediation plan targets Data Analyst missing skills with zero Athiq hardcoding."""
        match_result = MatchingEngine().match(priya_profile, data_analyst_jd)

        agent = CareerRemediationAgent()
        plan = agent.synthesize_deterministic_fallback(
            match_result=match_result,
            user_career_objective="Senior Data Analyst",
        )

        assert "data analyst" in plan.executive_summary.lower()
        assert "athiq" not in plan.executive_summary.lower()
        assert "hulsti" not in plan.executive_summary.lower()

    def test_linkedin_about_generation_for_data_analyst(self, priya_profile):
        """LinkedIn About generator derives Priya's name and Illinois degree without Athiq hardcoding."""
        about = LinkedInProfileValidator.generate_optimized_about(profile=priya_profile)
        full_text = about["complete_about_text"]

        assert "Priya Sharma" in full_text or "Illinois" in full_text
        assert "Mohamed Athiq" not in full_text
        assert "Sathyabama" not in full_text
        assert "Hulsti" not in full_text


# =============================================================================
# SCENARIO C: Software Engineer Candidate + Software Engineer JD + Objective
# =============================================================================

class TestScenarioC_SoftwareEngineerCandidate:
    """Scenario C: Software Engineer candidate + Software Engineer JD + Software Engineer objective."""

    @pytest.fixture
    def marcus_profile(self) -> ResumeProfile:
        pdf_bytes = make_test_pdf(RESUME_SOFTWARE_ENG_MARCUS)
        return ResumePipeline().process_resume(pdf_bytes, filename="Marcus_Chen_Resume.pdf")

    def test_experienced_profile_has_seniority_grounding(self, marcus_profile):
        """Marcus Chen has 4+ years verified experience and MS from Georgia Tech."""
        assert "Marcus Chen" in (marcus_profile.cleaned_text or "") or "Marcus" in marcus_profile.filename
        assert len(marcus_profile.experience) > 0
        exp0 = marcus_profile.experience[0]
        assert "Senior Software Engineer" in (exp0.job_title or "")
        assert "Apex Cloud Systems" in (exp0.job_title or "") or "Apex Cloud Systems" in (exp0.company or "") or "Apex Cloud Systems" in (exp0.original_text or "")

    def test_interview_prep_grounds_in_marcus_projects(self, marcus_profile):
        """InterviewPreparationAgent grounds technical questions in Marcus's project and Go stack."""
        dummy_match = JobMatchResult(
            job_metadata={"job_title": "Staff Infrastructure Engineer"},
            overall_score=82.0,
            fit_classification=CandidateFitLevel.STRONG,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            active_evaluation_coverage=85.0,
            unassessed_reserve=15.0,
            match_breakdown=[],
            matched_skills=[
                SkillMatch(canonical_name="Go", skill_name="Go", match_type=SkillMatchType.EXACT_MATCH, requirement_priority=RequirementPriority.REQUIRED, evidence_status=EvidenceStatus.VERIFIED_EVIDENCE),
                SkillMatch(canonical_name="Kubernetes", skill_name="Kubernetes", match_type=SkillMatchType.EXACT_MATCH, requirement_priority=RequirementPriority.REQUIRED, evidence_status=EvidenceStatus.VERIFIED_EVIDENCE),
            ],
            missing_evidence=[
                SkillGap(skill_name="Apache Kafka", priority=RequirementPriority.REQUIRED, gap_explanation="Missing Apache Kafka", evidence_status=EvidenceStatus.MISSING_EVIDENCE),
            ],
            missing_skills=[],
        )

        agent = InterviewPreparationAgent()
        kit = agent.synthesize_deterministic_fallback(
            match_result=dummy_match,
            profile=marcus_profile,
            user_career_objective="Staff Infrastructure Engineer",
        )

        # Technical deep dive references Marcus's actual project or skills
        tech_q = kit.technical_deep_dive_questions[0]
        assert "Distributed Event Ingestion Engine" in tech_q.question_text or "Go" in tech_q.question_text
        # Gap probe addresses Apache Kafka honestly
        gap_q = kit.gap_mitigation_questions[0]
        assert "Apache Kafka" in gap_q.question_text or "Kafka" in gap_q.question_text

    def test_linkedin_validator_allows_senior_title_with_sufficient_tenure(self, marcus_profile):
        """LinkedInProfileValidator permits Senior title for candidates with 3+ years experience."""
        valid_headline = "Senior Software Engineer | Go • Kubernetes • Distributed Systems | Georgia Tech"
        violations = LinkedInProfileValidator.validate_headline(
            headline=valid_headline,
            profile=marcus_profile,
        )
        seniority_viols = [v for v in violations if "senior" in v.message.lower()]
        assert len(seniority_viols) == 0


# =============================================================================
# SCENARIO D: Non-Technical Candidate + No JD + Non-Technical Objective
# =============================================================================

class TestScenarioD_NonTechnicalCandidate:
    """Scenario D: Non-technical candidate + no JD + non-technical objective."""

    @pytest.fixture
    def elena_profile(self) -> ResumeProfile:
        pdf_bytes = make_test_pdf(RESUME_NON_TECH_ELENA)
        return ResumePipeline().process_resume(pdf_bytes, filename="Elena_Rostova_Resume.pdf")

    def test_non_technical_profile_parses_cleanly(self, elena_profile):
        """Communications candidate parses cleanly without requiring tech skills."""
        assert "Elena Rostova" in (elena_profile.cleaned_text or "") or "Elena" in elena_profile.filename
        assert len(elena_profile.education) > 0
        assert "Georgetown" in (elena_profile.education[0].institution or "")

    def test_standalone_linkedin_optimization_no_tech_crash(self, elena_profile):
        """LinkedIn optimization generates clean About section without crashing on missing tech skills."""
        about = LinkedInProfileValidator.generate_optimized_about(
            profile=elena_profile,
            headline="Communications Manager | Strategic Communications • Public Relations",
        )
        full_text = about["complete_about_text"]
        assert "Elena Rostova" in full_text or "Georgetown" in full_text
        assert "Hulsti" not in full_text


# =============================================================================
# SCENARIO E: Same Candidate + 2 Different JDs (Matching Changes, Profile Invariant)
# =============================================================================

class TestScenarioE_SameCandidateDifferentJDs:
    """Scenario E: Same candidate + 2 different JDs (matching changes, candidate profile invariant)."""

    @pytest.fixture
    def marcus_profile(self) -> ResumeProfile:
        pdf_bytes = make_test_pdf(RESUME_SOFTWARE_ENG_MARCUS)
        return ResumePipeline().process_resume(pdf_bytes, filename="Marcus_Chen_Resume.pdf")

    def test_candidate_profile_invariant_across_different_jds(self, marcus_profile):
        """Candidate profile remains cryptographically invariant when evaluated against different JDs."""
        import hashlib
        profile_json = marcus_profile.model_dump_json()
        digest_initial = hashlib.sha256(profile_json.encode("utf-8")).hexdigest()

        engine = MatchingEngine()

        jd1 = process_job_description(JD_DATA_ANALYST)
        match1 = engine.match(marcus_profile, jd1)

        jd2 = process_job_description(JD_ML_RESEARCHER)
        match2 = engine.match(marcus_profile, jd2)

        # Profile is invariant
        profile_json_after = marcus_profile.model_dump_json()
        digest_after = hashlib.sha256(profile_json_after.encode("utf-8")).hexdigest()
        assert digest_initial == digest_after

        # Matching results reflect different JD criteria
        assert match1.job_metadata.get("job_title") != match2.job_metadata.get("job_title")


# =============================================================================
# SCENARIO F: 2 Different Resumes + Same JD (Distinct Evidence Results)
# =============================================================================

class TestScenarioF_TwoCandidatesSameJD:
    """Scenario F: 2 different resumes + same JD (results reflect differing candidate evidence)."""

    def test_different_resumes_yield_distinct_matches_against_same_jd(self):
        """Two candidates evaluated against the same JD produce distinct matched skills and gap lists."""
        marcus = ResumePipeline().process_resume(make_test_pdf(RESUME_SOFTWARE_ENG_MARCUS), "Marcus_Chen.pdf")
        priya = ResumePipeline().process_resume(make_test_pdf(RESUME_DATA_ANALYST_PRIYA), "Priya_Sharma.pdf")

        engine = MatchingEngine()
        jd = process_job_description(JD_DATA_ANALYST)

        match_marcus = engine.match(marcus, jd)
        match_priya = engine.match(priya, jd)

        # Priya has SQL, Python, Tableau -> matches data requirements
        # Marcus has Go, Kubernetes, Redis -> doesn't match SQL/Tableau
        priya_matched = {s.canonical_name.lower() for s in match_priya.matched_skills}
        marcus_matched = {s.canonical_name.lower() for s in match_marcus.matched_skills}

        assert priya_matched != marcus_matched
        assert "tableau" in priya_matched or "sql" in priya_matched
        assert "tableau" not in marcus_matched


# =============================================================================
# SCENARIO G: Candidate Data Isolation Across Uploads
# =============================================================================

class TestScenarioG_CandidateDataIsolation:
    """Scenario G: New resume uploaded -> verifies no previous candidate info leaks into new analysis."""

    def test_clear_candidate_session_state_purges_all_artifacts(self):
        """clear_candidate_session_state eliminates all previous match, workflow, and advisor artifacts."""
        import streamlit as st

        # Simulate Candidate A active session
        st.session_state["pipeline_result"] = "Candidate_A_Pipeline_Result"
        st.session_state["tab2_match_result"] = "Candidate_A_Match_Result"
        st.session_state["agentic_workflow_state"] = {"candidate": "Candidate A"}
        st.session_state["active_remediation_plan"] = "Candidate_A_Plan"
        st.session_state["active_interview_prep"] = "Candidate_A_Prep"
        st.session_state["workflow_thread_id"] = "thread_candidate_a"
        st.session_state["tab4_goal_selected"] = "FULL_PREPARATION"
        st.session_state["advisor_generated_plan"] = "Candidate_A_Advisor_Plan"
        st.session_state["linkedin_optimized_result"] = "Candidate_A_LI"

        # Invoke isolation function
        clear_candidate_session_state()

        # All Candidate A artifacts MUST be purged
        assert "pipeline_result" not in st.session_state
        assert "tab2_match_result" not in st.session_state
        assert "agentic_workflow_state" not in st.session_state
        assert "active_remediation_plan" not in st.session_state
        assert "active_interview_prep" not in st.session_state
        assert "workflow_thread_id" not in st.session_state
        assert "tab4_goal_selected" not in st.session_state
        assert "advisor_generated_plan" not in st.session_state
        assert "linkedin_optimized_result" not in st.session_state


# =============================================================================
# SCENARIO H: No Career Objective Entered by User
# =============================================================================

class TestScenarioH_NoCareerObjectiveEntered:
    """Scenario H: No career objective -> verifies recommendations label and functioning Career Advisor."""

    @pytest.fixture
    def priya_profile(self) -> ResumeProfile:
        pdf_bytes = make_test_pdf(RESUME_DATA_ANALYST_PRIYA)
        return ResumePipeline().process_resume(pdf_bytes, filename="Priya_Sharma_Resume.pdf")

    def test_remediation_agent_with_empty_objective_does_not_default_to_insurance(self, priya_profile):
        """When user_career_objective is empty or None, agent does not force Insurance Advisor."""
        dummy_match = JobMatchResult(
            job_metadata={"job_title": "Data Analyst"},
            overall_score=75.0,
            fit_classification=CandidateFitLevel.STRONG,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            active_evaluation_coverage=80.0,
            unassessed_reserve=20.0,
            match_breakdown=[],
            matched_skills=[
                SkillMatch(canonical_name="SQL", skill_name="SQL", match_type=SkillMatchType.EXACT_MATCH, requirement_priority=RequirementPriority.REQUIRED, evidence_status=EvidenceStatus.VERIFIED_EVIDENCE),
                SkillMatch(canonical_name="Python", skill_name="Python", match_type=SkillMatchType.EXACT_MATCH, requirement_priority=RequirementPriority.REQUIRED, evidence_status=EvidenceStatus.VERIFIED_EVIDENCE),
            ],
            missing_evidence=[
                SkillGap(skill_name="Snowflake", priority=RequirementPriority.REQUIRED, gap_explanation="Missing Snowflake", evidence_status=EvidenceStatus.MISSING_EVIDENCE),
            ],
            missing_skills=[],
        )

        agent = CareerRemediationAgent()
        plan = agent.synthesize_deterministic_fallback(
            match_result=dummy_match,
            user_career_objective="",  # Empty objective
        )

        # Must NOT default to Insurance Advisor
        assert "insurance" not in plan.executive_summary.lower()
        assert "ic-38" not in plan.executive_summary.lower()
        # Must focus on target role / missing skills
        assert "data analyst" in plan.executive_summary.lower() or "snowflake" in plan.milestones[0].concrete_deliverable.lower()

    def test_interview_agent_with_empty_objective_does_not_default_to_insurance(self, priya_profile):
        """When user_career_objective is empty or None, interview agent does not force Insurance questions."""
        dummy_match = JobMatchResult(
            job_metadata={"job_title": "Data Analyst"},
            overall_score=75.0,
            fit_classification=CandidateFitLevel.STRONG,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            active_evaluation_coverage=80.0,
            unassessed_reserve=20.0,
            match_breakdown=[],
            matched_skills=[
                SkillMatch(canonical_name="SQL", skill_name="SQL", match_type=SkillMatchType.EXACT_MATCH, requirement_priority=RequirementPriority.REQUIRED, evidence_status=EvidenceStatus.VERIFIED_EVIDENCE),
            ],
            missing_evidence=[
                SkillGap(skill_name="Snowflake", priority=RequirementPriority.REQUIRED, gap_explanation="Missing Snowflake", evidence_status=EvidenceStatus.MISSING_EVIDENCE),
            ],
            missing_skills=[],
        )

        agent = InterviewPreparationAgent()
        kit = agent.synthesize_deterministic_fallback(
            match_result=dummy_match,
            profile=priya_profile,
            user_career_objective="",  # Empty objective
        )

        # Must NOT formulate Insurance consultative questions
        assert "insurance" not in kit.role_context_overview.lower()
        assert "human life value" not in kit.role_context_overview.lower()
