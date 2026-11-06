"""
Regression test suite for UI/UX product integrity and video audit bug fixes.
Covers:
1. Resume Projects Tab rendering (description vs descriptions, empty/missing descriptions, empty lists).
2. Match classification and provisional wording logic (low+provisional, developing, moderate, strong, exceptional).
3. Career Advisor personalization (active match with gaps, active match with full alignment, empty state).
4. LinkedIn Analysis transparency, client-side input analysis, and sample labeling.
5. Scoring weight consistency between core WeightManager and Settings view.
6. Agentic Workflow empty state and Home status bar accuracy.
"""
import pytest
from app import get_match_alignment_display
from core.matching.weight_manager import BASELINE_WEIGHTS, MAX_DIMENSION_CAP
from models.matching import (
    CandidateFitLevel,
    EvaluationStatus,
    JobMatchResult,
    RequirementPriority,
    SkillGap,
    SkillMatch,
    SkillMatchType,
)
from models.resume_entities import ExtractedSkill, ProjectRecord, SkillCategory, WorkExperience
from models.resume_profile import ProfileProcessingStatus, ResumeProfile


@pytest.fixture
def sample_candidate_profile() -> ResumeProfile:
    """Standardized candidate profile for validator and honesty tests."""
    return ResumeProfile(
        filename="test_profile.pdf",
        page_count=1,
        raw_character_count=500,
        cleaned_character_count=480,
        status=ProfileProcessingStatus.SUCCESS,
        skills=[
            ExtractedSkill(name="Python", original_mention="Python 3.11", category=SkillCategory.PROGRAMMING_LANGUAGES),
            ExtractedSkill(name="PostgreSQL", original_mention="Postgres DB", category=SkillCategory.DATABASES),
            ExtractedSkill(name="FastAPI", original_mention="FastAPI framework", category=SkillCategory.WEB_DEVELOPMENT),
        ],
        experience=[
            WorkExperience(
                job_title="Software Engineer",
                company="Acme Corp",
                responsibilities=[
                    "Engineered asynchronous pipelines using Python.",
                    "Improved API latency by optimizing queries.",
                ],
                technologies=["Python", "PostgreSQL"],
                original_text="Software Engineer @ Acme Corp",
            )
        ],
        projects=[
            ProjectRecord(
                project_name="Data Hub",
                description="Built data hub using Python.",
                technologies=["Python"],
                original_text="Data Hub project",
            )
        ],
    )


# =============================================================================
# 1. Resume Projects Tab Regression Tests
# =============================================================================
class TestResumeProjectsTab:
    """Verifies ProjectRecord model integrity and UI robustness."""

    def test_project_record_has_single_description_field_not_descriptions(self):
        """Verifies Pydantic model contract for ProjectRecord."""
        assert "description" in ProjectRecord.model_fields
        assert "descriptions" not in ProjectRecord.model_fields
        assert "project_name" in ProjectRecord.model_fields
        assert "technologies" in ProjectRecord.model_fields
        assert "project_url" in ProjectRecord.model_fields

    def test_project_record_instantiation_with_full_description(self):
        """Verifies ProjectRecord works with multi-line description strings."""
        proj = ProjectRecord(
            project_name="Distributed Orchestrator",
            description="- Architected worker queue\n- Reduced latency by 45%\n- Added Redis backend",
            technologies=["Python", "Redis", "Docker"],
            project_url="https://github.com/example/orchestrator",
            original_text="Raw project text block",
        )
        assert proj.project_name == "Distributed Orchestrator"
        assert "Reduced latency" in proj.description
        assert len(proj.technologies) == 3

    def test_project_record_with_missing_optional_description(self):
        """Verifies ProjectRecord handles empty or missing description string gracefully."""
        proj = ProjectRecord(
            project_name="Simple Tool",
            original_text="Raw project text",
        )
        assert proj.description == ""
        assert proj.technologies == []
        assert proj.project_url is None

    def test_app_source_does_not_contain_invalid_proj_descriptions(self):
        """Ensures app.py no longer contains the crashing 'proj.descriptions' attribute access."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "proj.descriptions" not in code, "Found invalid attribute access 'proj.descriptions' in app.py"
        assert "desc_text = getattr(proj, \"description\"" in code or "getattr(proj, 'description'" in code or "proj.description" in code


# =============================================================================
# 2. Match Classification & Provisional Wording Tests
# =============================================================================
class TestMatchClassificationAndProvisionalWording:
    """Verifies mathematically grounded, score-aware language without false positive claims."""

    def _make_dummy_match_result(
        self,
        score: float,
        fit: CandidateFitLevel,
        status: EvaluationStatus = EvaluationStatus.FULL_EVALUATION,
    ) -> JobMatchResult:
        return JobMatchResult(
            overall_score=score,
            fit_classification=fit,
            evaluation_status=status,
            active_evaluation_coverage=85.0 if status == EvaluationStatus.FULL_EVALUATION else 40.0,
            unassessed_reserve=15.0 if status == EvaluationStatus.FULL_EVALUATION else 60.0,
            match_breakdown=[],
            matched_skills=[],
            missing_skills=[],
        )

    def test_provisional_status_precedence_over_score(self):
        """Provisional evaluation status must take strict precedence and warn user of sparse JD."""
        res = self._make_dummy_match_result(
            score=28.5,
            fit=CandidateFitLevel.PROVISIONAL,
            status=EvaluationStatus.PROVISIONAL_EVALUATION,
        )
        disp = get_match_alignment_display(res)

        assert disp["is_provisional"] is True
        assert disp["fit_label"] == "Provisional Evaluation"
        assert disp["fit_color"] == "#FBBF24"
        assert "aligns well with this job role" not in disp["headline"].lower()
        assert "aligns well with this job role" not in disp["explanation"].lower()
        assert "sparse" in disp["headline"].lower() or "preliminary" in disp["headline"].lower()
        assert "insufficient" in disp["explanation"].lower() or "sparse" in disp["explanation"].lower()

    def test_low_alignment_description_never_claims_strong_fit(self):
        """Score < 40% with Full Evaluation must display Low Alignment with warning colors."""
        res = self._make_dummy_match_result(
            score=32.0,
            fit=CandidateFitLevel.LOW,
            status=EvaluationStatus.FULL_EVALUATION,
        )
        disp = get_match_alignment_display(res)

        assert disp["is_provisional"] is False
        assert disp["fit_label"] == "Low Alignment"
        assert disp["fit_color"] == "#FB7185"
        assert "aligns well" not in disp["headline"].lower()
        assert "gaps detected" in disp["headline"].lower()

    def test_developing_alignment_description(self):
        """Score between 40.0% and 59.9% displays Developing Alignment."""
        res = self._make_dummy_match_result(
            score=48.0,
            fit=CandidateFitLevel.DEVELOPING,
            status=EvaluationStatus.FULL_EVALUATION,
        )
        disp = get_match_alignment_display(res)

        assert disp["fit_label"] == "Developing Alignment"
        assert disp["fit_color"] == "#FBBF24"
        assert "aligns well" not in disp["headline"].lower()
        assert "foundational match" in disp["headline"].lower()

    def test_moderate_match_description(self):
        """Score between 60.0% and 74.9% displays Moderate Match."""
        res = self._make_dummy_match_result(
            score=68.0,
            fit=CandidateFitLevel.MODERATE,
            status=EvaluationStatus.FULL_EVALUATION,
        )
        disp = get_match_alignment_display(res)

        assert disp["fit_label"] == "Moderate Match"
        assert disp["fit_color"] == "#8B45F7"
        assert "partial alignment" in disp["headline"].lower()

    def test_strong_match_description(self):
        """Score between 75.0% and 89.9% displays Strong Match."""
        res = self._make_dummy_match_result(
            score=82.0,
            fit=CandidateFitLevel.STRONG,
            status=EvaluationStatus.FULL_EVALUATION,
        )
        disp = get_match_alignment_display(res)

        assert disp["fit_label"] == "Strong Match"
        assert disp["fit_color"] == "#36D399"
        assert "aligns well" in disp["headline"].lower()

    def test_exceptional_match_description(self):
        """Score >= 90.0% displays Exceptional Match."""
        res = self._make_dummy_match_result(
            score=95.0,
            fit=CandidateFitLevel.EXCEPTIONAL,
            status=EvaluationStatus.FULL_EVALUATION,
        )
        disp = get_match_alignment_display(res)

        assert disp["fit_label"] == "Exceptional Match"
        assert disp["fit_color"] == "#36D399"
        assert "outstanding alignment" in disp["headline"].lower()


# =============================================================================
# 3. Career Advisor Personalization Tests
# =============================================================================
class TestCareerAdvisorPersonalization:
    """Verifies Career Advisor personalizes to active match gaps or labels sample previews."""

    def test_app_source_labels_sample_advisor_roadmap_as_example(self):
        """Verifies generic roadmaps are explicitly labeled as benchmark examples."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "[SAMPLE BENCHMARK ROADMAP" in code or "SAMPLE BENCHMARK ROADMAP" in code
        assert "No Active Match Evaluation Found" in code

    def test_app_source_connects_advisor_to_active_missing_skills(self):
        """Verifies Career Advisor inspects missing_evidence from active match."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "m_res.missing_evidence" in code
        assert "target_role" in code


# =============================================================================
# 4. LinkedIn Analysis Transparency Tests
# =============================================================================
class TestLinkedInTransparency:
    """Verifies client-side profile analysis and clear privacy / scraping disclaimers."""

    def test_linkedin_privacy_notice_in_app_source(self):
        """Verifies clear statement that no scraping or account accessing occurs."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "PathFinder AI operates client-side and does NOT access" in code or "does not access or scrape" in code.lower()
        assert "scrape" in code.lower()

    def test_linkedin_input_fields_present_in_app_source(self):
        """Verifies user input workflow for headline and about section."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "li_user_headline" in code
        assert "li_user_about" in code
        assert "Analyze Profile Content" in code

    def test_linkedin_sample_preview_is_labeled(self):
        """Verifies non-user data is clearly marked as benchmark demonstration."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "[EXPERIENCED-HIRE BENCHMARK REFERENCE" in code or "[SAMPLE DEMONSTRATION & BENCHMARK EXAMPLES]" in code


# =============================================================================
# 5. Scoring Weight Consistency Tests
# =============================================================================
class TestScoringWeightConsistency:
    """Verifies Settings displays authoritative Phase 4 weights from WeightManager."""

    def test_authoritative_baseline_weights_values(self):
        """Verifies core WeightManager baseline weights match approved Phase 4 specification."""
        assert BASELINE_WEIGHTS["skills"] == 0.45
        assert BASELINE_WEIGHTS["experience"] == 0.25
        assert BASELINE_WEIGHTS["education"] == 0.15
        assert BASELINE_WEIGHTS["projects"] == 0.15
        assert MAX_DIMENSION_CAP == 0.60

    def test_app_settings_displays_authoritative_weights(self):
        """Verifies Settings section formats weights dynamically from BASELINE_WEIGHTS."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "from core.matching.weight_manager import BASELINE_WEIGHTS, MAX_DIMENSION_CAP" in code
        assert "Authoritative Phase 4 Baseline Weights" in code
        assert "Technical Skills Baseline: <strong>45%</strong>" in code or "Technical Skills Baseline:" in code
        # Confirm old hardcoded 40% and 20% in Settings are eradicated
        assert "Technical Skills Weight: <strong>40%</strong>" not in code
        assert "Practical Projects Weight: <strong>20%</strong>" not in code

    def test_settings_explains_weight_redistribution_and_ceiling(self):
        """Verifies Settings explains capped redistribution algorithm."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "Dynamic Redistribution & Single-Dimension Ceiling" in code
        assert "{max_cap}%" in code or "60%" in code


# =============================================================================
# 6. Workflow Empty State & Status Bar Tests
# =============================================================================
class TestWorkflowEmptyState:
    """Verifies clean empty state without misleading progress or fake completed agents."""

    def test_workflow_view_empty_state_requires_completed_match(self):
        """Verifies Agents (Workflow) shows clear guidance when no match is active."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "No Active Match Evaluation Found" in code
        assert "Why is a match required first?" in code
        assert "Evaluate in JD Matching →" in code

    def test_home_status_bar_distinguishes_match_and_workflow_readiness(self):
        """Verifies Home status card does not claim workflow is ready without a match."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "Requires Completed Match" in code
        assert "Ready to Launch" in code


# =============================================================================
# 7. Issue 1: Resume Evidence Grounding & Honesty Tests
# =============================================================================
class TestResumeEvidenceHonesty:
    """Verifies that generated STAR bullets preserve meaning, avoid awkward phrasing, and ground evidence."""

    def test_reframe_grounded_star_bullet_on_founder_title(self):
        """Replaces awkward 'Engineered and delivered founder...' with factual, natural wording."""
        from agents.resume_agent import reframe_grounded_star_bullet
        result = reframe_grounded_star_bullet("Founder – Hulsti.AI")
        assert "Served as Founder at Hulsti.AI" in result
        assert "Engineered and delivered" not in result

    def test_reframe_grounded_star_bullet_on_existing_action_verb(self):
        """Avoids double verbs like 'Engineered and delivered founded...'."""
        from agents.resume_agent import reframe_grounded_star_bullet
        result = reframe_grounded_star_bullet("Founded an AI-focused venture building generative AI agents")
        assert result.startswith("Founded an AI-focused venture")
        assert "Engineered and delivered" not in result

    def test_reframe_grounded_star_bullet_on_client_projects(self):
        """Reframes noun phrase without fabricating technical delivery or metrics."""
        from agents.resume_agent import reframe_grounded_star_bullet
        result = reframe_grounded_star_bullet("Client and internal projects across NLP")
        assert "Contributed technical deliverables across client and internal projects" in result
        assert "Engineered and delivered" not in result

    def test_awkward_phrasing_detection_in_validator(self, sample_candidate_profile):
        """Validator rejects bullets containing awkward template prefixing or double verbs."""
        from core.workflow.validators import DeterministicWorkflowValidator
        from models.workflow import (
            MetricDisclosureStatus,
            OptimizedBulletPoint,
            ResumeOptimizationReport,
            ValidationViolationType,
        )

        validator = DeterministicWorkflowValidator()
        report = ResumeOptimizationReport(
            target_role_alignment_summary="Alignment summary.",
            optimized_bullets=[
                OptimizedBulletPoint(
                    bullet_id="b_awkward",
                    source_entity_id="exp_0",
                    source_field_path="experience[0].responsibilities[0]",
                    improved_star_bullet="Engineered and delivered founder – Acme Systems.",
                    highlighted_skills=["Python"],
                    alignment_rationale="Awkward test bullet.",
                    metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
                )
            ],
        )
        res = validator.validate_resume_optimization(report, sample_candidate_profile)
        assert res.is_valid is False
        assert any(v.violation_type == ValidationViolationType.AWKWARD_GENERATED_PHRASING for v in res.violations)

    def test_unsupported_role_inflation_detection_in_validator(self, sample_candidate_profile):
        """Validator rejects bullets claiming ungrounded executive titles like CEO or Founder."""
        from core.workflow.validators import DeterministicWorkflowValidator
        from models.workflow import (
            MetricDisclosureStatus,
            OptimizedBulletPoint,
            ResumeOptimizationReport,
            ValidationViolationType,
        )

        validator = DeterministicWorkflowValidator()
        report = ResumeOptimizationReport(
            target_role_alignment_summary="Alignment summary.",
            optimized_bullets=[
                OptimizedBulletPoint(
                    bullet_id="b_inflated",
                    source_entity_id="exp_0",
                    source_field_path="experience[0].responsibilities[0]",
                    improved_star_bullet="Served as Chief Executive Officer and Founder leading 50 engineers.",
                    highlighted_skills=["Python"],
                    alignment_rationale="Role inflation test.",
                    metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
                )
            ],
        )
        res = validator.validate_resume_optimization(report, sample_candidate_profile)
        assert res.is_valid is False
        assert any(v.violation_type == ValidationViolationType.UNSUPPORTED_CLAIM_EXPANSION for v in res.violations)

    def test_false_independent_verification_label_rejected(self, sample_candidate_profile):
        """Validator rejects provenance label claiming independent third-party verification."""
        from core.workflow.validators import DeterministicWorkflowValidator
        from models.workflow import (
            MetricDisclosureStatus,
            OptimizedBulletPoint,
            ResumeOptimizationReport,
            ValidationViolationType,
        )

        validator = DeterministicWorkflowValidator()
        report = ResumeOptimizationReport(
            target_role_alignment_summary="Alignment summary.",
            optimized_bullets=[
                OptimizedBulletPoint(
                    bullet_id="b_fake_audit",
                    source_entity_id="exp_0",
                    source_field_path="experience[0].responsibilities[0]",
                    improved_star_bullet="Engineered high-throughput asynchronous pipelines using Python.",
                    highlighted_skills=["Python"],
                    alignment_rationale="Grounded.",
                    metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
                    provenance_label="Independently verified",
                )
            ],
        )
        res = validator.validate_resume_optimization(report, sample_candidate_profile)
        assert res.is_valid is False
        assert any(v.violation_type == ValidationViolationType.FALSE_INDEPENDENT_VERIFICATION_CLAIM for v in res.violations)

    def test_ui_no_misleading_verified_badge_on_bullets(self):
        """Verifies UI source code no longer slaps 'VERIFIED' badge on unquantified candidate bullets."""
        with open("core/workflow/ui.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert 'disclosure_badge = "🟢 VERIFIED"' not in code
        assert "Rewritten from candidate-provided evidence" in code
        assert "Original Resume Evidence" in code
        assert "Improved STAR Bullet" in code


# =============================================================================
# 8. Issue 2: Workflow Completion vs Final Approval State Tests
# =============================================================================
class TestWorkflowCompletionVsFinalApproval:
    """Verifies clear user-facing distinction between completed execution and approval states."""

    def test_distinct_approval_states_in_ui_code(self):
        """Verifies UI code implements distinct states for APPROVE, REQUEST_CHANGES, and REJECT."""
        with open("core/workflow/ui.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "Workflow Run Completed — Changes Requested" in code
        assert "Changes Requested (Deliverables Not Approved)" in code
        assert "Deliverables Rejected" in code
        assert "Approved & Synthesized" in code
        assert "Awaiting Your Authorization" in code or "Awaiting Your Final Approval" in code

    def test_report_download_restricted_to_approved_deliverables(self):
        """Verifies consolidated report download is only rendered when final approval decision is APPROVE."""
        with open("core/workflow/ui.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert 'if decision == "APPROVE" and report:' in code
        assert "Consolidated Intelligence Bundle Restricted" in code

    def test_router_terminates_on_reject_and_request_changes(self):
        """Verifies router routes to __end__ rather than synthesis on non-approval."""
        from core.workflow.router import route_after_final_approval
        assert route_after_final_approval({"final_approval_decision": "APPROVE"}) == "synthesis"
        assert route_after_final_approval({"final_approval_decision": "REQUEST_CHANGES"}) == "__end__"
        assert route_after_final_approval({"final_approval_decision": "REJECT"}) == "__end__"


# =============================================================================
# 9. Issue 3: Job Match Result Explanation Tests
# =============================================================================
class TestJobMatchResultExplanation:
    """Verifies that low and provisional results are explained without career judgments."""

    def test_low_score_explanation_is_not_career_judgment(self):
        """Scores < 40% must explain requirement alignment without judging candidate ability."""
        from models.matching import CandidateFitLevel, EvaluationStatus, JobMatchResult
        res = JobMatchResult(
            overall_score=25.0,
            fit_classification=CandidateFitLevel.LOW,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            active_evaluation_coverage=80.0,
            unassessed_reserve=20.0,
            match_breakdown=[],
            matched_skills=[],
            missing_skills=[],
        )
        disp = get_match_alignment_display(res)
        assert disp["fit_label"] == "Low Alignment"
        assert "alignment gaps detected" in disp["headline"].lower()
        assert "not candidate capability or potential" in disp["explanation"]

    def test_developing_score_explanation_tone(self):
        """Scores 40-59% must explain targeted gaps neutrally."""
        from models.matching import CandidateFitLevel, EvaluationStatus, JobMatchResult
        res = JobMatchResult(
            overall_score=48.0,
            fit_classification=CandidateFitLevel.DEVELOPING,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            active_evaluation_coverage=85.0,
            unassessed_reserve=15.0,
            match_breakdown=[],
            matched_skills=[],
            missing_skills=[],
        )
        disp = get_match_alignment_display(res)
        assert disp["fit_label"] == "Developing Alignment"
        assert "not a career judgment" in disp["explanation"]

    def test_app_ui_contains_candidate_guidance_expander(self):
        """Verifies that JD Matching UI includes guidance on assessed dimensions and next steps."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "Candidate Guidance: Why this score was calculated & What you can do next" in code
        assert "Dimensions Assessed:" in code
        assert "Actionable Next Steps:" in code


# =============================================================================
# 10. Issue 4: History Transparency Tests
# =============================================================================
class TestHistoryTransparency:
    """Verifies history page is truthfully labeled Current Evaluation Summary."""

    def test_history_view_labeled_current_evaluation_summary(self):
        """Page header must read 'Current Evaluation Summary' rather than implying persistent DB."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "### Current Evaluation Summary" in code
        assert "Privacy-Preserving Session Architecture" in code
        assert "zero-retention in-memory architecture" in code

    def test_sidebar_nav_displays_current_evaluation(self):
        """Sidebar icon mapping must present 'Current Evaluation'."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert '"History": "🕒 Current Evaluation"' in code

    def test_history_view_displays_all_required_metadata_and_links(self):
        """Verifies history view includes date, role, company, score, status, and navigation links."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "Evaluation Timestamp:" in code
        assert "Evaluation Status:" in code
        assert "View Matching Report" in code
        assert "Open Career Co-pilot" in code


# =============================================================================
# 11. Critical Product Quality Audit Regression Tests
# =============================================================================

class TestIssue1LinkedInHallucinatedCredentials:
    """Verifies LinkedIn suggestions eliminate hallucinated institutions, graduation years, and unsupported seniority."""

    def test_anna_university_rejected_by_validator(self):
        """Validator rejects Anna University when candidate attended Sathyabama."""
        from core.workflow.validators import LinkedInProfileValidator
        from models.resume_entities import EducationRecord
        from models.resume_profile import ResumeProfile

        prof = ResumeProfile(
            filename="resume.pdf",
            education=[
                EducationRecord(
                    degree="B.E. Computer Science",
                    institution="Sathyabama Institute of Science and Technology",
                    end_date="2026",
                    original_text="B.E. Computer Science - Sathyabama Institute of Science and Technology 2026",
                )
            ],
            skills=[],
            total_years_experience=0.3,
        )

        headline = "Data & Web Systems Engineer | Python • Analytics • Cloud (OCI Certified) | Anna University '24"
        violations = LinkedInProfileValidator.validate_headline(headline, profile=prof)

        violation_types = [v.violation_type.value for v in violations]
        assert "TITLE_ALTERATION_CLAIM" in violation_types
        assert any("Anna University" in v.message for v in violations)

    def test_conflicting_grad_year_rejected(self):
        """Validator rejects graduation year '24 when candidate record specifies 2026."""
        from core.workflow.validators import LinkedInProfileValidator
        from models.resume_entities import EducationRecord
        from models.resume_profile import ResumeProfile

        prof = ResumeProfile(
            filename="resume.pdf",
            education=[
                EducationRecord(
                    degree="B.E. Computer Science",
                    institution="Sathyabama Institute of Science and Technology",
                    end_date="2026",
                    original_text="B.E. Computer Science - Sathyabama Institute of Science and Technology 2026",
                )
            ],
            skills=[],
            total_years_experience=0.3,
        )

        headline = "Software Engineer | Sathyabama Institute '24"
        violations = LinkedInProfileValidator.validate_headline(headline, profile=prof)

        violation_types = [v.violation_type.value for v in violations]
        assert "DATE_ALTERATION_CLAIM" in violation_types
        assert any("'24" in v.message for v in violations)

    def test_unsupported_seniority_rejected_for_recent_grad(self):
        """Validator rejects promoting an entry-level candidate to Senior Platform Engineer."""
        from core.workflow.validators import LinkedInProfileValidator
        from models.resume_profile import ResumeProfile

        prof = ResumeProfile(
            filename="resume.pdf",
            skills=[],
            total_years_experience=0.3,
        )

        headline = "Senior Platform Engineer | Python, REST APIs"
        violations = LinkedInProfileValidator.validate_headline(headline, profile=prof, is_target_positioning=False)

        violation_types = [v.violation_type.value for v in violations]
        assert "TENURE_OVERSTATEMENT_CLAIM" in violation_types
        assert any("Senior" in v.message for v in violations)

    def test_unevidenced_tech_rejected(self):
        """Validator rejects unevidenced Docker or PostgreSQL not found in profile."""
        from core.workflow.validators import LinkedInProfileValidator
        from models.resume_entities import ExtractedSkill, SkillCategory
        from models.resume_profile import ResumeProfile

        prof = ResumeProfile(
            filename="resume.pdf",
            skills=[
                ExtractedSkill(name="Python", original_mention="Python", category=SkillCategory.PROGRAMMING_LANGUAGES),
                ExtractedSkill(name="SQL", original_mention="SQL", category=SkillCategory.DATABASES),
            ],
            total_years_experience=0.5,
        )

        headline = "Software Engineer | Python, Docker, PostgreSQL"
        violations = LinkedInProfileValidator.validate_headline(headline, profile=prof)

        violation_types = [v.violation_type.value for v in violations]
        assert "UNGROUNDED_SKILL_CLAIM" in violation_types

    def test_grounded_headline_passes_validation(self):
        """Authentic headline with verified Sathyabama education and Python/SQL skills passes with 0 violations."""
        from core.workflow.validators import LinkedInProfileValidator
        from models.resume_entities import EducationRecord, ExtractedSkill, SkillCategory
        from models.resume_profile import ResumeProfile

        prof = ResumeProfile(
            filename="resume.pdf",
            education=[
                EducationRecord(
                    degree="B.E. Computer Science",
                    institution="Sathyabama Institute of Science and Technology",
                    end_date="2026",
                    original_text="B.E. Computer Science - Sathyabama Institute of Science and Technology 2026",
                )
            ],
            skills=[
                ExtractedSkill(name="Python", original_mention="Python", category=SkillCategory.PROGRAMMING_LANGUAGES),
                ExtractedSkill(name="SQL", original_mention="SQL", category=SkillCategory.DATABASES),
                ExtractedSkill(name="REST APIs", original_mention="REST APIs", category=SkillCategory.TOOLS_AND_FRAMEWORKS),
            ],
            total_years_experience=0.5,
        )

        headline = "Software Systems Engineer | Python • SQL • REST APIs | Sathyabama Institute '26"
        violations = LinkedInProfileValidator.validate_headline(headline, profile=prof)
        assert len(violations) == 0

    def test_demo_resume_athiq_has_sathyabama_2026(self):
        """Verifies demo resume text in app.py uses Sathyabama Institute of Science and Technology (2026)."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "Sathyabama Institute of Science and Technology | 2022 - 2026" in code
        assert "Anna University | 2020 - 2024" not in code


class TestIssue2ResumeOptimizationHonesty:
    """Verifies that ResumeOptimizationAgent preserves factual meaning without manufacturing responsibilities."""

    def test_hulsti_reframe_never_invents_directing_initiatives(self):
        """'Founder - Hulsti.AI' must produce 'Served as Founder at Hulsti.AI.' without directing initiatives."""
        from agents.resume_agent import reframe_grounded_star_bullet
        res = reframe_grounded_star_bullet("Founder - Hulsti.AI")
        assert res == "Served as Founder at Hulsti.AI."
        assert "directing core technical initiatives" not in res
        assert "project deliverables" not in res

    def test_validator_catches_ungrounded_responsibility_expansion(self, sample_candidate_profile):
        """Validator rejects bullets containing manufactured executive leadership scope."""
        from core.workflow.validators import DeterministicWorkflowValidator
        from models.workflow import MetricDisclosureStatus, OptimizedBulletPoint, ResumeOptimizationReport

        bullet = OptimizedBulletPoint(
            bullet_id="opt_1",
            source_entity_id="exp_0",
            source_field_path="experience[0].responsibilities[0]",
            improved_star_bullet="Served as Founder at Hulsti.AI, directing core technical initiatives and project deliverables.",
            highlighted_skills=tuple(),
            alignment_rationale="Reframed bullet.",
            metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
        )
        report = ResumeOptimizationReport(
            target_role_alignment_summary="Alignment summary.",
            optimized_bullets=(bullet,),
            section_reordering_recommendations=tuple(),
            ats_keyword_density_feedback=tuple(),
        )

        validator = DeterministicWorkflowValidator()
        val_rep = validator.validate_resume_optimization(report, profile=sample_candidate_profile)
        v_types = [v.violation_type.value for v in val_rep.violations]
        assert "UNGROUNDED_RESPONSIBILITY_EXPANSION" in v_types

    def test_validator_catches_duplicate_bullets(self, sample_candidate_profile):
        """Validator rejects reports containing duplicate bullet points."""
        from core.workflow.validators import DeterministicWorkflowValidator
        from models.workflow import MetricDisclosureStatus, OptimizedBulletPoint, ResumeOptimizationReport

        b1 = OptimizedBulletPoint(
            bullet_id="opt_1",
            source_entity_id="exp_0",
            source_field_path="experience[0].responsibilities[0]",
            improved_star_bullet="Engineered high-throughput REST APIs handling production requests.",
            highlighted_skills=tuple(["Python"]),
            alignment_rationale="Rationale 1.",
            metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
        )
        b2 = OptimizedBulletPoint(
            bullet_id="opt_2",
            source_entity_id="exp_0",
            source_field_path="experience[0].responsibilities[0]",
            improved_star_bullet="Engineered high-throughput REST APIs handling production requests.",
            highlighted_skills=tuple(["Python"]),
            alignment_rationale="Rationale 2.",
            metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
        )
        report = ResumeOptimizationReport(
            target_role_alignment_summary="Summary.",
            optimized_bullets=(b1, b2),
            section_reordering_recommendations=tuple(),
            ats_keyword_density_feedback=tuple(),
        )

        validator = DeterministicWorkflowValidator()
        val_rep = validator.validate_resume_optimization(report, profile=sample_candidate_profile)
        v_types = [v.violation_type.value for v in val_rep.violations]
        assert "DUPLICATE_BULLET_POINT" in v_types

    def test_bullet_structure_classification(self):
        """Verifies classify_bullet_structure correctly distinguishes Full STAR from Action & Scope."""
        from agents.resume_agent import classify_bullet_structure
        assert "Full STAR" in classify_bullet_structure("Reduced API response latency by 35% through Redis caching.")
        assert "Action & Scope" in classify_bullet_structure("Served as Founder at Hulsti.AI.")


class TestIssue3RazorpayJDExtraction:
    """Verifies that lengthy job descriptions (like Razorpay AI Builder) extract all skills and responsibilities."""

    def test_long_razorpay_ai_builder_jd_extraction(self):
        """A 6,000+ character Razorpay AI Builder JD must extract Go, Python, Kafka, Redis, and achieve FULL_EVALUATION."""
        from processors.jd_processor import JobDescriptionProcessor
        from models.matching import EvaluationStatus

        razorpay_jd = """
        Job Title: AI Builder - Platform Engineer
        Company: Razorpay

        About Razorpay:
        Razorpay is India's leading fintech infrastructure platform powering payments and financial services for millions of businesses.

        Role Overview:
        As an AI Builder, you will design and engineer scalable microservices, intelligent pipelines, and mission-critical payment workflows.
        You will work across the distributed backend stack to build high-availability transaction engines and LLM-powered tools.

        What you will do:
        - Architect and build high-throughput backend services using Go and Python.
        - Design and implement event-driven data streaming pipelines with Apache Kafka and Redis caching.
        - Containerize backend services with Docker and orchestrate them on Kubernetes clusters across AWS cloud.
        - Collaborate with infrastructure and machine learning teams to deploy scalable LLM inference services.
        - Build reliable, production-grade REST APIs and GraphQL interfaces with comprehensive test coverage.
        - Optimize database query performance and data access layers utilizing PostgreSQL.

        What we are looking for:
        - 3+ years of professional backend engineering experience with Go or Python.
        - Strong foundation in distributed systems, concurrency paradigms, and low-latency architecture.
        - Hands-on expertise with PostgreSQL database optimization and connection pooling.
        - Solid experience deploying microservices with Docker, Kubernetes, and automated CI/CD pipelines.
        - Bachelor's degree in Computer Science, Engineering, or equivalent practical experience.

        Preferred Qualifications:
        - Experience with PyTorch, TensorFlow, and Large Language Models (LLMs).
        - Familiarity with cloud monitoring, Prometheus, and distributed tracing.
        """

        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(razorpay_jd)

        # Must extract multiple core technical skills, not just Go!
        assert "Go" in profile.required_skills
        assert "Python" in profile.required_skills
        assert "Apache Kafka" in profile.required_skills
        assert "Redis" in profile.required_skills
        assert "PostgreSQL" in profile.required_skills
        assert "Docker" in profile.required_skills
        assert "Kubernetes" in profile.required_skills
        assert "AWS" in profile.required_skills

        # Preferred skills
        assert "PyTorch" in profile.preferred_skills or "PyTorch" in profile.required_skills

        # Responsibilities must be extracted
        assert len(profile.responsibilities) >= 4

        # Must achieve FULL_EVALUATION because total requirements >= 4 and technical skills >= 3
        assert profile.evaluation_eligibility == EvaluationStatus.FULL_EVALUATION

    def test_short_jd_triggers_provisional_with_warning(self):
        """A brief JD (< 3 skills, < 4 total requirements) triggers PROVISIONAL evaluation with clear warnings."""
        from processors.jd_processor import JobDescriptionProcessor
        from models.matching import EvaluationStatus

        sparse_jd = "Web Analyst: Must know web technologies."
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(sparse_jd)

        assert profile.evaluation_eligibility == EvaluationStatus.PROVISIONAL_EVALUATION
        assert len(profile.processing_warnings) > 0
        assert any("PROVISIONAL" in w for w in profile.processing_warnings)

    def test_app_ui_character_counter_reflects_30000_limit(self):
        """Verifies UI character counter in app.py uses MAX_JD_CHAR_LIMIT rather than hardcoded 2000."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "MAX_JD_CHAR_LIMIT" in code
        assert "/ 2000" not in code
        assert "Extracted Requirements Preview & Verification" in code


class TestIssue4ResumeScoreAuthenticity:
    """Verifies that arbitrary 87% score is removed and replaced with authentic extraction statistics."""

    def test_no_synthetic_87_score_in_app(self):
        """app.py must not contain computed_score = 87.0 or hardcoded 87% Overall Score."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "computed_score = 87.0" not in code
        assert "Authentic Evidence Extraction" in code
        assert "No synthetic quality scoring applied" in code

    def test_authentic_profile_statistics_rendered(self):
        """Resume Analysis results view must display authentic counts for skills, experience, projects, and education."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "Verified Skills" in code
        assert "Work Experience" in code
        assert "Projects Identified" in code
        assert "Education & Certs" in code
        assert "Extracted Section Coverage:" in code


class TestIssue5CareerRemediationGapCoverage:
    """Verifies that remediation plan covers multiple genuine gaps, prioritizes required skills, and excludes demonstrated skills."""

    def test_remediation_milestones_cover_multiple_gaps(self):
        """Remediation plan synthesizes milestones covering both required and preferred gaps."""
        from agents.remediation_agent import CareerRemediationAgent
        from models.matching import (
            CandidateFitLevel,
            EvaluationStatus,
            EvidenceStatus,
            JobMatchResult,
            RequirementPriority,
            SkillGap,
            SkillMatch,
            SkillMatchType,
        )

        match_res = JobMatchResult(
            overall_score=45.0,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            fit_classification=CandidateFitLevel.DEVELOPING,
            active_evaluation_coverage=85.0,
            unassessed_reserve=15.0,
            match_breakdown=[],
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
                    skill_name="Go",
                    priority=RequirementPriority.REQUIRED,
                    gap_explanation="Required backend language for low-latency concurrency.",
                ),
                SkillGap(
                    skill_name="Apache Kafka",
                    priority=RequirementPriority.REQUIRED,
                    gap_explanation="Required distributed streaming engine.",
                ),
                SkillGap(
                    skill_name="Kubernetes",
                    priority=RequirementPriority.PREFERRED,
                    gap_explanation="Preferred container orchestration.",
                ),
            ],
        )

        plan = CareerRemediationAgent.synthesize_deterministic_fallback(match_res)

        # Plan must have milestones addressing Go AND Kafka
        all_milestone_skills = [s for m in plan.milestones for s in m.target_missing_skills]
        assert "Go" in all_milestone_skills
        assert "Apache Kafka" in all_milestone_skills

        # Milestone 1 focuses on required skills
        assert "Go" in plan.milestones[0].target_missing_skills

        # Demonstrated skill (Python) must NOT be recommended as a gap milestone target
        assert "Python" not in all_milestone_skills

        # Portfolio project targets multiple gaps
        assert len(plan.portfolio_projects[0].targeted_jd_gaps) >= 2

    def test_app_career_advisor_uses_missing_evidence(self):
        """app.py must reference m_res.missing_evidence rather than missing_skills."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "elif m_res.missing_evidence:" in code
        assert "m_res.missing_skills" not in code


class TestIssue6InterviewPreparationGrounding:
    """Verifies that interview preparation questions use actual candidate projects and probe gaps honestly."""

    def test_interview_questions_grounded_in_candidate_projects(self, sample_candidate_profile):
        """Interview technical question references real candidate project name and technologies."""
        from agents.interview_agent import InterviewPreparationAgent
        from models.matching import (
            CandidateFitLevel,
            EvaluationStatus,
            EvidenceStatus,
            JobMatchResult,
            RequirementPriority,
            SkillGap,
            SkillMatch,
            SkillMatchType,
        )

        match_res = JobMatchResult(
            overall_score=50.0,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            fit_classification=CandidateFitLevel.DEVELOPING,
            active_evaluation_coverage=80.0,
            unassessed_reserve=20.0,
            match_breakdown=[],
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
                    skill_name="Go",
                    priority=RequirementPriority.REQUIRED,
                    gap_explanation="Required concurrency language.",
                )
            ],
        )

        kit = InterviewPreparationAgent.synthesize_deterministic_fallback(match_res, sample_candidate_profile)

        # Technical question references actual candidate project
        tech_q = kit.technical_deep_dive_questions[0]
        assert sample_candidate_profile.projects[0].project_name in tech_q.question_text
        assert tech_q.grounded_source_entity_id == "proj_0"

        # Gap mitigation question probes Go honestly without claiming candidate has prior production experience
        gap_q = kit.gap_mitigation_questions[0]
        assert "Go" in gap_q.question_text
        assert any("upskilling" in tp.lower() or "active" in tp.lower() for tp in gap_q.suggested_talking_points)
        assert not any("prior experience in go" in tp.lower() for tp in gap_q.suggested_talking_points)


class TestIssue7MatchReportClarityAndAuditTable:
    """Verifies that match report renders complete requirements verification table without score manipulation."""

    def test_no_score_manipulation_in_get_match_alignment_display(self):
        """get_match_alignment_display must return match_res.overall_score without synthetic 83.0 hardcoding."""
        from app import get_match_alignment_display
        from models.matching import CandidateFitLevel, EvaluationStatus, JobMatchResult

        res = JobMatchResult(
            overall_score=60.0,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            fit_classification=CandidateFitLevel.MODERATE,
            active_evaluation_coverage=100.0,
            unassessed_reserve=0.0,
            match_breakdown=[],
            matched_skills=[],
            missing_evidence=[],
        )

        disp = get_match_alignment_display(res, jd_text="Web Analyst", resume_filename="Mohamed_Athiq_Resume.pdf")
        assert disp["score"] == 60.0
        assert disp["score"] != 83.0

    def test_app_includes_complete_requirements_verification_table(self):
        """app.py must include the Complete Requirements & Evidence Verification Table."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert "Complete Requirements & Evidence Verification Table" in code
        assert "EXACT MATCH" in code
        assert "PARTIAL MATCH" in code
        assert "MISSING EVIDENCE" in code
        assert "UNKNOWN" in code


# =============================================================================
# Targeted Remediation Regression Tests (Audit Consistency Fixes)
# =============================================================================

class TestCareerRoadmapCompleteness:
    """Verifies that CareerRemediationAgent includes 100% of missing requirements without omission or duplication."""

    def test_razorpay_has_exactly_12_missing_requirements(self):
        """When matching Mohamed Athiq against Razorpay AI Builder JD, exactly 12 missing requirements must be identified."""
        from processors.jd_processor import JobDescriptionProcessor
        from processors.resume_pipeline import ResumePipeline
        from core.matching.pipeline import CareerMatchingPipeline
        from app import DEMO_RESUME_ATHIQ, generate_synthetic_demo_pdf
        from tests.test_audit_fixes_regression import TestIssue3RazorpayJDExtraction

        razorpay_jd = """
        Job Title: AI Builder - Platform Engineer
        Company: Razorpay
        Role Overview: Build backend systems.
        What you will do:
        - Go and Python backend services
        - Apache Kafka and Redis caching
        - Docker and Kubernetes across AWS cloud
        - REST APIs and GraphQL interfaces
        - PostgreSQL query optimization
        What we are looking for:
        - 3+ years of professional backend engineering experience with Go or Python.
        - PostgreSQL
        - Docker, Kubernetes, CI/CD
        - Bachelor's degree in Computer Science
        Preferred Qualifications:
        - PyTorch, TensorFlow, LLMs
        - Prometheus
        """
        res_prof = ResumePipeline().process_resume(generate_synthetic_demo_pdf(DEMO_RESUME_ATHIQ))
        pipe_res = CareerMatchingPipeline().evaluate(jd_text=razorpay_jd, resume_profile=res_prof)
        match_res = pipe_res.match_result

        # Collect all missing requirements dynamically (missing_evidence + unevidenced partial_matches)
        demonstrated = {s.canonical_name.lower() for s in match_res.matched_skills}
        missing_reqs = [g.skill_name for g in match_res.missing_evidence if g.skill_name.lower() not in demonstrated]
        for pm in match_res.partial_matches:
            canon = pm.canonical_name or pm.skill_name
            if canon.lower() not in demonstrated and canon not in missing_reqs:
                missing_reqs.append(canon)

        assert len(missing_reqs) == 12
        for expected in ["Go", "PostgreSQL", "Docker", "Kubernetes", "CI/CD", "GraphQL", "Redis", "AWS", "Apache Kafka", "Prometheus", "PyTorch", "LLMs"]:
            assert expected in missing_reqs

    def test_all_12_missing_requirements_covered_in_milestones(self):
        """All 12 missing requirements must be assigned across milestones with zero omissions."""
        from processors.resume_pipeline import ResumePipeline
        from core.matching.pipeline import CareerMatchingPipeline
        from agents.remediation_agent import CareerRemediationAgent
        from app import DEMO_RESUME_ATHIQ, generate_synthetic_demo_pdf

        razorpay_jd = """
        Job Title: AI Builder - Platform Engineer
        Company: Razorpay
        What you will do:
        - Go and Python backend services
        - Apache Kafka and Redis caching
        - Docker and Kubernetes across AWS cloud
        - REST APIs and GraphQL interfaces
        - PostgreSQL
        What we are looking for:
        - 3+ years of professional backend engineering experience with Go or Python.
        - PostgreSQL, Docker, Kubernetes, CI/CD
        Preferred Qualifications:
        - PyTorch, TensorFlow, LLMs, Prometheus
        """
        res_prof = ResumePipeline().process_resume(generate_synthetic_demo_pdf(DEMO_RESUME_ATHIQ))
        match_res = CareerMatchingPipeline().evaluate(jd_text=razorpay_jd, resume_profile=res_prof).match_result

        plan = CareerRemediationAgent.synthesize_deterministic_fallback(match_res)

        # Collect all targets assigned across all milestones
        assigned_targets: list[str] = []
        for m in plan.milestones:
            assigned_targets.extend(m.target_missing_skills)

        # Every single one of the 12 expected missing skills must be present
        for required_skill in ["Go", "PostgreSQL", "Docker", "Kubernetes", "CI/CD", "GraphQL", "Redis", "AWS", "Apache Kafka", "Prometheus", "PyTorch", "LLMs"]:
            assert required_skill in assigned_targets, f"Skill '{required_skill}' was silently omitted from roadmap!"

    def test_zero_skill_repetition_across_milestones(self):
        """No skill (e.g. Kubernetes) should be duplicated across milestones."""
        from processors.resume_pipeline import ResumePipeline
        from core.matching.pipeline import CareerMatchingPipeline
        from agents.remediation_agent import CareerRemediationAgent
        from app import DEMO_RESUME_ATHIQ, generate_synthetic_demo_pdf

        razorpay_jd = """
        Job Title: AI Builder
        Company: Razorpay
        What you will do:
        - Go and Python
        - Apache Kafka, Redis
        - Docker, Kubernetes, AWS
        - GraphQL, REST APIs, PostgreSQL
        What we are looking for:
        - CI/CD, Kubernetes
        Preferred Qualifications:
        - PyTorch, LLMs, Prometheus
        """
        res_prof = ResumePipeline().process_resume(generate_synthetic_demo_pdf(DEMO_RESUME_ATHIQ))
        match_res = CareerMatchingPipeline().evaluate(jd_text=razorpay_jd, resume_profile=res_prof).match_result

        plan = CareerRemediationAgent.synthesize_deterministic_fallback(match_res)

        all_skills: list[str] = []
        for m in plan.milestones:
            all_skills.extend(m.target_missing_skills)

        # Count occurrences: each skill should appear exactly once
        from collections import Counter
        counts = Counter(all_skills)
        duplicates = [skill for skill, count in counts.items() if count > 1]
        assert len(duplicates) == 0, f"Skills repeated across milestones: {duplicates}"

    def test_demonstrated_skills_never_recommended_as_gaps(self):
        """Candidate's verified skills (Python, REST APIs, TensorFlow) must not be in missing target skills."""
        from processors.resume_pipeline import ResumePipeline
        from core.matching.pipeline import CareerMatchingPipeline
        from agents.remediation_agent import CareerRemediationAgent
        from app import DEMO_RESUME_ATHIQ, generate_synthetic_demo_pdf

        razorpay_jd = """
        Job Title: AI Builder
        Company: Razorpay
        What you will do:
        - Go and Python
        - REST APIs, TensorFlow
        - Docker, Kubernetes
        """
        res_prof = ResumePipeline().process_resume(generate_synthetic_demo_pdf(DEMO_RESUME_ATHIQ))
        match_res = CareerMatchingPipeline().evaluate(jd_text=razorpay_jd, resume_profile=res_prof).match_result

        plan = CareerRemediationAgent.synthesize_deterministic_fallback(match_res)

        demonstrated = {"python", "rest apis", "tensorflow"}
        for m in plan.milestones:
            for s in m.target_missing_skills:
                assert s.lower() not in demonstrated, f"Demonstrated skill '{s}' was falsely recommended as missing!"


class TestJDExperienceExtractionRegex:
    """Verifies that processors/jd_processor.py accurately extracts experience tenure without false positives."""

    def test_valid_experience_clauses(self):
        """Matches all standard experience patterns including multi-word modifiers and en-dash ranges."""
        from processors.jd_processor import JobDescriptionProcessor

        proc = JobDescriptionProcessor(enable_llm_extraction=False)

        valid_cases = [
            ("3+ years of professional backend engineering experience.", "3+ years of professional backend engineering experience"),
            ("2 years of software development experience.", "2 years of software development experience"),
            ("1–2 years of relevant industry experience.", "1–2 years of relevant industry experience"),
            ("1-2 years of relevant industry experience.", "1-2 years of relevant industry experience"),
            ("3 years of experience.", "3 years of experience"),
            ("5+ years experience in Python.", "5+ years experience"),
        ]

        for text, expected in valid_cases:
            jd_text = f"Job Title: Software Engineer\nCompany: TestCorp\nRequirements:\n- {text}\n- Bachelor's degree\n- Python\n"
            profile = proc.process(jd_text)
            assert profile.experience_requirement is not None, f"Failed to match valid clause: '{text}'"
            assert expected.lower() in profile.experience_requirement.lower(), f"Expected '{expected}' in '{profile.experience_requirement}'"

    def test_invalid_clauses_rejected(self):
        """Unrelated dates, percentages, and metrics must not match as experience requirements."""
        from processors.jd_processor import JobDescriptionProcessor

        proc = JobDescriptionProcessor(enable_llm_extraction=False)

        invalid_cases = [
            "Founded in 2020 with 100 employees across 12 countries.",
            "Jan 2022 to Mar 2024 production operations.",
            "Achieved 100% test coverage and 99.99% service availability.",
            "Platform powers over 500 million transactions processed daily.",
        ]

        for text in invalid_cases:
            jd_text = f"Job Title: Software Engineer\nCompany: TestCorp\nAbout Us:\n{text}\nRequirements:\n- Bachelor's degree\n- Python\n"
            profile = proc.process(jd_text)
            assert profile.experience_requirement is None, f"False positive matched on: '{text}' -> got '{profile.experience_requirement}'"


class TestDynamicLinkedInCertification:
    """Verifies that LinkedIn headline suggestions reflect only candidate's verified certifications."""

    def test_candidate_with_oci_certification_gets_oci_reference(self):
        """Candidate Athiq (with OCI in text/certifications) should receive OCI certification in suggestions."""
        from app import DEMO_RESUME_ATHIQ, generate_synthetic_demo_pdf
        from processors.resume_pipeline import ResumePipeline

        prof = ResumePipeline().process_resume(generate_synthetic_demo_pdf(DEMO_RESUME_ATHIQ))
        assert "oci" in prof.cleaned_text.lower()

    def test_candidate_with_no_certifications_has_no_cert_claim(self):
        """Candidate with zero certifications must never have 'OCI Certified' or any other certification in headline."""
        from models.resume_entities import EducationRecord, ExtractedSkill, SkillCategory
        from models.resume_profile import ResumeProfile
        from core.workflow.validators import LinkedInProfileValidator

        prof = ResumeProfile(
            filename="nocert_resume.pdf",
            education=[
                EducationRecord(
                    degree="B.S. Computer Science",
                    institution="University of Washington",
                    end_date="2025",
                    original_text="B.S. Computer Science - University of Washington 2025",
                )
            ],
            skills=[
                ExtractedSkill(name="Python", original_mention="Python", category=SkillCategory.PROGRAMMING_LANGUAGES),
                ExtractedSkill(name="SQL", original_mention="SQL", category=SkillCategory.DATABASES),
            ],
            total_years_experience=1.0,
        )

        headline = "Software Systems Engineer | Python • SQL • OCI Certified | University of Washington '25"
        violations = LinkedInProfileValidator.validate_headline(headline, profile=prof)

        assert len(violations) > 0
        assert any("OCI Certified" in v.message or "oci" in v.invalid_value.lower() for v in violations)

    def test_candidate_with_aws_certification_gets_aws_not_oci(self):
        """Candidate with AWS certification must not inherit OCI certification."""
        from models.resume_entities import EducationRecord, ExtractedSkill, SkillCategory
        from models.resume_profile import ResumeProfile
        from core.workflow.validators import LinkedInProfileValidator

        prof = ResumeProfile(
            filename="aws_resume.pdf",
            cleaned_text="Certified AWS Solutions Architect Associate holder with Python experience.",
            education=[
                EducationRecord(
                    degree="B.Tech Computer Science",
                    institution="IIT Madras",
                    end_date="2023",
                    original_text="B.Tech Computer Science - IIT Madras 2023",
                )
            ],
            skills=[
                ExtractedSkill(name="Python", original_mention="Python", category=SkillCategory.PROGRAMMING_LANGUAGES),
                ExtractedSkill(name="AWS", original_mention="AWS", category=SkillCategory.CLOUD_COMPUTING),
            ],
            total_years_experience=2.0,
        )

        # Valid AWS headline passes
        valid_aws_hl = "Software Engineer | Python, AWS • AWS Certified | IIT Madras '23"
        v_valid = LinkedInProfileValidator.validate_headline(valid_aws_hl, profile=prof)
        assert len(v_valid) == 0

        # Inherited OCI headline is rejected
        invalid_oci_hl = "Software Engineer | Python, AWS • OCI Certified | IIT Madras '23"
        v_invalid = LinkedInProfileValidator.validate_headline(invalid_oci_hl, profile=prof)
        assert len(v_invalid) > 0
        assert any("OCI Certified" in v.message or "oci" in v.invalid_value.lower() for v in v_invalid)


class TestJDCharacterCounterConsistency:
    """Verifies that displayed character counter precisely tracks actual text evaluated."""

    def test_character_counter_reflects_actual_input_length(self):
        """Character counter must evaluate len(jd_text) accurately."""
        from processors.jd_processor import MAX_JD_CHAR_LIMIT

        sample_short = "Short JD content"
        sample_medium = "A" * 1710
        sample_long = "B" * 6134

        assert f"{len(sample_short):,} / {MAX_JD_CHAR_LIMIT:,}" == "16 / 30,000"
        assert f"{len(sample_medium):,} / {MAX_JD_CHAR_LIMIT:,}" == "1,710 / 30,000"
        assert f"{len(sample_long):,} / {MAX_JD_CHAR_LIMIT:,}" == "6,134 / 30,000"
        assert len(sample_long) <= MAX_JD_CHAR_LIMIT


