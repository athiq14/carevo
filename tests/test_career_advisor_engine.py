import re
import pytest
from unittest.mock import patch, MagicMock
from agents.career_advisor_engine import CareerAdvisorEngine, detect_role_archetype
from models.resume_profile import ResumeProfile
from models.resume_entities import ExtractedSkill, SkillCategory


def test_detect_role_archetype():
    assert detect_role_archetype("Insurance Advisor") == "INSURANCE"
    assert detect_role_archetype("Home Loan Advisor") == "MORTGAGE_LENDING"
    assert detect_role_archetype("Home Advisor") == "REAL_ESTATE_PROPERTY"
    assert detect_role_archetype("Financial Consultant") == "FINANCE"
    assert detect_role_archetype("HR Executive") == "HUMAN_RESOURCES"
    assert detect_role_archetype("Marketing Manager") == "MARKETING_SALES"
    assert detect_role_archetype("Senior Data Analyst") == "DATA_ANALYTICS"
    assert detect_role_archetype("Staff Software Engineer") == "SOFTWARE_TECH"


def test_home_loan_advisor_roadmap_governed_by_rbi_not_nism():
    engine = CareerAdvisorEngine()
    roadmap = engine.generate_roadmap("Home Loan Advisor")

    assert roadmap.archetype == "MORTGAGE_LENDING"
    corpus = roadmap.model_dump_json().lower()
    # Must reference RBI DSA guidelines and credit appraisal
    assert "rbi" in corpus
    assert "dsa" in corpus
    assert "credit" in corpus or "foir" in corpus
    # Must NOT claim NISM is mandatory for home loan advisory
    assert "nism" not in corpus or "securities and mutual funds, not retail mortgage" in corpus
    # No software engineering jargon
    assert "docker" not in corpus
    assert "kubernetes" not in corpus


def test_home_advisor_roadmap_governed_by_rera():
    engine = CareerAdvisorEngine()
    roadmap = engine.generate_roadmap("Home Advisor")

    assert roadmap.archetype == "REAL_ESTATE_PROPERTY"
    corpus = roadmap.model_dump_json().lower()
    # Must reference RERA compliance under Real Estate Act
    assert "rera" in corpus
    # Must NOT mandate NISM for property consulting
    assert "nism series" not in corpus
    # No software development jargon
    assert "docker" not in corpus


def test_app_unique_adv_launch_button_key():
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()

    matches = [m.start() for m in re.finditer(r'key="adv_launch_wf_hero_btn"', code)]
    assert len(matches) == 1, f"Expected exactly 1 adv_launch_wf_hero_btn, found {len(matches)}"


def test_app_truthful_provenance_labeling():
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()

    assert "AI-GENERATED PERSONALIZED ROADMAP" in code
    assert "[SAMPLE BENCHMARK ROADMAP" in code
    assert 'Role Specialization Clarification: "Home Advisor"' in code


def test_insurance_roadmap_has_zero_tech_jargon():
    engine = CareerAdvisorEngine()
    roadmap = engine.generate_roadmap("Insurance Advisor")

    assert roadmap.archetype == "INSURANCE"
    # Verify insurance specific milestones and certifications
    milestone_deliverables = " ".join([m["deliverable"] for m in roadmap.milestones]).lower()
    assert "ic-38" in milestone_deliverables or "insurance" in milestone_deliverables
    # Verify no software development jargon forced onto insurance
    corpus = json_str = roadmap.model_dump_json().lower()
    assert "kubernetes" not in corpus
    assert "docker" not in corpus
    assert "next.js" not in corpus


def test_data_analyst_roadmap_targets_data_stack():
    engine = CareerAdvisorEngine()
    roadmap = engine.generate_roadmap("Senior Data Analyst")

    assert roadmap.archetype == "DATA_ANALYTICS"
    corpus = roadmap.model_dump_json().lower()
    assert "sql" in corpus
    assert "tableau" in corpus or "dashboard" in corpus
    assert "insurance" not in corpus


def test_live_groq_synthesis_when_api_key_provided():
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="""{
                    "direction_narrative": "AI Engineer trajectory focusing on agentic workflows.",
                    "target_roles": [{"title": "AI Builder", "description": "0-to-1 prototyping"}],
                    "transferable_strengths": [{"title": "Python Skills", "explanation": "Strong foundation"}],
                    "skill_priorities": [{"name": "LLM Tuning", "priority": "High", "explanation": "Deep dive"}],
                    "milestones": [{"timeframe": "Month 1", "focus": "Prompt Engineering", "actions": "Study", "deliverable": "Pipeline"}],
                    "portfolio_work_samples": [{"title": "Agent Prototype", "domain": "AI", "concept": "Agent", "deliverable": "Repo"}],
                    "interview_prep": [{"type": "AI System", "question": "Explain RAG", "tip": "Mention retrieval"}]
                }"""
            )
        )
    ]

    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_cls.return_value = mock_client

        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key_12345"}):
            engine = CareerAdvisorEngine()
            roadmap = engine.generate_roadmap("AI Engineer")
            assert roadmap.provenance == "LIVE_GROQ_AI"
            assert roadmap.provider_name == "Groq"
            assert "AI Engineer trajectory" in roadmap.direction_narrative

