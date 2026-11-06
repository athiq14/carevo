"""
Unit and integration tests for the unified ResumeProfile schema and ResumePipeline orchestrator.
Verifies end-to-end PDF processing, metadata tracking, entity integration,
status transitions (SUCCESS, PARTIAL, FAILED), OCR handling, and zero fabrication.
"""
import io
import json
from datetime import datetime
import pymupdf
import pytest

from models.resume import ExtractionStatus, ResumeExtractionResult
from models.resume_sections import (
    ParsedResumeSections,
    ResumeSection,
    SectionType,
)
from processors.section_detector import CANONICAL_SECTIONS
from models.resume_entities import (
    EducationRecord,
    ExtractedSkill,
    ProjectRecord,
    ResumeEntities,
    SkillCategory,
    WorkExperience,
)
from models.resume_profile import (
    ProfileProcessingStatus,
    ResumeProfile,
)
from processors.pdf_processor import (
    EmptyFileError,
    InvalidPDFError,
    PDFProcessingError,
)
from processors.resume_pipeline import (
    ResumePipeline,
    process_resume_pipeline,
)


def create_synthetic_pdf(text_pages: list[str]) -> bytes:
    """Helper to generate in-memory PDF bytes from a list of page strings."""
    doc = pymupdf.open()
    for page_text in text_pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), page_text, fontsize=9)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def comprehensive_resume_pdf() -> bytes:
    """Fixture providing a standard multi-section PDF resume."""
    resume_text = (
        "Alex Morgan\n"
        "Email: alex.morgan@example.com | Phone: (555) 234-5678 | San Francisco, CA\n\n"
        "PROFESSIONAL SUMMARY\n"
        "Accomplished Cloud & DevOps Engineer with 6 years of expertise building resilient distributed systems.\n\n"
        "TECHNICAL SKILLS\n"
        "Programming Languages: Python, Go, TypeScript\n"
        "Databases: PostgreSQL, Redis, MongoDB\n"
        "Cloud Computing: AWS, GCP, Azure\n"
        "DevOps and Infrastructure: Docker, Kubernetes, Terraform, CI/CD\n\n"
        "WORK EXPERIENCE\n"
        "Staff Infrastructure Engineer at CloudScale Systems Inc. | January 2021 - Present\n"
        "- Architected multi-region Kubernetes clusters with automated GitOps workflows.\n"
        "- Reduced infrastructure deployment latency by 45% using Terraform and Docker.\n"
        "- Integrated Prometheus and Grafana for real-time observability across 200+ microservices.\n\n"
        "EDUCATION\n"
        "Bachelor of Science in Computer Science\n"
        "University of California, Berkeley\n"
        "2015 - 2019\n"
        "GPA: 3.85 / 4.0\n\n"
        "PROJECTS\n"
        "- Automated Cloud Provisioner: Developed open-source multi-cloud orchestration tool leveraging Python, Terraform, and Docker.\n"
        "  Repository: https://github.com/alexmorgan/cloud-provisioner\n"
    )
    return create_synthetic_pdf([resume_text])


class TestResumePipeline:
    """Tests for ResumePipeline execution, orchestrating all extraction steps."""

    def test_successful_end_to_end_pipeline(self, comprehensive_resume_pdf):
        """Verify complete pipeline execution returns SUCCESS status and populates all profile components."""
        pipeline = ResumePipeline()
        profile = pipeline.process_resume(
            file_input=comprehensive_resume_pdf,
            filename="alex_morgan_resume.pdf",
        )

        assert isinstance(profile, ResumeProfile)
        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert profile.filename == "alex_morgan_resume.pdf"
        assert profile.page_count == 1
        assert profile.raw_character_count > 200
        assert profile.cleaned_character_count > 200
        assert profile.requires_ocr is False
        assert profile.ocr_warning is None

        # Content preservation
        assert "Alex Morgan" in profile.original_text
        assert "CloudScale Systems" in profile.cleaned_text

        # Sections verification
        assert len(profile.sections) >= 5
        assert "Professional Summary" in profile.detected_section_names
        assert "Technical Skills" in profile.detected_section_names
        assert "Work Experience" in profile.detected_section_names
        assert "Education" in profile.detected_section_names
        assert "Projects" in profile.detected_section_names

        # Entities verification
        skill_names = {s.name for s in profile.skills}
        assert "Python" in skill_names
        assert "Kubernetes" in skill_names
        assert "Docker" in skill_names
        assert "PostgreSQL" in skill_names

        # Experience verification
        assert len(profile.experience) >= 1
        exp = profile.experience[0]
        assert exp.job_title == "Staff Infrastructure Engineer"
        assert exp.company == "CloudScale Systems Inc."
        assert exp.employment_start == "January 2021"
        assert exp.employment_end == "Present"
        assert len(exp.responsibilities) >= 2

        # Education verification
        assert len(profile.education) >= 1
        edu = profile.education[0]
        assert "Bachelor of Science" in edu.degree
        assert "University of California" in edu.institution
        assert "3.85" in (edu.grade_or_cgpa or "")

        # Projects verification
        assert len(profile.projects) >= 1
        proj = profile.projects[0]
        assert "Automated Cloud Provisioner" in proj.project_name
        assert proj.project_url == "https://github.com/alexmorgan/cloud-provisioner"

    def test_unified_profile_structure_and_serialization(self, comprehensive_resume_pdf):
        """Verify profile serialization to dictionary and JSON format with all metadata."""
        profile = process_resume_pipeline(comprehensive_resume_pdf, filename="test.pdf")

        # Pydantic dump dict
        data_dict = profile.model_dump()
        assert data_dict["filename"] == "test.pdf"
        assert data_dict["status"] == "SUCCESS"
        assert isinstance(data_dict["skills"], list)
        assert isinstance(data_dict["experience"], list)
        assert isinstance(data_dict["education"], list)
        assert isinstance(data_dict["projects"], list)
        assert isinstance(data_dict["sections"], list)

        # JSON serialization and direct string validation via model_validate_json()
        json_str = profile.model_dump_json()
        validated_profile = ResumeProfile.model_validate_json(json_str)
        assert validated_profile.filename == "test.pdf"
        assert validated_profile.status == ProfileProcessingStatus.SUCCESS
        assert len(validated_profile.skills) == len(profile.skills)
        assert validated_profile == profile

    def test_metadata_preservation(self, comprehensive_resume_pdf):
        """Verify all extraction metadata fields are properly populated and consistent."""
        pipeline = ResumePipeline()
        profile = pipeline.process_resume(comprehensive_resume_pdf, filename="candidate.pdf")

        assert profile.filename == "candidate.pdf"
        assert profile.page_count == 1
        assert profile.raw_character_count == len(profile.original_text)
        assert profile.cleaned_character_count == len(profile.cleaned_text)
        assert isinstance(profile.processed_at, datetime)

    def test_text_preservation(self, comprehensive_resume_pdf):
        """Verify both original text and cleaned text are preserved separately."""
        pipeline = ResumePipeline()
        profile = pipeline.process_resume(comprehensive_resume_pdf)

        assert len(profile.original_text) > 0
        assert len(profile.cleaned_text) > 0
        assert profile.original_text is not profile.cleaned_text

    def test_sections_and_entities_integration(self, comprehensive_resume_pdf):
        """Verify helper queries and integration of sections with entities."""
        pipeline = ResumePipeline()
        profile = pipeline.process_resume(comprehensive_resume_pdf)

        # Section querying
        skills_sec = profile.get_section_by_type(SectionType.TECHNICAL_SKILLS)
        assert skills_sec is not None
        assert "Python" in skills_sec.content

        # Skill category grouping helper
        skills_by_cat = profile.get_skills_by_category()
        assert SkillCategory.PROGRAMMING_LANGUAGES in skills_by_cat
        assert any(s.name == "Python" for s in skills_by_cat[SkillCategory.PROGRAMMING_LANGUAGES])

    def test_missing_fields_remain_none(self):
        """Verify that absent fields in resume are not fabricated and remain None."""
        resume_sparse = (
            "Alex Smith\n\n"
            "TECHNICAL SKILLS\n"
            "Python, SQL\n\n"
            "WORK EXPERIENCE\n"
            "Software Engineer\n"
            "InnovateTech\n"
            "• Implemented backend data service.\n\n"
            "EDUCATION\n"
            "Bachelor of Science in Computer Science\n"
            "MIT\n\n"
            "PROJECTS\n"
            "Web Analytics Portal\n"
            "Built analytics reporting engine using Python.\n"
        )
        pdf_bytes = create_synthetic_pdf([resume_sparse])
        profile = process_resume_pipeline(pdf_bytes, filename="sparse.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS

        # Experience dates absent -> must be None
        assert len(profile.experience) >= 1
        exp = profile.experience[0]
        assert exp.employment_start is None
        assert exp.employment_end is None
        assert exp.employment_type is None

        # Education dates and GPA absent -> must be None
        assert len(profile.education) >= 1
        edu = profile.education[0]
        assert edu.start_date is None
        assert edu.end_date is None
        assert edu.grade_or_cgpa is None

        # Project URL absent -> must be None
        assert len(profile.projects) >= 1
        proj = profile.projects[0]
        assert proj.project_url is None

    def test_ambiguity_survival_in_profile(self):
        """Verify ambiguous entity detections survive into profile with ambiguity indicators."""
        resume_ambiguous = (
            "Jane Doe\n\n"
            "TECHNICAL SKILLS\n"
            "Python, AWS\n\n"
            "WORK EXPERIENCE\n"
            "• Worked on backend distributed system without explicit job title or employer header.\n\n"
            "PROJECTS\n"
            "Vague Initiative\n"
            "Investigated experimental machine learning techniques.\n"
        )
        pdf_bytes = create_synthetic_pdf([resume_ambiguous])
        profile = process_resume_pipeline(pdf_bytes, filename="ambiguous.pdf")

        # Entity ambiguity preserved
        assert profile.has_ambiguities is True
        assert any("ambiguous or incomplete" in w.lower() for w in profile.warnings)

    def test_ocr_required_handling(self):
        """Verify scanned or low-text PDFs set PARTIAL status and populate ocr_warning."""
        # PDF with text below 80 characters triggers OCR recommendation
        scanned_pdf = create_synthetic_pdf(["Scanned Page"])
        profile = process_resume_pipeline(scanned_pdf, filename="scanned.pdf")

        assert profile.status == ProfileProcessingStatus.PARTIAL
        assert profile.requires_ocr is True
        assert profile.ocr_warning is not None
        assert "OCR" in profile.ocr_warning
        assert any("OCR" in w for w in profile.warnings)

    def test_invalid_pdf_bytes_rejected(self):
        """Verify invalid PDF bytes produce FAILED status or raise InvalidPDFError."""
        invalid_bytes = b"NOT_A_VALID_PDF_STREAM"

        # Safe mode returns FAILED profile
        profile = process_resume_pipeline(
            invalid_bytes,
            filename="invalid.pdf",
            raise_on_error=False,
        )
        assert profile.status == ProfileProcessingStatus.FAILED
        assert profile.page_count == 0
        assert len(profile.skills) == 0
        assert len(profile.experience) == 0
        assert len(profile.warnings) >= 1
        assert "Invalid PDF" in profile.warnings[0] or "signature" in profile.warnings[0].lower()

        # Strict mode raises exception
        with pytest.raises(InvalidPDFError):
            process_resume_pipeline(
                invalid_bytes,
                filename="invalid.pdf",
                raise_on_error=True,
            )

    def test_empty_file_rejected(self):
        """Verify empty 0-byte input produces FAILED status or raises EmptyFileError."""
        empty_bytes = b""

        # Safe mode
        profile = process_resume_pipeline(
            empty_bytes,
            filename="empty.pdf",
            raise_on_error=False,
        )
        assert profile.status == ProfileProcessingStatus.FAILED
        assert profile.page_count == 0
        assert len(profile.warnings) >= 1
        assert "empty" in profile.warnings[0].lower()

        # Strict mode
        with pytest.raises(EmptyFileError):
            process_resume_pipeline(
                empty_bytes,
                filename="empty.pdf",
                raise_on_error=True,
            )

    def test_stream_binary_input(self, comprehensive_resume_pdf):
        """Verify io.BytesIO stream is correctly processed by pipeline."""
        stream = io.BytesIO(comprehensive_resume_pdf)
        profile = process_resume_pipeline(stream, filename="stream_resume.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert profile.filename == "stream_resume.pdf"
        assert len(profile.skills) > 0


class TestResumeProfileModel:
    """Direct unit tests for ResumeProfile model and from_components factory."""

    def test_from_components_factory_success(self):
        """Verify from_components correctly combines disparate model instances."""
        pdf_res = ResumeExtractionResult(
            filename="manual.pdf",
            page_count=2,
            extracted_text="Sample text content for resume.",
            character_count=31,
            extraction_status=ExtractionStatus.SUCCESS,
            requires_ocr=False,
        )
        parsed_secs = ParsedResumeSections(
            original_text=pdf_res.extracted_text,
            cleaned_text="Sample text content for resume.",
            sections=[
                ResumeSection(
                    section_type=SectionType.TECHNICAL_SKILLS,
                    original_heading="SKILLS",
                    content="Python, Docker",
                    start_line=1,
                    end_line=1,
                )
            ],
            detected_section_names=["Technical Skills"],
            undetected_section_names=["Education"],
        )
        entities = ResumeEntities(
            skills=[
                ExtractedSkill(
                    name="Python",
                    original_mention="Python",
                    category=SkillCategory.PROGRAMMING_LANGUAGES,
                )
            ]
        )

        profile = ResumeProfile.from_components(
            pdf_result=pdf_res,
            cleaned_text=parsed_secs.cleaned_text,
            parsed_sections=parsed_secs,
            entities=entities,
        )

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert profile.filename == "manual.pdf"
        assert profile.page_count == 2
        assert len(profile.skills) == 1
        assert profile.skills[0].name == "Python"
        assert profile.requires_ocr is False

    def test_from_components_factory_ocr_partial(self):
        """Verify from_components marks PARTIAL and populates warnings when OCR is needed."""
        pdf_res = ResumeExtractionResult(
            filename="ocr_manual.pdf",
            page_count=1,
            extracted_text="Few chars",
            character_count=9,
            extraction_status=ExtractionStatus.WARNING,
            requires_ocr=True,
            error_message="Insufficient text. OCR needed.",
        )
        parsed_secs = ParsedResumeSections(
            original_text="Few chars",
            cleaned_text="Few chars",
            sections=[],
            detected_section_names=[],
            undetected_section_names=[],
        )
        entities = ResumeEntities()

        profile = ResumeProfile.from_components(
            pdf_result=pdf_res,
            cleaned_text=parsed_secs.cleaned_text,
            parsed_sections=parsed_secs,
            entities=entities,
        )

        assert profile.status == ProfileProcessingStatus.PARTIAL
        assert profile.requires_ocr is True
        assert profile.ocr_warning == "Insufficient text. OCR needed."
        assert "Insufficient text. OCR needed." in profile.warnings
