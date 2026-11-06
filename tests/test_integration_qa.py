"""
Phase 2.5 Comprehensive Integration Test Suite & Quality Assurance.
Validates the end-to-end resume processing pipeline from PDF upload stream
to final immutable ResumeProfile across format variations, edge cases,
data integrity constraints, and privacy guarantees.
"""
import io
import json
import os
import tempfile
import pymupdf
import pytest

from config import AppConfig, load_config
from models.resume import ExtractionStatus
from models.resume_sections import SectionType
from models.resume_entities import SkillCategory
from models.resume_profile import ProfileProcessingStatus, ResumeProfile
from processors.pdf_processor import (
    CorruptedPDFError,
    EmptyFileError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFProcessingError,
    PDFProcessor,
)
from processors.resume_pipeline import ResumePipeline, process_resume_pipeline


# ---------------------------------------------------------------------------
# Synthetic PDF Generation Helpers
# ---------------------------------------------------------------------------

def make_pdf(page_texts: list[str], password: str | None = None) -> bytes:
    """Generate in-memory synthetic PDF bytes with optional encryption."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=9)
    if password:
        # Encrypt document with standard permissions
        perm = int(
            pymupdf.PDF_PERM_ACCESSIBILITY
            | pymupdf.PDF_PERM_PRINT
            | pymupdf.PDF_PERM_COPY
            | pymupdf.PDF_PERM_ANNOTATE
        )
        pdf_bytes = doc.tobytes(
            deflate=True,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw=password,
            user_pw=password,
            permissions=perm,
        )
    else:
        pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# 1. Resume Format Variations
# ---------------------------------------------------------------------------

class TestResumeFormatVariations:
    """Tests covering various resume structural layouts, ordering, and conventions."""

    def test_standard_chronological_resume(self):
        """Standard layout: Contact -> Summary -> Skills -> Experience -> Education -> Projects."""
        text = (
            "Marcus Aurelius\n"
            "marcus@example.com | (555) 123-4567 | Austin, TX\n\n"
            "PROFESSIONAL SUMMARY\n"
            "Staff Platform Engineer with 8 years of experience scaling Kubernetes infrastructure.\n\n"
            "TECHNICAL SKILLS\n"
            "Programming Languages: Python, Go\n"
            "Cloud Computing: AWS, GCP\n"
            "DevOps and Infrastructure: Docker, Kubernetes, Terraform\n\n"
            "WORK EXPERIENCE\n"
            "Lead Systems Architect at Apex Cloud Solutions | 2021 - Present\n"
            "- Spearheaded migration of 150+ microservices to Kubernetes clusters.\n"
            "- Reduced AWS operational costs by 32% via autoscaling policies.\n\n"
            "EDUCATION\n"
            "Bachelor of Science in Computer Science\n"
            "University of Texas at Austin | 2013 - 2017\n"
            "GPA: 3.88 / 4.0\n\n"
            "PROJECTS\n"
            "- Distributed Cache Orchestrator: Built high-throughput caching tier in Go and Redis.\n"
            "  https://github.com/marcus/cache-orchestrator\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="standard.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert profile.page_count == 1
        assert "Marcus Aurelius" in profile.original_text
        assert len(profile.sections) >= 5
        assert len(profile.skills) >= 5
        assert len(profile.experience) == 1
        assert len(profile.education) == 1
        assert len(profile.projects) == 1
        assert profile.education[0].institution == "University of Texas at Austin"
        assert profile.projects[0].project_url == "https://github.com/marcus/cache-orchestrator"

    def test_different_section_ordering(self):
        """Education and Projects preceding Experience and Skills."""
        text = (
            "Sarah Connor\n"
            "sarah@cyberdyne.org | (555) 999-0000\n\n"
            "EDUCATION\n"
            "Master of Science in Robotics\n"
            "Carnegie Mellon University | 2018 - 2020\n\n"
            "PROJECTS\n"
            "- Autonomous Navigation Rover: Developed SLAM navigation package with ROS and Python.\n"
            "  Repository: https://github.com/sarah/rover-slam\n\n"
            "TECHNICAL SKILLS\n"
            "Python, C++, ROS, Docker, Linux\n\n"
            "WORK EXPERIENCE\n"
            "Robotics Software Engineer at Boston Dynamics | 2020 - Present\n"
            "- Implemented real-time path planning in C++ and Python."
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="inverted_order.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert "Education" in profile.detected_section_names
        assert "Projects" in profile.detected_section_names
        assert "Technical Skills" in profile.detected_section_names
        assert "Work Experience" in profile.detected_section_names
        assert len(profile.education) == 1
        assert profile.education[0].institution == "Carnegie Mellon University"
        assert len(profile.projects) == 1
        assert len(profile.experience) == 1

    def test_missing_sections(self):
        """Resume missing Projects and Summary sections (only Skills + Education)."""
        text = (
            "David Miller\n"
            "david@example.com | San Francisco, CA\n\n"
            "TECHNICAL SKILLS\n"
            "Programming Languages: Java, Kotlin, SQL\n"
            "Databases: PostgreSQL, Redis\n\n"
            "EDUCATION\n"
            "Bachelor of Science in Information Systems\n"
            "San Jose State University | 2016 - 2020\n"
            "GPA: 3.75 / 4.0\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="sparse_sections.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert "Technical Skills" in profile.detected_section_names
        assert "Education" in profile.detected_section_names
        assert "Work Experience" in profile.undetected_section_names
        assert "Projects" in profile.undetected_section_names
        assert len(profile.experience) == 0
        assert len(profile.projects) == 0
        assert len(profile.skills) >= 4

    def test_multiple_education_records(self):
        """Multiple academic credentials (B.S., M.S., Ph.D.)."""
        text = (
            "Dr. Elena Rostova\n"
            "elena@research.edu\n\n"
            "EDUCATION\n"
            "Doctor of Philosophy in Computer Science\n"
            "Stanford University | 2019 - 2023\n\n"
            "Master of Science in Applied Mathematics\n"
            "MIT | 2017 - 2019\n\n"
            "Bachelor of Science in Physics\n"
            "Harvard University | 2013 - 2017\n\n"
            "TECHNICAL SKILLS\n"
            "Python, PyTorch, C++, Machine Learning, Data Science"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="multi_edu.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert len(profile.education) == 3
        degrees = [e.degree for e in profile.education]
        insts = [e.institution for e in profile.education]
        assert any("Doctor of Philosophy" in d for d in degrees)
        assert any("Master of Science" in d for d in degrees)
        assert any("Bachelor of Science" in d for d in degrees)
        assert "Stanford University" in insts
        assert "MIT" in insts
        assert "Harvard University" in insts

    def test_multiple_employment_records(self):
        """Resume with 3 sequential employment records with dates and tech."""
        text = (
            "Jordan Lee\n"
            "jordan@tech.io\n\n"
            "TECHNICAL SKILLS\n"
            "Go, Python, Docker, Kubernetes, AWS\n\n"
            "WORK EXPERIENCE\n"
            "Principal Software Engineer at Stripe | 2022 - Present\n"
            "- Designed distributed ledger processing pipeline handling $50M/day.\n\n"
            "Senior Backend Engineer at Twilio | 2019 - 2022\n"
            "- Built high-throughput SMS dispatch service in Go.\n\n"
            "Software Engineer at Box | 2017 - 2019\n"
            "- Developed REST APIs in Python and PostgreSQL.\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="multi_jobs.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert len(profile.experience) == 3
        companies = [e.company for e in profile.experience]
        assert "Stripe" in companies
        assert "Twilio" in companies
        assert "Box" in companies

    def test_multiple_projects(self):
        """Resume with multiple distinct engineering projects and repository URLs."""
        text = (
            "Lucas Thorne\n"
            "lucas@github.io\n\n"
            "TECHNICAL SKILLS\n"
            "Rust, Go, Python, Docker\n\n"
            "PROJECTS\n"
            "- Raft Consensus Engine: High-performance replication engine written in Rust.\n"
            "  Repository: https://github.com/lucas/raft-rs\n\n"
            "- FastKV: Distributed in-memory storage node in Go and Docker.\n"
            "  URL: https://github.com/lucas/fastkv\n\n"
            "- AutoScaler Daemon: Kubernetes controller for dynamic pod autoscaling.\n"
            "  Link: https://github.com/lucas/k8s-autoscaler\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="multi_projects.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert len(profile.projects) == 3
        p_names = [p.project_name for p in profile.projects]
        assert any("Raft Consensus" in n for n in p_names)
        assert any("FastKV" in n for n in p_names)
        assert any("AutoScaler" in n for n in p_names)
        urls = [p.project_url for p in profile.projects]
        assert "https://github.com/lucas/raft-rs" in urls
        assert "https://github.com/lucas/fastkv" in urls
        assert "https://github.com/lucas/k8s-autoscaler" in urls

    def test_different_skill_group_headings(self):
        """Skills grouped with alternative subcategory labels and colons."""
        text = (
            "Maya Lin\n"
            "maya@code.dev\n\n"
            "TECHNICAL SKILLS\n"
            "Core Languages: Python, Go, C++\n"
            "Data & Storage: PostgreSQL, Redis, Apache Kafka\n"
            "Cloud & Deployment: AWS, Docker, Kubernetes, CI/CD\n\n"
            "EDUCATION\n"
            "B.S. in Computer Science | UCLA | 2018 - 2022\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="skills_grouped.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        skill_names = {s.name for s in profile.skills}
        assert "Python" in skill_names
        assert "Apache Kafka" in skill_names
        assert "Docker" in skill_names
        assert "PostgreSQL" in skill_names

    def test_unknown_section_headings_preserved(self):
        """Resume containing atypical sections (e.g. Patents, Volunteering)."""
        text = (
            "Dr. Alan Turing\n"
            "alan@cambridge.ac.uk\n\n"
            "TECHNICAL SKILLS\n"
            "Python, Cryptography\n\n"
            "PATENTS AND INVENTIONS\n"
            "US Patent 9876543: Electromechanical Cryptanalytic Computing Apparatus.\n\n"
            "COMMUNITY AND VOLUNTEERING\n"
            "Code Mentor for National Computing Society 2021 - Present.\n\n"
            "EDUCATION\n"
            "Ph.D. in Mathematics | Cambridge University | 1938\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="unknown_headings.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        unknown_secs = [s for s in profile.sections if s.section_type == SectionType.UNKNOWN]
        assert len(unknown_secs) >= 2
        unknown_headings = [s.original_heading for s in unknown_secs if s.original_heading]
        assert any("PATENTS" in h for h in unknown_headings)
        assert any("COMMUNITY" in h for h in unknown_headings)

    def test_unusual_date_formats(self):
        """Date ranges formatted as '05/2019 - 08/2022' or 'Summer 2020'."""
        text = (
            "Rachel Green\n"
            "rachel@green.net\n\n"
            "TECHNICAL SKILLS\n"
            "Python, Django\n\n"
            "WORK EXPERIENCE\n"
            "Software Engineer at Metro Corp | 2018 - 2021\n"
            "- Maintained core database integrations.\n\n"
            "EDUCATION\n"
            "Bachelor of Science in Computer Science\n"
            "NYU | 2014 - 2018\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="dates_variant.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert profile.experience[0].employment_start == "2018"
        assert profile.experience[0].employment_end == "2021"
        assert profile.education[0].start_date == "2014"
        assert profile.education[0].end_date == "2018"


# ---------------------------------------------------------------------------
# 2. Edge Cases & Resilience
# ---------------------------------------------------------------------------

class TestResumeEdgeCases:
    """Rigorous tests covering corrupted, empty, encrypted, scanned, and malformed inputs."""

    def test_empty_pdf_rejection(self):
        """Empty 0-byte input rejected cleanly with FAILED status."""
        profile = process_resume_pipeline(b"", filename="empty.pdf", raise_on_error=False)
        assert profile.status == ProfileProcessingStatus.FAILED
        assert profile.page_count == 0
        assert len(profile.skills) == 0
        assert any("empty" in w.lower() for w in profile.warnings)

    def test_corrupted_pdf_rejection(self):
        """Malformed PDF signature rejected with FAILED status."""
        corrupted_bytes = b"%PDF-1.4\ncorrupted content that cannot be parsed by mupdf"
        profile = process_resume_pipeline(corrupted_bytes, filename="corrupt.pdf", raise_on_error=False)
        assert profile.status == ProfileProcessingStatus.FAILED
        assert len(profile.warnings) >= 1

    def test_encrypted_pdf_rejection(self):
        """Password-protected PDF rejected with clear error message."""
        enc_pdf = make_pdf(["Confidential Candidate Resume\nTop Secret Experience"], password="SecretPassword123")
        profile = process_resume_pipeline(enc_pdf, filename="encrypted.pdf", raise_on_error=False)
        assert profile.status == ProfileProcessingStatus.FAILED
        assert any("password" in w.lower() or "encrypted" in w.lower() for w in profile.warnings)

    def test_scanned_image_only_pdf_requires_ocr(self):
        """PDF with text character count below OCR threshold (<80 chars) flags PARTIAL status."""
        short_scanned_text = "Scan Doc"  # 8 characters, well below 80
        pdf = make_pdf([short_scanned_text])
        profile = process_resume_pipeline(pdf, filename="scanned_resume.pdf")

        assert profile.status == ProfileProcessingStatus.PARTIAL
        assert profile.requires_ocr is True
        assert profile.ocr_warning is not None
        assert any("OCR" in w for w in profile.warnings)

    def test_very_short_resume_passes_threshold(self):
        """Resume with just enough text to exceed OCR threshold (>80 non-ws chars) parses safely."""
        text = (
            "Jane Doe | jane@doe.com | (555) 123-4567\n\n"
            "TECHNICAL SKILLS\nPython, SQL, Docker, Linux\n\n"
            "WORK EXPERIENCE\nDeveloper at TechCorp\n\n"
            "EDUCATION\nB.S. at MIT | 2020\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="short_resume.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert profile.requires_ocr is False
        assert len(profile.skills) >= 2
        assert profile.education[0].institution == "MIT"

    def test_large_multipage_resume(self):
        """5-page extensive resume parses accurately with correct page count and metrics."""
        pages = [
            f"Candidate MultiPage — Page {i + 1}\n\n"
            f"TECHNICAL SKILLS (Continued {i + 1})\n"
            f"Python, Go, Docker, Kubernetes, AWS\n\n"
            f"WORK EXPERIENCE\n"
            f"Senior Architect {i + 1} at Cloud Tech {i + 1} | 202{i} - Present\n"
            f"- Managed cluster deployment for service tier {i + 1}.\n"
            for i in range(5)
        ]
        pdf = make_pdf(pages)
        profile = process_resume_pipeline(pdf, filename="multipage.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert profile.page_count == 5
        assert profile.raw_character_count > 500
        assert profile.cleaned_character_count > 500

    def test_missing_institution_remains_none(self):
        """Missing academic institution remains strictly None (zero hallucination)."""
        text = (
            "John Doe | john@doe.com | Austin, TX\n\n"
            "TECHNICAL SKILLS\nPython, PostgreSQL, Docker\n\n"
            "EDUCATION\n"
            "Bachelor of Science in Software Engineering\n"
            "2017 - 2021\n"
            "GPA: 3.9 / 4.0\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="missing_inst.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert len(profile.education) == 1
        assert profile.education[0].institution is None
        assert profile.education[0].degree == "Bachelor of Science"
        assert profile.education[0].confidence == "AMBIGUOUS"
        assert profile.education[0].is_ambiguous is True

    def test_missing_company_name_remains_none(self):
        """Missing company header marks experience record as AMBIGUOUS with company=None."""
        text = (
            "Jane Doe\n\n"
            "TECHNICAL SKILLS\nPython\n\n"
            "WORK EXPERIENCE\n"
            "- Built high-throughput backend services without formal company header.\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="missing_company.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        if profile.experience:
            assert profile.experience[0].company is None
            assert profile.experience[0].is_ambiguous is True

    def test_ambiguous_dates_handling(self):
        """Experience with missing dates leaves employment_start/end as None."""
        text = (
            "Taylor Swift\n\n"
            "TECHNICAL SKILLS\nSwift, Python\n\n"
            "WORK EXPERIENCE\n"
            "iOS Engineer at MobileTech\n"
            "- Developed native iOS applications using Swift.\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="no_dates.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert len(profile.experience) >= 1
        assert profile.experience[0].employment_start is None
        assert profile.experience[0].employment_end is None

    def test_duplicate_skills_deduplicated(self):
        """Duplicate skill mentions within skills section are deduplicated cleanly."""
        text = (
            "Alex\n\n"
            "TECHNICAL SKILLS\n"
            "Python, python, PYTHON, Docker, docker, DOCKER\n\n"
            "EDUCATION\n"
            "B.S. at MIT | 2020\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="duplicates.pdf")

        skill_names = [s.name for s in profile.skills]
        assert skill_names.count("Python") == 1
        assert skill_names.count("Docker") == 1

    def test_github_url_does_not_fabricate_git_skill(self):
        """GitHub link in contact or project section must never produce a false Git skill."""
        text = (
            "Alex Coder\n"
            "Portfolio: https://github.com/alexcoder/sample-project\n\n"
            "TECHNICAL SKILLS\n"
            "Python, SQL\n\n"
            "EDUCATION\n"
            "B.S. at MIT | 2020\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="github_url.pdf")

        skill_names = {s.name for s in profile.skills}
        assert "Python" in skill_names
        assert "SQL" in skill_names
        assert "Git" not in skill_names

    def test_unusual_characters_and_formatting(self):
        """Resilience against unusual Unicode characters, multiple blank lines, and tabs."""
        text = (
            "Candidate\tName\r\n"
            "Email:\talex@domain.com\r\n\r\n\r\n"
            "TECHNICAL SKILLS\r\n"
            "- Python\r\n"
            "- PostgreSQL\r\n\r\n"
            "EDUCATION\r\n"
            "Bachelor of Science | Stanford University | 2020\r\n"
        )
        pdf = make_pdf([text])
        profile = process_resume_pipeline(pdf, filename="unusual_chars.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert "\r" not in profile.cleaned_text
        assert len(profile.skills) >= 2


# ---------------------------------------------------------------------------
# 3. Data Integrity & Immutability
# ---------------------------------------------------------------------------

class TestDataIntegrityAndImmutability:
    """Verifies that data models maintain absolute fidelity and immutability."""

    @pytest.fixture
    def sample_profile(self) -> ResumeProfile:
        text = (
            "Alex Smith\n"
            "alex@smith.com\n\n"
            "TECHNICAL SKILLS\n"
            "Python, Go, Docker\n\n"
            "WORK EXPERIENCE\n"
            "Software Engineer at InnovateTech | 2020 - Present\n"
            "- Built backend services in Python and Go.\n\n"
            "EDUCATION\n"
            "Bachelor of Science in Computer Science\n"
            "MIT | 2016 - 2020\n"
        )
        pdf = make_pdf([text])
        return process_resume_pipeline(pdf, filename="integrity_test.pdf")

    def test_original_text_preserved_and_isolated(self, sample_profile):
        """Original PDF text is preserved exactly and kept separate from cleaned text."""
        assert sample_profile.original_text is not sample_profile.cleaned_text
        assert len(sample_profile.original_text) > 0
        assert len(sample_profile.cleaned_text) > 0

    def test_explicit_vs_mentioned_skills_distinguishable(self, sample_profile):
        """Skills explicitly listed in skills section have is_explicitly_listed=True."""
        for skill in sample_profile.skills:
            if skill.name in {"Python", "Go", "Docker"}:
                assert skill.is_explicitly_listed is True
                assert skill.source_section == "Technical Skills"

    def test_profile_immutability(self, sample_profile):
        """ResumeProfile model is frozen and rejects attribute mutation."""
        with pytest.raises(Exception):
            sample_profile.filename = "mutated.pdf"

        with pytest.raises(Exception):
            sample_profile.status = ProfileProcessingStatus.FAILED

    def test_json_roundtrip_serialization(self, sample_profile):
        """JSON serialization and deserialization retains all fields identically."""
        # 1. Serialize to raw JSON string via model_dump_json()
        json_str = sample_profile.model_dump_json()
        assert isinstance(json_str, str)

        # 2. Reconstruct directly from JSON string via ResumeProfile.model_validate_json()
        reconstructed = ResumeProfile.model_validate_json(json_str)
        assert reconstructed.filename == sample_profile.filename
        assert reconstructed.status == sample_profile.status
        assert len(reconstructed.skills) == len(sample_profile.skills)
        assert len(reconstructed.education) == len(sample_profile.education)
        assert len(reconstructed.experience) == len(sample_profile.experience)
        assert len(reconstructed.projects) == len(sample_profile.projects)
        assert len(reconstructed.sections) == len(sample_profile.sections)
        assert reconstructed == sample_profile

        # 3. Reconstruct from dictionary via ResumeProfile.model_validate()
        parsed_dict = json.loads(json_str)
        assert parsed_dict["filename"] == "integrity_test.pdf"
        assert parsed_dict["status"] == "SUCCESS"
        reconstructed_from_dict = ResumeProfile.model_validate(parsed_dict)
        assert reconstructed_from_dict == sample_profile


# ---------------------------------------------------------------------------
# 4. Privacy, Security & In-Memory Guarantees
# ---------------------------------------------------------------------------

class TestPrivacyAndReliability:
    """Verifies that no resume files are saved to disk, API keys are safe, and memory is cleanly released."""

    def test_no_disk_files_created_during_pipeline(self):
        """Ensure no resume or temporary PDF files are created on disk during processing."""
        text = (
            "Privacy Test Candidate\n"
            "privacy@secure.internal\n\n"
            "TECHNICAL SKILLS\nPython\n\n"
            "EDUCATION\nB.S. at MIT | 2020\n"
        )
        pdf = make_pdf([text])
        stream = io.BytesIO(pdf)

        # Snapshot workspace directory files
        workspace_files_before = set(os.listdir("."))

        profile = process_resume_pipeline(stream, filename="in_memory_only.pdf")

        workspace_files_after = set(os.listdir("."))

        # Verify zero new files created in workspace root
        new_files = workspace_files_after - workspace_files_before
        assert len(new_files) == 0, f"Unexpected files created on disk: {new_files}"
        assert profile.status == ProfileProcessingStatus.SUCCESS

    def test_pipeline_zero_external_network_activity(self, monkeypatch):
        """Verify resume pipeline makes 0 external network/socket connection attempts."""
        import socket
        connection_attempts = []

        def mock_connect(self, address):
            connection_attempts.append(address)
            raise RuntimeError(f"Unexpected external socket connection to {address}")

        monkeypatch.setattr(socket.socket, "connect", mock_connect)

        # Process a multi-section resume rich in external URLs and links
        rich_text = (
            "Taylor Swift\n"
            "taylor@swift.io | https://github.com/taylor/repo | https://taylor.dev\n\n"
            "TECHNICAL SKILLS\nPython, Docker, AWS, Kubernetes\n\n"
            "WORK EXPERIENCE\nSenior Platform Engineer at CloudTech | 2020 - Present\n- Deployed services.\n\n"
            "EDUCATION\nB.S. in CS | Stanford University | 2020\n\n"
            "PROJECTS\n- OpenSource: https://github.com/taylor/opensource\n"
        )
        pdf = make_pdf([rich_text])
        profile = process_resume_pipeline(pdf, filename="zero_network.pdf")

        assert profile.status == ProfileProcessingStatus.SUCCESS
        assert len(connection_attempts) == 0

    def test_streamlit_app_zero_external_api_calls(self):
        """Audit app.py and config.py ensuring zero outbound HTTP/AI client imports or invocations."""
        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()

        with open("config.py", "r", encoding="utf-8") as f:
            config_code = f.read()

        prohibited_clients = [
            "google.generativeai",
            "google.genai",
            "groq",
            "langchain",
            "langgraph",
            "urllib.request",
            "requests",
            "httpx",
            "aiohttp",
        ]

        for prohibited in prohibited_clients:
            assert f"import {prohibited}" not in app_code, f"Unauthorized client in app.py: {prohibited}"
            assert f"from {prohibited}" not in app_code, f"Unauthorized client in app.py: {prohibited}"
            assert f"import {prohibited}" not in config_code, f"Unauthorized client in config.py: {prohibited}"
            assert f"from {prohibited}" not in config_code, f"Unauthorized client in config.py: {prohibited}"

    def test_api_keys_not_exposed_in_config_representation(self, monkeypatch):
        """Config object safely hides raw API keys from representations."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestGeminiKey123")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_groq_key_456")
        cfg = load_config()
        cfg_repr = repr(cfg)

        # Verify keys are not leaked in repr
        assert "AIzaSyTestGeminiKey123" not in cfg_repr
        assert "gsk_test_groq_key_456" not in cfg_repr
        assert cfg.gemini_key_configured is True
        assert cfg.groq_key_configured is True
