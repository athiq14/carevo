"""
Comprehensive unit tests for SectionDetector and ParsedResumeSections.
Verifies canonical alias matching, case handling, unknown section preservation,
and false positive rejection on synthetic resume text.
"""
import pytest
from models.resume_sections import SectionType, ParsedResumeSections
from processors.section_detector import (
    CANONICAL_SECTIONS,
    SectionDetector,
    detect_resume_sections,
)


@pytest.fixture
def detector() -> SectionDetector:
    return SectionDetector()


def test_empty_input(detector: SectionDetector):
    """Verify empty input returns empty sections and all canonical sections as undetected."""
    res = detector.detect_sections("")
    assert isinstance(res, ParsedResumeSections)
    assert len(res.sections) == 0
    assert len(res.detected_section_names) == 0
    assert len(res.undetected_section_names) == len(CANONICAL_SECTIONS)


def test_uppercase_headings(detector: SectionDetector):
    """Verify standard uppercase headings are detected."""
    text = (
        "PROFESSIONAL SUMMARY\n"
        "Lead Backend Engineer with 8 years of experience.\n\n"
        "TECHNICAL SKILLS\n"
        "- Python, Docker, Kubernetes, PostgreSQL\n\n"
        "EDUCATION\n"
        "B.S. in Computer Science, Tech University"
    )
    res = detector.detect_sections(text)

    types = [s.section_type for s in res.sections]
    assert types == [
        SectionType.PROFESSIONAL_SUMMARY,
        SectionType.TECHNICAL_SKILLS,
        SectionType.EDUCATION,
    ]
    assert "Professional Summary" in res.detected_section_names
    assert "Technical Skills" in res.detected_section_names
    assert "Education" in res.detected_section_names


def test_mixed_case_headings(detector: SectionDetector):
    """Verify mixed-case and title-case headings are properly matched."""
    text = (
        "Work Experience\n"
        "Staff Engineer at Cloud Corp (2020 - Present)\n\n"
        "Projects\n"
        "- High-throughput messaging pipeline\n\n"
        "Certifications\n"
        "- AWS Certified Solutions Architect"
    )
    res = detector.detect_sections(text)

    types = [s.section_type for s in res.sections]
    assert types == [
        SectionType.WORK_EXPERIENCE,
        SectionType.PROJECTS,
        SectionType.CERTIFICATIONS,
    ]


def test_alternative_section_names(detector: SectionDetector):
    """Verify that common aliases map to the correct canonical SectionType."""
    text = (
        "Employment History\n"
        "Senior Developer at Systems Inc.\n\n"
        "Core Competencies\n"
        "- Distributed Systems, Microservices\n\n"
        "Academic Background\n"
        "Master of Science in Software Engineering\n\n"
        "Honors & Awards\n"
        "Employee of the Year 2023"
    )
    res = detector.detect_sections(text)

    types = [s.section_type for s in res.sections]
    assert types == [
        SectionType.WORK_EXPERIENCE,
        SectionType.TECHNICAL_SKILLS,
        SectionType.EDUCATION,
        SectionType.ACHIEVEMENTS,
    ]


def test_different_section_ordering(detector: SectionDetector):
    """Verify arbitrary section order is honored without re-sorting."""
    text = (
        "SKILLS\n"
        "- Python, Go\n\n"
        "EDUCATION\n"
        "B.S. Computer Engineering\n\n"
        "WORK EXPERIENCE\n"
        "Software Engineer at Enterprise Labs"
    )
    res = detector.detect_sections(text)

    types = [s.section_type for s in res.sections]
    assert types == [
        SectionType.TECHNICAL_SKILLS,
        SectionType.EDUCATION,
        SectionType.WORK_EXPERIENCE,
    ]


def test_contact_information_preamble_detected(detector: SectionDetector):
    """Verify that an unheaded header containing contact info is classified as Contact Information."""
    text = (
        "Jane Doe\n"
        "jane.doe@example.com | (555) 987-6543 | linkedin.com/in/janedoe\n\n"
        "SUMMARY\n"
        "Experienced Cloud Engineer."
    )
    res = detector.detect_sections(text)

    assert len(res.sections) == 2
    assert res.sections[0].section_type == SectionType.CONTACT_INFORMATION
    assert "jane.doe@example.com" in res.sections[0].content
    assert res.sections[0].original_heading is None

    assert res.sections[1].section_type == SectionType.PROFESSIONAL_SUMMARY
    assert res.sections[1].original_heading == "SUMMARY"


def test_unknown_section_preserved(detector: SectionDetector):
    """Verify unclassified custom headings are preserved under SectionType.UNKNOWN."""
    text = (
        "WORK EXPERIENCE\n"
        "Senior Systems Engineer at Tech Inc.\n\n"
        "PATENT PORTFOLIO:\n"
        "US Patent 9,999,999 - Distributed Consensus Protocol\n\n"
        "VOLUNTEERING\n"
        "STEM Mentor for high school robotics club"
    )
    res = detector.detect_sections(text)

    types = [s.section_type for s in res.sections]
    assert types == [
        SectionType.WORK_EXPERIENCE,
        SectionType.UNKNOWN,
        SectionType.OTHER_INFORMATION,
    ]

    unknown_sec = res.sections[1]
    assert unknown_sec.original_heading == "PATENT PORTFOLIO:"
    assert "US Patent 9,999,999" in unknown_sec.content


def test_false_positive_sentence_not_treated_as_heading(detector: SectionDetector):
    """Verify inline sentences containing keywords are not mistaken for headings."""
    text = (
        "PROFESSIONAL SUMMARY\n"
        "I have extensive experience with Python, Docker and microservices architecture.\n"
        "My education includes ongoing self-directed study and certifications."
    )
    res = detector.detect_sections(text)

    # Should only produce 1 section, not 3!
    assert len(res.sections) == 1
    assert res.sections[0].section_type == SectionType.PROFESSIONAL_SUMMARY
    assert "extensive experience with Python" in res.sections[0].content


def test_line_numbering_accuracy(detector: SectionDetector):
    """Verify start_line and end_line are tracked accurately."""
    text = (
        "SUMMARY\n"
        "First line of summary.\n"
        "Second line of summary.\n\n"
        "SKILLS\n"
        "- Python\n"
        "- SQL"
    )
    res = detector.detect_sections(text)

    assert len(res.sections) == 2

    # Section 1: SUMMARY (line 1), lines 2, 3
    assert res.sections[0].start_line == 1
    assert res.sections[0].end_line == 3

    # Section 2: SKILLS (line 5), lines 6, 7
    assert res.sections[1].start_line == 5
    assert res.sections[1].end_line == 7


def test_missing_sections_reported(detector: SectionDetector):
    """Verify undetected_section_names accurately lists canonical sections not present."""
    text = (
        "SKILLS\n"
        "- Python\n\n"
        "PROJECTS\n"
        "- Web Scraper"
    )
    res = detector.detect_sections(text)

    assert "Technical Skills" in res.detected_section_names
    assert "Projects" in res.detected_section_names

    # Check that other canonical sections appear in undetected list
    assert "Work Experience" in res.undetected_section_names
    assert "Education" in res.undetected_section_names
    assert "Certifications" in res.undetected_section_names


def test_original_and_cleaned_text_preservation(detector: SectionDetector):
    """Verify original_text and cleaned_text are stored without modification."""
    orig = "Raw   Original   Text"
    cleaned = "Cleaned Original Text"
    res = detector.detect_sections(cleaned_text=cleaned, original_text=orig)

    assert res.original_text == orig
    assert res.cleaned_text == cleaned

