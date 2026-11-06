"""
Unit tests for ResumeTextCleaner.
Verifies line ending normalization, bullet standardization, whitespace handling,
and data preservation on synthetic resume text.
"""
import pytest
from processors.text_cleaner import ResumeTextCleaner, clean_resume_text


def test_empty_input():
    """Verify empty or whitespace-only strings return an empty string."""
    assert clean_resume_text("") == ""
    assert clean_resume_text("   \n\t   ") == ""


def test_line_ending_normalization():
    """Verify Windows (CRLF), classic Mac (CR), and Unix (LF) line endings are unified."""
    raw_mixed = "Line 1\r\nLine 2\rLine 3\nLine 4"
    cleaned = clean_resume_text(raw_mixed)
    assert cleaned == "Line 1\nLine 2\nLine 3\nLine 4"
    assert "\r" not in cleaned


def test_whitespace_normalization():
    """Verify internal excessive spaces and tabs are collapsed without losing words."""
    raw_text = "Software    Engineer   at     Tech   Corp"
    cleaned = clean_resume_text(raw_text)
    assert cleaned == "Software Engineer at Tech Corp"


def test_excessive_newlines_normalization():
    """Verify 3+ consecutive newlines are normalized to at most two (single blank line)."""
    raw_text = "Section One\n\n\n\n\nSection Two\n\n\nSection Three"
    cleaned = clean_resume_text(raw_text)
    assert cleaned == "Section One\n\nSection Two\n\nSection Three"


def test_bullet_normalization():
    """Verify various Unicode bullet characters are standardized to '- '."""
    raw_bullets = (
        "• Developed scalable APIs\n"
        "● Managed database migrations\n"
        "▪ Optimized CI/CD pipeline\n"
        "* Mentored junior engineers\n"
        "– Led incident post-mortems\n"
        "— Configured Kubernetes clusters"
    )
    cleaned = clean_resume_text(raw_bullets)
    expected = (
        "- Developed scalable APIs\n"
        "- Managed database migrations\n"
        "- Optimized CI/CD pipeline\n"
        "- Mentored junior engineers\n"
        "- Led incident post-mortems\n"
        "- Configured Kubernetes clusters"
    )
    assert cleaned == expected


def test_preservation_of_emails_and_urls():
    """Verify that email addresses and web URLs remain intact."""
    raw_contact = (
        "Contact: alex.dev@enterprise-ai.co.uk\n"
        "Portfolio: https://alex-dev.github.io/projects?tab=overview\n"
        "LinkedIn: https://www.linkedin.com/in/alex-systems-engineer/"
    )
    cleaned = clean_resume_text(raw_contact)
    assert "alex.dev@enterprise-ai.co.uk" in cleaned
    assert "https://alex-dev.github.io/projects?tab=overview" in cleaned
    assert "https://www.linkedin.com/in/alex-systems-engineer/" in cleaned


def test_preservation_of_phone_numbers_and_punctuation():
    """Verify that phone numbers and meaningful symbols (+, #, -, (), :) are preserved."""
    raw_line = "Phone: +1 (555) 019-2834 | C# & C++ Developer [Senior Level]: Score: 98%"
    cleaned = clean_resume_text(raw_line)
    assert "+1 (555) 019-2834" in cleaned
    assert "C# & C++ Developer [Senior Level]: Score: 98%" in cleaned


def test_control_character_removal():
    """Verify non-printable control characters are purged while retaining newlines and text."""
    raw_text = "Clean\x00Header\x07Text\x1bWith\x0cFormFeed"
    cleaned = clean_resume_text(raw_text)
    assert cleaned == "CleanHeaderTextWithFormFeed"


def test_original_text_separate():
    """Verify that raw text is not mutated when cleaning."""
    original = "  Original   Text   \r\n"
    cleaned = clean_resume_text(original)
    assert cleaned == "Original Text"
    assert original == "  Original   Text   \r\n"

